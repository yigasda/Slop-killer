// ====================================================================
// Slop Killer — repeated-phrase detector & blocker for SillyTavern
// Auto-aggregates overused n-grams from the current chat, injects a
// "avoid these phrases" note before generation, and highlights repeats.
// ====================================================================

const MODULE_NAME = "slop_killer";

const PASTEL_CHIPS = [
    "#ffadad", "#ffb347", "#ffd6a5", "#fdffb6",
    "#caffbf", "#9bf6ff", "#a0c4ff", "#bdb2ff",
    "#ffc6ff", "#ffb3c6", "#b5ead7", "#f9c74f",
];

const TABS = ["banned", "global", "detect", "settings"];
const SEARCH_THRESHOLD = 10;   // show search/sort UI when list has ≥ this many items
const VALID_THEMES = new Set(["mono", "rose", "ivory", "sky"]);

const DEFAULT_SETTINGS = Object.freeze({
    enabled: true,
    theme: "mono",      // mono | rose | ivory | sky
    minN: 2,            // shortest phrase length (words)
    maxN: 4,            // longest phrase length (words)
    threshold: 3,       // occurrences needed to count as slop
    scanDepth: 50,      // how many recent AI messages to scan
    injectEnabled: true,
    maxInject: 12,      // max phrases sent to the model
    injectTemplate:
        "[System note — writing variety]\n" +
        "STRICTLY FORBIDDEN — Under no circumstances, never write these phrases or any close variation, under any circumstance: {{banned}}.\n" +
        "Also avoid overusing these repeated phrases, or close paraphrases: {{slop}}.\n" +
        "Vary your sentence structure and reach for fresh wording and new sensory detail instead.",
    highlightEnabled: true,
    highlightColor: "#ff6b6b",
    penaltyEnabled: true,
    penaltyBoost: 0.3,  // added to freq/pres penalty on OpenAI-compatible backends
    autoReroll: true,   // re-generate (via continue) when a banned phrase appears
    rerollMax: 3,       // max continue attempts per message
    characters: {},     // charName -> { banned: [], allowed: [] }
    global: { banned: [], allowed: [] },   // applied across every character
    activeTab: "banned",                   // remembered between sessions
    customStopwords: "",                   // user stopwords (comma/newline separated)
    detectPresets: {},                     // name -> { minN, maxN, threshold, scanDepth, customStopwords }
});

// Backends that support freq_pen_openai / pres_pen_openai in oai_settings.
const PENALTY_BACKENDS = new Set([
    "openai", "deepseek", "custom", "openrouter", "mistralai",
    "groq", "azure_openai", "xai", "aimlapi", "fireworks",
    "siliconflow", "workers_ai", "chutes", "nanogpt", "moonshot",
]);

// Saved penalty values while a generation is in-flight; restored on GENERATION_ENDED.
let _penaltyRestore = null;

// Auto-reroll bookkeeping.
let _rerollBusy = false;            // true while we are driving continue retries
let _lastFreshId = null;            // id of the most recent freshly-generated reply
const _rerollCount = new Map();     // mesId -> attempts already spent

// Search + sort state per scope (persists across re-renders within a session).
const _filter = {
    charBanned:   { q: "", sort: "added" },
    globalBanned: { q: "", sort: "added" },
};

// Old default templates — auto-upgraded to the current default on load.
const LEGACY_INJECT_TEMPLATES = new Set([
    "[System note — writing variety] Avoid reusing the following overused phrases, or any close paraphrase, in your next reply: {{phrases}}. Vary your sentence structure and reach for fresh wording and new sensory detail instead.",
    "[System note — writing variety]\n" +
        "STRICTLY FORBIDDEN — never write these phrases or any close variation, under any circumstance: {{banned}}.\n" +
        "Also avoid overusing these repeated phrases, or close paraphrases: {{slop}}.\n" +
        "Vary your sentence structure and reach for fresh wording and new sensory detail instead.",
]);

// English stopwords — phrases made of ONLY these are ignored.
const STOPWORDS = new Set((
    "a an and the of to in on at for with as is are was were be been being it its " +
    "this that these those i you he she they we him her them his hers their our your " +
    "my me us do did does have has had not no so but or if then than too very can " +
    "could will would should may might must just about into over under again more " +
    "most some any all out up down off there here what which who whom from by what's " +
    "i'm you're he's she's they're we're it's don't didn't won't can't"
).split(/\s+/));

// Common Korean function words / fillers — ignored in detection (parallels STOPWORDS).
const KO_STOPWORDS = new Set((
    "그리고 그러나 하지만 그래서 그런데 그러면 그러니까 그리하여 그래도 또한 " +
    "그 이 저 것 거 수 등 때 곳 점 채 뿐 만큼 대로 " +
    "나 너 우리 너희 당신 그대 자기 저희 " +
    "그녀 그들 이것 그것 저것 여기 거기 저기 " +
    "정말 진짜 너무 매우 아주 더 덜 좀 잘 막 또 다시 그냥 마치 거의 " +
    "이런 그런 저런 어떤 무슨 모든 여러 가장"
).split(/\s+/).filter(Boolean));

// Korean particles / endings stripped (longest match first) to group inflected
// variants ("그녀는"/"그녀가" → "그녀"). Heuristic: only when the remaining stem
// is ≥ 2 Hangul chars, so short nouns like "바다" stay intact.
const KO_SUFFIXES = [
    "이라고", "이라는", "으로부터", "로부터", "에게서", "한테서", "으로서", "으로써", "에서는",
    "었습니다", "았습니다", "였습니다", "습니다",
    "라고", "라는", "에게", "한테", "께서", "처럼", "보다", "마다", "조차", "마저",
    "이라도", "라도", "까지", "부터", "에서", "으로",
    "었다", "았다", "였다", "는다", "겠다", "는데", "은데", "면서", "으며", "지만",
    "어서", "아서", "니까", "구나", "네요", "어요", "아요", "에요", "예요",
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "과", "와", "로", "나", "고", "며", "다", "요",
].sort((a, b) => b.length - a.length);

// ====================================================================
// Context helpers
// ====================================================================
function ctx() { return SillyTavern.getContext(); }

function normalizePhrase(p) { return String(p).toLowerCase().trim(); }

function getSettings() {
    const { extensionSettings } = ctx();
    if (!extensionSettings[MODULE_NAME]) {
        extensionSettings[MODULE_NAME] = structuredClone(DEFAULT_SETTINGS);
    }
    const s = extensionSettings[MODULE_NAME];
    for (const k of Object.keys(DEFAULT_SETTINGS)) {
        if (!Object.hasOwn(s, k)) s[k] = structuredClone(DEFAULT_SETTINGS[k]);
    }
    if (LEGACY_INJECT_TEMPLATES.has(s.injectTemplate)) s.injectTemplate = DEFAULT_SETTINGS.injectTemplate;
    if (!VALID_THEMES.has(s.theme)) s.theme = DEFAULT_SETTINGS.theme;
    return s;
}

function save() { ctx().saveSettingsDebounced(); }

function getCurrentCharName() {
    const c = ctx();
    return c.name2 || c.characters?.[c.characterId]?.name || null;
}

function getCharData(name) {
    if (!name) return { banned: [], allowed: [] };
    const s = getSettings();
    if (!s.characters[name]) s.characters[name] = { banned: [], allowed: [] };
    const d = s.characters[name];
    if (!Array.isArray(d.banned)) d.banned = [];
    if (!Array.isArray(d.allowed)) d.allowed = [];
    return d;
}

function getGlobal() {
    const s = getSettings();
    if (!s.global || typeof s.global !== "object") s.global = { banned: [], allowed: [] };
    if (!Array.isArray(s.global.banned)) s.global.banned = [];
    if (!Array.isArray(s.global.allowed)) s.global.allowed = [];
    return s.global;
}

// Global ∪ character, deduped and normalized. Used everywhere a "effective"
// ban/allow list is needed (injection, highlight, reroll, ranking).
function mergedBanned() {
    const cd = getCharData(getCurrentCharName());
    const g = getGlobal();
    const seen = new Set(), out = [];
    for (const p of [...g.banned, ...cd.banned]) {
        const k = normalizePhrase(p);
        if (k && !seen.has(k)) { seen.add(k); out.push(k); }
    }
    return out;
}

function mergedAllowed() {
    const cd = getCharData(getCurrentCharName());
    const g = getGlobal();
    const seen = new Set(), out = [];
    for (const p of [...g.allowed, ...cd.allowed]) {
        const k = normalizePhrase(p);
        if (k && !seen.has(k)) { seen.add(k); out.push(k); }
    }
    return out;
}

// ====================================================================
// Text → n-grams → counts (recomputed live, never stored)
// ====================================================================
function tokenize(text) {
    return text
        .replace(/```[\s\S]*?```/g, " ")          // drop code blocks
        .replace(/[*_`~>#\[\]()]/g, " ")           // strip markdown chars
        .split(/\s+/)
        .map(w => w.replace(/^[^\p{L}\p{N}']+|[^\p{L}\p{N}']+$/gu, "").toLowerCase())
        .filter(Boolean);
}

function isHangulToken(w) { return /^[가-힣]+$/.test(w); }

// Strip one trailing particle/ending to normalize Korean inflected variants.
function stripKoSuffix(w) {
    if (!isHangulToken(w)) return w;
    for (const suf of KO_SUFFIXES) {
        if (w.length > suf.length && w.length - suf.length >= 2 && w.endsWith(suf)) {
            return w.slice(0, -suf.length);
        }
    }
    return w;
}

// Effective stopword set = English ∪ Korean ∪ custom.
function effectiveStopwords() {
    const s = getSettings();
    const set = new Set(STOPWORDS);
    for (const w of KO_STOPWORDS) set.add(w);
    if (s.customStopwords) {
        for (const w of String(s.customStopwords).split(/[\s,]+/)) {
            const k = w.trim().toLowerCase();
            if (k) set.add(k);
        }
    }
    return set;
}

// Count n-grams across recent AI messages. When Korean mode is on, n-grams are
// keyed by their suffix-stripped (normalized) form so inflected variants merge,
// while the most frequent surface form is kept as the display representative.
function computeCounts() {
    const s = getSettings();
    const stop = effectiveStopwords();
    const chat = ctx().chat || [];
    const msgs = chat.filter(m =>
        m && !m.is_user && !m.is_system && typeof m.mes === "string" && m.mes.trim()
    ).slice(-s.scanDepth);

    const agg = new Map();   // normKey -> { count, surfaces: Map<surface, count> }
    for (const m of msgs) {
        const surf = tokenize(m.mes);
        const norm = surf.map(stripKoSuffix);
        for (let n = s.minN; n <= s.maxN; n++) {
            for (let i = 0; i + n <= surf.length; i++) {
                const normGram = norm.slice(i, i + n);
                if (normGram.filter(w => !stop.has(w)).length < 2) continue;
                const key = normGram.join(" ");
                let e = agg.get(key);
                if (!e) { e = { count: 0, surfaces: new Map() }; agg.set(key, e); }
                e.count++;
                const sf = surf.slice(i, i + n).join(" ");
                e.surfaces.set(sf, (e.surfaces.get(sf) || 0) + 1);
            }
        }
    }

    const counts = new Map();
    for (const e of agg.values()) {
        let rep = "", best = -1;
        for (const [sf, c] of e.surfaces) if (c > best) { best = c; rep = sf; }
        counts.set(rep, (counts.get(rep) || 0) + e.count);
    }
    return counts;
}

// Direct regex count of a (possibly long) phrase in recent AI messages.
// Used for the "현재 채팅 등장 횟수" sort, where a banned phrase may be longer
// than maxN and therefore absent from the n-gram count map.
function countOccurrencesInChat(phrase) {
    const s = getSettings();
    const chat = ctx().chat || [];
    const msgs = chat.filter(m =>
        m && !m.is_user && !m.is_system && typeof m.mes === "string" && m.mes.trim()
    ).slice(-s.scanDepth);
    const needle = String(phrase).trim();
    if (!needle) return 0;
    const re = new RegExp(escapeRe(needle), "gi");
    let n = 0;
    for (const m of msgs) {
        const matches = m.mes.match(re);
        if (matches) n += matches.length;
    }
    return n;
}

// True if two phrases belong to the same repeated run (containment or a
// boundary shift of >= 2 shared words — e.g. sliding-window fragments).
function overlaps(a, b) {
    if ((" " + a + " ").includes(" " + b + " ") || (" " + b + " ").includes(" " + a + " ")) return true;
    const aw = a.split(" "), bw = b.split(" ");
    for (let k = Math.min(aw.length, bw.length) - 1; k >= 2; k--) {
        if (aw.slice(-k).join(" ") === bw.slice(0, k).join(" ")) return true;
        if (bw.slice(-k).join(" ") === aw.slice(0, k).join(" ")) return true;
    }
    return false;
}

// Raw n-grams above threshold, excluding allowed phrases (and anything that
// overlaps an allowed phrase, so the allow action clears the whole run). Sorted by count.
function aboveThreshold() {
    const s = getSettings();
    const allowed = mergedAllowed();
    return [...computeCounts().entries()]
        .filter(([p, c]) =>
            c >= s.threshold &&
            !allowed.includes(p) &&
            !allowed.some(a => overlaps(a, p)))
        .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length);
}

// Chain-merge consecutive sliding-window fragments back into their full run.
// Two grams that overlap by (shorter length − 1) words at the boundary are
// adjacent windows of the same passage, so stitch them into one phrase.
function mergeChains(entries) {
    const items = entries.map(([p, c]) => ({ words: p.split(" "), count: c }));
    let merged = true;
    while (merged) {
        merged = false;
        outer:
        for (let i = 0; i < items.length; i++) {
            for (let j = 0; j < items.length; j++) {
                if (i === j) continue;
                const A = items[i].words, B = items[j].words;
                const k = Math.min(A.length, B.length) - 1;
                if (k >= 2 && A.slice(-k).join(" ") === B.slice(0, k).join(" ")) {
                    items[i].words = A.concat(B.slice(k));
                    items[i].count = Math.min(items[i].count, items[j].count);
                    items.splice(j, 1);
                    merged = true;
                    break outer;
                }
            }
        }
    }
    return items.map(it => ({ phrase: it.words.join(" "), count: it.count }));
}

// Deduped representatives (one per repeated run) — for display + injection.
// Already-banned phrases (and fragments overlapping them) are dropped so the
// ranking only shows phrases that haven't been categorized yet.
function rankSlop() {
    const banned = mergedBanned();
    const isBanned = p => banned.includes(p) || banned.some(b => overlaps(b, p));
    const merged = mergeChains(aboveThreshold())
        .sort((a, b) => b.count - a.count || b.phrase.length - a.phrase.length);
    const kept = [];
    for (const { phrase, count } of merged) {
        if (isBanned(phrase)) continue;
        if (kept.some(k => overlaps(k.phrase, phrase))) continue;
        kept.push({ phrase, count });
    }
    return kept;
}

// Injection lists, split by source: manual banned (strict) vs auto-detected
// slop (soft). Deduped against each other; banned takes priority in the cap.
function injectionLists(cap) {
    const seen = new Set();
    const collect = (src, into) => {
        for (const p of src) {
            const k = normalizePhrase(p);
            if (k && !seen.has(k)) { seen.add(k); into.push(k); }
        }
    };
    const banned = [];
    collect(mergedBanned(), banned);
    const slop = [];
    collect(rankSlop().map(x => x.phrase), slop);
    if (!cap) return { banned, slop };
    const bannedCapped = banned.slice(0, cap);
    const slopCapped = slop.slice(0, Math.max(0, cap - bannedCapped.length));
    return { banned: bannedCapped, slop: slopCapped };
}

// Renders an injection template. {{banned}} / {{slop}} / {{phrases}} are
// substituted with quoted lists; a line is dropped entirely if its placeholder
// list is empty. Templates with no placeholder get the merged list appended.
function buildInjectionText(template, banned, slop) {
    const fmt = arr => arr.map(p => `"${p}"`).join(", ");
    const all = [...banned, ...slop];
    const out = [];
    for (const line of template.split("\n")) {
        if (line.includes("{{banned}}") && banned.length === 0) continue;
        if (line.includes("{{slop}}") && slop.length === 0) continue;
        if (line.includes("{{phrases}}") && all.length === 0) continue;
        out.push(line
            .replaceAll("{{banned}}", fmt(banned))
            .replaceAll("{{slop}}", fmt(slop))
            .replaceAll("{{phrases}}", fmt(all)));
    }
    let text = out.join("\n").trim();
    if (!/\{\{(banned|slop|phrases)\}\}/.test(template) && all.length) {
        text = `${text} ${fmt(all)}`.trim();
    }
    return text;
}

// Highlight set = manual banned ∪ all above-threshold n-grams (broad coverage
// so a whole repeated passage gets colored, not just one fragment).
function highlightPhrases() {
    const out = [];
    const push = (p) => {
        const k = normalizePhrase(p);
        if (k && !out.includes(k)) out.push(k);
    };
    mergedBanned().forEach(push);
    aboveThreshold().forEach(([p]) => push(p));
    return out;
}

// ====================================================================
// Penalty boost — temporarily raises freq/pres penalty on supported backends
// ====================================================================
function boostPenalty(s) {
    const oai = SillyTavern.getContext().chatCompletionSettings;
    if (!oai) return;
    const source = oai.chat_completion_source || "";
    if (!PENALTY_BACKENDS.has(source)) return;
    _penaltyRestore = { oai, freq: oai.freq_pen_openai, pres: oai.pres_pen_openai };
    oai.freq_pen_openai = Math.min(2, (oai.freq_pen_openai || 0) + s.penaltyBoost);
    oai.pres_pen_openai = Math.min(2, (oai.pres_pen_openai || 0) + s.penaltyBoost);
}

function restorePenalty() {
    if (!_penaltyRestore) return;
    _penaltyRestore.oai.freq_pen_openai = _penaltyRestore.freq;
    _penaltyRestore.oai.pres_pen_openai = _penaltyRestore.pres;
    _penaltyRestore = null;
}

// ====================================================================
// Auto-reroll — when a banned phrase appears, truncate at the sentence
// boundary before it and continue-generate from there (cheaper than a full
// swipe; keeps all the good text preceding the offending passage).
// ====================================================================

// Index of the earliest banned-phrase match in text, or -1.
function earliestBannedPos(text, phrases) {
    const re = buildPhraseRegex(phrases);
    if (!re) return -1;
    re.lastIndex = 0;
    const m = re.exec(text);
    return m ? m.index : -1;
}

// Index just after the last sentence/line terminator before `idx`, or -1.
function lastBoundaryBefore(text, idx) {
    const slice = text.slice(0, idx);
    const re = /[.!?…]["'”’)\]]*\s+|\n+/g;
    let best = -1, m;
    while ((m = re.exec(slice)) !== null) best = m.index + m[0].length;
    return best;
}

// Returns the truncated message kept before the offending passage, or null
// if there is nothing worth keeping (banned phrase at the very start).
function truncateForReroll(text, phrases) {
    const pos = earliestBannedPos(text, phrases);
    if (pos < 0) return null;
    const boundary = lastBoundaryBefore(text, pos);
    const cut = boundary > 0 ? boundary : pos;   // fall back to mid-sentence cut
    const kept = text.slice(0, cut).trimEnd();
    return kept.length ? kept : null;
}

async function maybeReroll(rawId) {
    const s = getSettings();
    if (!s.enabled || !s.autoReroll || _rerollBusy) return;

    const mesId = Number(rawId);
    const c = ctx();
    const chat = c.chat || [];
    if (mesId !== chat.length - 1) return;            // only the freshest message
    const msg = chat[mesId];
    if (!msg || msg.is_user || msg.is_system) return;

    const phrases = mergedBanned();
    if (phrases.length === 0) return;
    if (earliestBannedPos(msg.mes, phrases) < 0) return;

    const spent = _rerollCount.get(mesId) || 0;
    if (spent >= s.rerollMax) return;

    _rerollBusy = true;
    try {
        let cur = msg;
        for (let attempt = spent; attempt < s.rerollMax; attempt++) {
            const kept = truncateForReroll(cur.mes, phrases);
            if (kept === null) break;                 // nothing to salvage — leave as-is
            cur.mes = kept;
            c.updateMessageBlock(mesId, cur);
            await c.saveChat();
            _rerollCount.set(mesId, attempt + 1);
            await c.executeSlashCommandsWithOptions("/continue");
            cur = c.chat[mesId];                       // continue may replace the object
            if (!cur || earliestBannedPos(cur.mes, phrases) < 0) break;
        }
    } catch (err) {
        console.error("[SlopKiller] auto-reroll error:", err);
    } finally {
        _rerollBusy = false;
        refreshAllHighlights();
    }
}

// ====================================================================
// Prompt interceptor — injected before every (non-quiet) generation
// ====================================================================
globalThis.slopKillerInterceptor = async function (chat, _contextSize, _abort, type) {
    try {
        if (type === "quiet") return;
        const s = getSettings();
        if (!s.enabled || (!s.injectEnabled && !s.penaltyEnabled)) return;
        if (!Array.isArray(chat) || chat.length === 0) return;

        const { banned, slop } = injectionLists(s.maxInject);
        if (banned.length === 0 && slop.length === 0) return;

        if (s.injectEnabled) {
            const template = s.injectTemplate || DEFAULT_SETTINGS.injectTemplate;
            const mes = buildInjectionText(template, banned, slop);
            if (mes) {
                const note = {
                    is_user: false,
                    name: "System",
                    send_date: Date.now(),
                    mes,
                };
                chat.splice(chat.length - 1, 0, note);
            }
        }

        if (s.penaltyEnabled) boostPenalty(s);
    } catch (err) {
        console.error("[SlopKiller] interceptor error:", err);
    }
};

// ====================================================================
// Highlighting (operates on text nodes only — never breaks markup)
// ====================================================================
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

// Korean particle/ending alternation — pre-compiled for buildPhraseRegex.
const KO_SUFFIX_ALT = KO_SUFFIXES.map(escapeRe).join("|");

// Wrap a single phrase into a regex pattern that tolerates Korean inflection:
// any pure-Hangul token is matched as `stem(?:KO_SUFFIX)?`, so a banned
// "그녀는 미소지었다" also catches "그녀가 미소지으며", "그녀 미소지", etc.
// Non-Hangul tokens are escaped literally (English behavior unchanged).
function expandPhraseForKo(phrase) {
    return phrase.trim().split(/\s+/).map(tok => {
        if (!isHangulToken(tok)) return escapeRe(tok);
        const stem = stripKoSuffix(tok);
        return escapeRe(stem) + `(?:${KO_SUFFIX_ALT})?`;
    }).join("\\s+");
}

function buildPhraseRegex(phrases) {
    if (!phrases.length) return null;
    const alts = [...phrases]
        .sort((a, b) => b.length - a.length)
        .map(expandPhraseForKo)
        .filter(Boolean)
        .join("|");
    if (!alts) return null;
    return new RegExp(`(${alts})`, "gi");
}

function highlightInElement(root, phrases) {
    const re = buildPhraseRegex(phrases);
    if (!re) return;

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
            if (node.parentElement?.closest(".slop-hl")) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
        },
    });

    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    for (const node of nodes) {
        const text = node.nodeValue;
        re.lastIndex = 0;
        if (!re.test(text)) continue;

        re.lastIndex = 0;
        const frag = document.createDocumentFragment();
        let last = 0, m;
        while ((m = re.exec(text)) !== null) {
            if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
            const span = document.createElement("span");
            span.className = "slop-hl";
            span.title = "반복된 표현";
            span.textContent = m[0];
            frag.appendChild(span);
            last = m.index + m[0].length;
            if (m[0].length === 0) re.lastIndex++;
        }
        if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
        node.parentNode.replaceChild(frag, node);
    }
}

function clearHighlights(root) {
    root.querySelectorAll(".slop-hl").forEach(span => {
        span.replaceWith(document.createTextNode(span.textContent));
    });
    root.normalize();
}

function highlightMessage(mesId) {
    const s = getSettings();
    const el = document.querySelector(`#chat .mes[mesid="${mesId}"] .mes_text`);
    if (!el) return;
    clearHighlights(el);
    if (!s.enabled || !s.highlightEnabled) return;
    highlightInElement(el, highlightPhrases());
}

function refreshAllHighlights() {
    const s = getSettings();
    const phrases = (s.enabled && s.highlightEnabled) ? highlightPhrases() : [];
    document.querySelectorAll("#chat .mes").forEach(mes => {
        const el = mes.querySelector(".mes_text");
        if (!el) return;
        clearHighlights(el);
        if (mes.getAttribute("is_user") === "true") return;
        if (phrases.length) highlightInElement(el, phrases);
    });
}

// ====================================================================
// Color → CSS variables
// ====================================================================
function hexToRgba(hex, a) {
    const h = hex.replace("#", "");
    const n = h.length === 3
        ? h.split("").map(c => c + c).join("")
        : h.padEnd(6, "0").slice(0, 6);
    const r = parseInt(n.slice(0, 2), 16);
    const g = parseInt(n.slice(2, 4), 16);
    const b = parseInt(n.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function applyColor() {
    const color = getSettings().highlightColor || "#ff6b6b";
    document.body.style.setProperty("--sk-hl-color", color);
    document.body.style.setProperty("--sk-hl-bg", hexToRgba(color, 0.18));
}

// ====================================================================
// Theme — sets data-sk-theme on <body>; all panel styling is variable-driven
// in style.css. Themes: mono (flat B/W) + rose / ivory / sky pastels.
// ====================================================================
function applyTheme() {
    const theme = getSettings().theme || DEFAULT_SETTINGS.theme;
    document.body.setAttribute("data-sk-theme", theme);
    document.querySelectorAll("#slop_killer_panel .sk_theme_btn").forEach(btn => {
        btn.classList.toggle("sk_theme_active", btn.dataset.theme === theme);
    });
}

// ====================================================================
// Ban / allow actions — per-character and global, plus promote/demote
// ====================================================================
function addBanned(phrase) {
    const p = normalizePhrase(phrase);
    if (!p) return;
    const cd = getCharData(getCurrentCharName());
    cd.allowed = cd.allowed.filter(x => normalizePhrase(x) !== p);
    if (!cd.banned.some(x => normalizePhrase(x) === p)) cd.banned.push(p);
    save();
    renderPanel();
    refreshAllHighlights();
}

function addAllowed(phrase) {
    const p = normalizePhrase(phrase);
    if (!p) return;
    const cd = getCharData(getCurrentCharName());
    cd.banned = cd.banned.filter(x => normalizePhrase(x) !== p);
    if (!cd.allowed.some(x => normalizePhrase(x) === p)) cd.allowed.push(p);
    save();
    renderPanel();
    refreshAllHighlights();
}

function removeFrom(kind, phrase) {
    const p = normalizePhrase(phrase);
    const cd = getCharData(getCurrentCharName());
    if (kind === "banned") cd.banned = cd.banned.filter(x => normalizePhrase(x) !== p);
    else cd.allowed = cd.allowed.filter(x => normalizePhrase(x) !== p);
    save();
    renderPanel();
    refreshAllHighlights();
}

function addBannedGlobal(phrase) {
    const p = normalizePhrase(phrase);
    if (!p) return;
    const g = getGlobal();
    g.allowed = g.allowed.filter(x => normalizePhrase(x) !== p);
    if (!g.banned.some(x => normalizePhrase(x) === p)) g.banned.push(p);
    save();
    renderPanel();
    refreshAllHighlights();
}

function addAllowedGlobal(phrase) {
    const p = normalizePhrase(phrase);
    if (!p) return;
    const g = getGlobal();
    g.banned = g.banned.filter(x => normalizePhrase(x) !== p);
    if (!g.allowed.some(x => normalizePhrase(x) === p)) g.allowed.push(p);
    save();
    renderPanel();
    refreshAllHighlights();
}

function removeFromGlobal(kind, phrase) {
    const p = normalizePhrase(phrase);
    const g = getGlobal();
    if (kind === "banned") g.banned = g.banned.filter(x => normalizePhrase(x) !== p);
    else g.allowed = g.allowed.filter(x => normalizePhrase(x) !== p);
    save();
    renderPanel();
    refreshAllHighlights();
}

// Move a phrase from current-character list → global list (or vice versa).
// Idempotent: dedupes at the destination, removes from the source.
function promoteToGlobal(phrase, kind) {
    const p = normalizePhrase(phrase);
    if (!p) return;
    const cd = getCharData(getCurrentCharName());
    const g = getGlobal();
    if (kind === "banned") {
        cd.banned = cd.banned.filter(x => normalizePhrase(x) !== p);
        if (!g.banned.some(x => normalizePhrase(x) === p)) g.banned.push(p);
    } else {
        cd.allowed = cd.allowed.filter(x => normalizePhrase(x) !== p);
        if (!g.allowed.some(x => normalizePhrase(x) === p)) g.allowed.push(p);
    }
    save();
    renderPanel();
    refreshAllHighlights();
}

function demoteToCharacter(phrase, kind) {
    const p = normalizePhrase(phrase);
    const name = getCurrentCharName();
    if (!p) return;
    if (!name) {
        toastr?.warning?.("캐릭터가 선택되지 않았습니다");
        return;
    }
    const cd = getCharData(name);
    const g = getGlobal();
    if (kind === "banned") {
        g.banned = g.banned.filter(x => normalizePhrase(x) !== p);
        if (!cd.banned.some(x => normalizePhrase(x) === p)) cd.banned.push(p);
    } else {
        g.allowed = g.allowed.filter(x => normalizePhrase(x) !== p);
        if (!cd.allowed.some(x => normalizePhrase(x) === p)) cd.allowed.push(p);
    }
    save();
    renderPanel();
    refreshAllHighlights();
}

// ====================================================================
// Filter + sort for chip lists
// ====================================================================
function applyFilterSort(list, q, sort) {
    let arr = [...list];
    if (q) {
        const needle = q.toLowerCase();
        arr = arr.filter(p => String(p).toLowerCase().includes(needle));
    }
    if (sort === "alpha") {
        arr.sort((a, b) => String(a).localeCompare(String(b)));
    } else if (sort === "count") {
        const counted = arr.map(p => [p, countOccurrencesInChat(p)]);
        counted.sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
        arr = counted.map(x => x[0]);
    }
    // "added" — keep insertion order
    return arr;
}

// ====================================================================
// Settings panel
// ====================================================================
function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const SORT_OPTIONS = `
    <option value="added">추가순</option>
    <option value="alpha">알파벳</option>
    <option value="count">현재 채팅 등장순</option>
`;

function buildPanel() {
    if (document.getElementById(`${MODULE_NAME}_panel`)) return;
    const host = document.body;
    if (!host) return;

    const s = getSettings();
    const html = `
    <div id="sk_backdrop" class="sk_backdrop" style="display:none;"></div>
    <div id="${MODULE_NAME}_panel" class="sk_window" style="display:none;">
        <div class="sk_window_header">
            <span class="sk_window_title">AI 반복 킬러</span>
            <div class="sk_window_close fa-solid fa-xmark" title="닫기"></div>
        </div>
        <div class="sk_window_body">

                <label class="checkbox_label sk_master_toggle">
                    <input id="sk_enabled" type="checkbox" ${s.enabled ? "checked" : ""}>
                    <span>확장 활성화</span>
                </label>

                <div class="sk_tabs">
                    <button class="sk_tab_btn" data-tab="banned">금지어</button>
                    <button class="sk_tab_btn" data-tab="global">글로벌</button>
                    <button class="sk_tab_btn" data-tab="detect">감지</button>
                    <button class="sk_tab_btn" data-tab="settings">설정</button>
                </div>

                <!-- 금지어 (캐릭터별) -->
                <div class="sk_tab_panel" data-tab="banned">
                    <h4><i class="fa-solid fa-chart-simple sk_h4_icon"></i>현재 캐릭터: <span id="sk_charname" class="sk_charname"></span></h4>
                    <p class="sk_hint">자주 반복된 표현입니다 (많은 순). <i class="fa-solid fa-ban sk_ic_ban"></i> 누르면 금지어로 추가 · <i class="fa-solid fa-check sk_ic_allow"></i> 누르면 반복 아님으로 제외</p>
                    <div id="sk_ranking" class="sk_ranking"></div>
                    <button id="sk_rescan" class="menu_button sk_rescan_btn">다시 스캔</button>

                    <hr>
                    <h4><i class="fa-solid fa-ban sk_h4_icon"></i>등록된 금지어</h4>
                    <p class="sk_hint">위 목록에서 <i class="fa-solid fa-ban sk_ic_ban"></i>를 누르거나, 아래에 직접 입력해 추가할 수 있습니다.</p>
                    <div class="sk_ban_row">
                        <input id="sk_ban_input" type="text" class="text_pole" placeholder="금지할 표현 입력">
                        <button id="sk_ban_add" class="menu_button">추가</button>
                    </div>
                    <div id="sk_char_banned_filter" class="sk_filter" hidden>
                        <input id="sk_char_banned_search" type="text" class="text_pole sk_search" placeholder="검색">
                        <select id="sk_char_banned_sort" class="text_pole sk_sort">${SORT_OPTIONS}</select>
                    </div>
                    <div id="sk_banned_list" class="sk_list"></div>

                    <hr>
                    <h4><i class="fa-solid fa-circle-check sk_h4_icon"></i>허용어 (반복 아님)</h4>
                    <div id="sk_allowed_list" class="sk_list"></div>
                </div>

                <!-- 글로벌 -->
                <div class="sk_tab_panel" data-tab="global" hidden>
                    <h4><i class="fa-solid fa-earth-americas sk_h4_icon"></i>글로벌 금지어</h4>
                    <p class="sk_hint">모든 캐릭터에 적용됩니다. 캐릭터별 금지어와 합쳐서 동작합니다. ▼ 누르면 현재 캐릭터로 옮깁니다.</p>
                    <div class="sk_ban_row">
                        <input id="sk_global_ban_input" type="text" class="text_pole" placeholder="글로벌 금지어 입력">
                        <button id="sk_global_ban_add" class="menu_button">추가</button>
                    </div>
                    <div id="sk_global_banned_filter" class="sk_filter" hidden>
                        <input id="sk_global_banned_search" type="text" class="text_pole sk_search" placeholder="검색">
                        <select id="sk_global_banned_sort" class="text_pole sk_sort">${SORT_OPTIONS}</select>
                    </div>
                    <div id="sk_global_banned_list" class="sk_list"></div>

                    <hr>
                    <h4><i class="fa-solid fa-earth-americas sk_h4_icon"></i>글로벌 허용어</h4>
                    <p class="sk_hint">모든 캐릭터에서 '반복 아님'으로 제외됩니다.</p>
                    <div id="sk_global_allowed_list" class="sk_list"></div>
                </div>

                <!-- 감지 -->
                <div class="sk_tab_panel" data-tab="detect" hidden>
                    <h4><i class="fa-solid fa-magnifying-glass sk_h4_icon"></i>감지 설정</h4>
                    <label>표현 길이 — 최소 <span id="sk_minN_val">${s.minN}</span> 단어</label>
                    <input id="sk_minN" type="range" min="1" max="5" value="${s.minN}" class="sk_slider">
                    <label>표현 길이 — 최대 <span id="sk_maxN_val">${s.maxN}</span> 단어</label>
                    <input id="sk_maxN" type="range" min="1" max="6" value="${s.maxN}" class="sk_slider">
                    <label>반복으로 볼 기준 — <span id="sk_threshold_val">${s.threshold}</span>회 이상</label>
                    <input id="sk_threshold" type="range" min="2" max="15" value="${s.threshold}" class="sk_slider">
                    <label>훑어볼 범위 — 최근 <span id="sk_scanDepth_val">${s.scanDepth}</span>개 메시지</label>
                    <input id="sk_scanDepth" type="range" min="5" max="200" step="5" value="${s.scanDepth}" class="sk_slider">

                    <hr>
                    <h4><i class="fa-solid fa-filter sk_h4_icon"></i>불용어</h4>
                    <label>커스텀 불용어 — 감지에서 무시할 단어 (쉼표·줄바꿈 구분)</label>
                    <textarea id="sk_customStopwords" class="text_pole sk_template" rows="2" spellcheck="false">${escapeHtml(s.customStopwords)}</textarea>
                    <p class="sk_hint">캐릭터 이름이나 자주 쓰는 호칭을 넣으면 반복 후보에서 빠집니다.</p>

                    <hr>
                    <h4><i class="fa-solid fa-bookmark sk_h4_icon"></i>프리셋</h4>
                    <p class="sk_hint">현재 감지 슬라이더 + 불용어 설정을 묶음으로 저장합니다.</p>
                    <div class="sk_preset_row">
                        <select id="sk_preset_select" class="text_pole"></select>
                        <button id="sk_preset_load" class="menu_button" title="선택한 프리셋 적용">불러오기</button>
                        <button id="sk_preset_delete" class="menu_button" title="선택한 프리셋 삭제">삭제</button>
                    </div>
                    <div class="sk_preset_row">
                        <input id="sk_preset_name" type="text" class="text_pole" placeholder="새 프리셋 이름 (예: 한국어)">
                        <button id="sk_preset_save" class="menu_button" title="현재 설정을 이 이름으로 저장">저장</button>
                    </div>
                </div>

                <!-- 설정 (테마 / 프롬프트 / 페널티 / 리롤 / 하이라이트 통합) -->
                <div class="sk_tab_panel" data-tab="settings" hidden>
                    <h4><i class="fa-solid fa-palette sk_h4_icon"></i>테마</h4>
                    <div class="sk_theme_picker">
                        <button class="sk_theme_btn" data-theme="mono"  title="Mono (흑백)"></button>
                        <button class="sk_theme_btn" data-theme="rose"  title="Rose (로즈)"></button>
                        <button class="sk_theme_btn" data-theme="ivory" title="Ivory (아이보리)"></button>
                        <button class="sk_theme_btn" data-theme="sky"   title="Sky (스카이)"></button>
                    </div>

                    <hr>
                    <h4><i class="fa-solid fa-comment-dots sk_h4_icon"></i>프롬프트</h4>
                    <label class="checkbox_label">
                        <input id="sk_injectEnabled" type="checkbox" ${s.injectEnabled ? "checked" : ""}>
                        <span>사용</span>
                    </label>
                    <label>한 번에 알려줄 표현 — 최대 <span id="sk_maxInject_val">${s.maxInject}</span>개</label>
                    <input id="sk_maxInject" type="range" min="1" max="40" value="${s.maxInject}" class="sk_slider">
                    <label>모델에게 보낼 문구</label>
                    <p class="sk_hint"><code>{{banned}}</code> 자리엔 등록한 금지어, <code>{{slop}}</code> 자리엔 자동으로 찾은 반복 표현, <code>{{phrases}}</code> 자리엔 둘 다. 해당 목록이 비어 있으면 그 줄은 자동 생략됩니다.</p>
                    <textarea id="sk_injectTemplate" class="text_pole sk_template" rows="4" spellcheck="false">${escapeHtml(s.injectTemplate)}</textarea>
                    <button id="sk_injectReset" class="menu_button sk_reset_btn">기본 문구로 복원</button>

                    <hr>
                    <h4><i class="fa-solid fa-gauge-high sk_h4_icon"></i>반복 페널티 올리기</h4>
                    <label class="checkbox_label">
                        <input id="sk_penaltyEnabled" type="checkbox" ${s.penaltyEnabled ? "checked" : ""}>
                        <span>반복이 감지되면 frequency / presence penalty를 자동으로 올려줍니다</span>
                    </label>
                    <p class="sk_hint">오픈AI 호환 백엔드(예: OpenRouter, DeepSeek, Moonshot 등)에서만 작동합니다. Gemini·Claude는 무시됩니다.</p>
                    <label>부스트 강도 — <span id="sk_penaltyBoost_val">${s.penaltyBoost}</span></label>
                    <input id="sk_penaltyBoost" type="range" min="0.1" max="1.0" step="0.1" value="${s.penaltyBoost}" class="sk_slider">

                    <hr>
                    <h4><i class="fa-solid fa-arrows-rotate sk_h4_icon"></i>자동 리롤</h4>
                    <label class="checkbox_label">
                        <input id="sk_autoReroll" type="checkbox" ${s.autoReroll ? "checked" : ""}>
                        <span>등록한 금지어가 답변에 나오면 자동으로 다시 생성합니다</span>
                    </label>
                    <p class="sk_hint">금지어 직전 문장까지 남기고 그 뒤만 이어쓰기로 재생성합니다. 토큰이 추가로 소모됩니다.</p>
                    <label>다시 시도 — 최대 <span id="sk_rerollMax_val">${s.rerollMax}</span>회</label>
                    <input id="sk_rerollMax" type="range" min="1" max="5" value="${s.rerollMax}" class="sk_slider">

                    <hr>
                    <h4><i class="fa-solid fa-highlighter sk_h4_icon"></i>하이라이트</h4>
                    <label class="checkbox_label">
                        <input id="sk_highlightEnabled" type="checkbox" ${s.highlightEnabled ? "checked" : ""}>
                        <span>반복 표현 색칠</span>
                    </label>
                    <label>하이라이트 색상</label>
                    <div class="sk_color_row">
                        <div class="sk_color_preview" id="sk_color_preview" style="background:${s.highlightColor}"></div>
                        <input id="sk_highlightColor" type="text" class="text_pole sk_color_input"
                               value="${s.highlightColor}" placeholder="#rrggbb" maxlength="7" spellcheck="false">
                    </div>
                    <div class="sk_color_chips">
                        ${PASTEL_CHIPS.map(c => `<button class="sk_color_chip" data-color="${c}" style="background:${c}" title="${c}"></button>`).join("")}
                    </div>
                </div>

        </div>
    </div>`;

    host.insertAdjacentHTML("beforeend", html);
    bindPanel();
    switchTab(s.activeTab || "banned");
    renderPanel();
}

function switchTab(tabName) {
    const tab = TABS.includes(tabName) ? tabName : "banned";
    document.querySelectorAll("#slop_killer_panel .sk_tab_btn").forEach(btn => {
        btn.classList.toggle("sk_tab_active", btn.dataset.tab === tab);
    });
    document.querySelectorAll("#slop_killer_panel .sk_tab_panel").forEach(p => {
        p.hidden = p.dataset.tab !== tab;
    });
    const s = getSettings();
    if (s.activeTab !== tab) {
        s.activeTab = tab;
        save();
    }
}

function bindPanel() {
    const s = getSettings();

    const bindCheckbox = (id, key, after) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener("change", () => {
            s[key] = el.checked;
            save();
            after?.();
        });
    };

    const bindSlider = (id, key, parser = parseInt, after) => {
        const el = document.getElementById(id);
        const lbl = document.getElementById(`${id}_val`);
        if (!el) return;
        el.addEventListener("input", () => {
            s[key] = parser(el.value);
            if (lbl) lbl.textContent = el.value;
        });
        el.addEventListener("change", () => { save(); after?.(); });
    };

    bindCheckbox("sk_enabled", "enabled", () => { renderPanel(); refreshAllHighlights(); });
    bindSlider("sk_minN", "minN", parseInt, () => { renderPanel(); refreshAllHighlights(); });
    bindSlider("sk_maxN", "maxN", parseInt, () => { renderPanel(); refreshAllHighlights(); });
    bindSlider("sk_threshold", "threshold", parseInt, () => { renderPanel(); refreshAllHighlights(); });
    bindSlider("sk_scanDepth", "scanDepth", parseInt, () => { renderPanel(); refreshAllHighlights(); });

    const csEl = document.getElementById("sk_customStopwords");
    if (csEl) {
        csEl.addEventListener("input", () => { s.customStopwords = csEl.value; });
        csEl.addEventListener("change", () => { save(); renderPanel(); refreshAllHighlights(); });
    }

    // ---- 감지 프리셋 ----
    const presetSel  = document.getElementById("sk_preset_select");
    const presetName = document.getElementById("sk_preset_name");
    const presetSave = document.getElementById("sk_preset_save");
    const presetLoad = document.getElementById("sk_preset_load");
    const presetDel  = document.getElementById("sk_preset_delete");

    function renderPresetSelect(selectedName) {
        if (!presetSel) return;
        const names = Object.keys(s.detectPresets || {}).sort((a, b) => a.localeCompare(b, "ko"));
        if (!names.length) {
            presetSel.innerHTML = `<option value="">— 저장된 프리셋 없음 —</option>`;
            presetSel.disabled = true;
        } else {
            presetSel.disabled = false;
            presetSel.innerHTML = names.map(n =>
                `<option value="${escapeHtml(n)}"${n === selectedName ? " selected" : ""}>${escapeHtml(n)}</option>`
            ).join("");
        }
    }

    function applyPresetValues(p) {
        if (typeof p.minN      === "number") s.minN      = p.minN;
        if (typeof p.maxN      === "number") s.maxN      = p.maxN;
        if (typeof p.threshold === "number") s.threshold = p.threshold;
        if (typeof p.scanDepth === "number") s.scanDepth = p.scanDepth;
        if (typeof p.customStopwords === "string") s.customStopwords = p.customStopwords;

        for (const k of ["minN", "maxN", "threshold", "scanDepth"]) {
            const el  = document.getElementById(`sk_${k}`);
            const lbl = document.getElementById(`sk_${k}_val`);
            if (el)  el.value = s[k];
            if (lbl) lbl.textContent = String(s[k]);
        }
        if (csEl) csEl.value = s.customStopwords || "";
    }

    presetSave?.addEventListener("click", () => {
        const name = (presetName?.value || "").trim();
        if (!name) { toastr?.warning?.("프리셋 이름을 입력하세요"); return; }
        s.detectPresets[name] = {
            minN: s.minN, maxN: s.maxN, threshold: s.threshold, scanDepth: s.scanDepth,
            customStopwords: s.customStopwords || "",
        };
        save();
        renderPresetSelect(name);
        if (presetName) presetName.value = "";
        toastr?.success?.(`프리셋 저장: ${name}`);
    });

    presetLoad?.addEventListener("click", () => {
        const name = presetSel?.value;
        const p = name && s.detectPresets?.[name];
        if (!p) { toastr?.warning?.("프리셋을 선택하세요"); return; }
        applyPresetValues(p);
        save();
        renderPanel();
        refreshAllHighlights();
        toastr?.success?.(`프리셋 적용: ${name}`);
    });

    presetDel?.addEventListener("click", () => {
        const name = presetSel?.value;
        if (!name || !s.detectPresets?.[name]) { toastr?.warning?.("프리셋을 선택하세요"); return; }
        delete s.detectPresets[name];
        save();
        renderPresetSelect();
        toastr?.info?.(`프리셋 삭제: ${name}`);
    });

    renderPresetSelect();

    bindCheckbox("sk_injectEnabled", "injectEnabled");
    bindSlider("sk_maxInject", "maxInject");

    const tmplEl = document.getElementById("sk_injectTemplate");
    tmplEl.addEventListener("input", () => { s.injectTemplate = tmplEl.value; });
    tmplEl.addEventListener("change", save);
    document.getElementById("sk_injectReset").addEventListener("click", () => {
        s.injectTemplate = DEFAULT_SETTINGS.injectTemplate;
        tmplEl.value = s.injectTemplate;
        save();
    });

    bindCheckbox("sk_penaltyEnabled", "penaltyEnabled");
    bindSlider("sk_penaltyBoost", "penaltyBoost", parseFloat);

    bindCheckbox("sk_autoReroll", "autoReroll");
    bindSlider("sk_rerollMax", "rerollMax");

    bindCheckbox("sk_highlightEnabled", "highlightEnabled", refreshAllHighlights);

    const colorInput = document.getElementById("sk_highlightColor");
    const colorPreview = document.getElementById("sk_color_preview");

    function isHex(v) { return /^#[0-9a-fA-F]{6}$/.test(v); }

    function syncColorChips(hex) {
        document.querySelectorAll("#slop_killer_panel .sk_color_chip").forEach(btn => {
            btn.classList.toggle("sk_color_active", btn.dataset.color.toLowerCase() === hex.toLowerCase());
        });
    }

    colorInput.addEventListener("input", () => {
        const val = colorInput.value.trim();
        if (!isHex(val)) return;
        s.highlightColor = val;
        colorPreview.style.background = val;
        syncColorChips(val);
        applyColor();
    });
    colorInput.addEventListener("change", () => { if (isHex(colorInput.value.trim())) save(); });

    document.querySelectorAll("#slop_killer_panel .sk_color_chip").forEach(btn => {
        btn.addEventListener("click", () => {
            const hex = btn.dataset.color;
            s.highlightColor = hex;
            colorInput.value = hex;
            colorPreview.style.background = hex;
            syncColorChips(hex);
            applyColor();
            save();
        });
    });
    syncColorChips(s.highlightColor);

    document.getElementById("sk_rescan").addEventListener("click", () => {
        renderPanel();
        refreshAllHighlights();
    });

    // Character-scope ban input
    const banInput = document.getElementById("sk_ban_input");
    const doAddChar = () => { if (banInput.value.trim()) { addBanned(banInput.value); banInput.value = ""; } };
    document.getElementById("sk_ban_add").addEventListener("click", doAddChar);
    banInput.addEventListener("keydown", (e) => { if (e.key === "Enter") doAddChar(); });

    // Global-scope ban input
    const globalBanInput = document.getElementById("sk_global_ban_input");
    const doAddGlobal = () => {
        if (globalBanInput.value.trim()) { addBannedGlobal(globalBanInput.value); globalBanInput.value = ""; }
    };
    document.getElementById("sk_global_ban_add").addEventListener("click", doAddGlobal);
    globalBanInput.addEventListener("keydown", (e) => { if (e.key === "Enter") doAddGlobal(); });

    // Search + sort
    const bindFilter = (searchId, sortId, state) => {
        const sEl = document.getElementById(searchId);
        const sortEl = document.getElementById(sortId);
        if (!sEl || !sortEl) return;
        sEl.value = state.q;
        sortEl.value = state.sort;
        sEl.addEventListener("input", () => { state.q = sEl.value; renderChips(); });
        sortEl.addEventListener("change", () => { state.sort = sortEl.value; renderChips(); });
    };
    bindFilter("sk_char_banned_search", "sk_char_banned_sort", _filter.charBanned);
    bindFilter("sk_global_banned_search", "sk_global_banned_sort", _filter.globalBanned);

    // Tabs
    document.querySelectorAll("#slop_killer_panel .sk_tab_btn").forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // Theme
    document.querySelectorAll("#slop_killer_panel .sk_theme_btn").forEach(btn => {
        btn.addEventListener("click", () => {
            s.theme = btn.dataset.theme;
            applyTheme();
            save();
        });
    });

    // Window chrome — close button + backdrop click closes
    const win = document.getElementById(`${MODULE_NAME}_panel`);
    win.querySelector(".sk_window_close")?.addEventListener("click", closeWindow);
    document.getElementById("sk_backdrop")?.addEventListener("click", closeWindow);
}

function renderPanel() {
    renderRanking();
    renderChips();
}

function renderRanking() {
    const box = document.getElementById("sk_ranking");
    if (!box) return;

    const name = getCurrentCharName();
    const nameEl = document.getElementById("sk_charname");
    if (nameEl) nameEl.textContent = name || "(선택 안 됨)";

    if (!name) { box.innerHTML = `<div class="sk_hint">캐릭터를 선택하세요</div>`; return; }

    const list = rankSlop();
    if (!list.length) {
        box.innerHTML = `<div class="sk_hint">반복된 표현이 없습니다 (${getSettings().threshold}회 이상 반복되면 여기 표시됩니다)</div>`;
        return;
    }

    box.innerHTML = list.slice(0, 40).map(({ phrase, count }) => `
        <div class="sk_rank_row">
            <span class="sk_rank_count">×${count}</span>
            <span class="sk_rank_phrase">${escapeHtml(phrase)}</span>
            <button class="menu_button sk_ban_btn" data-p="${escapeHtml(phrase)}" title="금지어로 추가"><i class="fa-solid fa-ban sk_ic_ban"></i></button>
            <button class="menu_button sk_allow_btn" data-p="${escapeHtml(phrase)}" title="반복 아님으로 제외"><i class="fa-solid fa-check sk_ic_allow"></i></button>
        </div>`).join("");

    box.querySelectorAll(".sk_ban_btn").forEach(b =>
        b.addEventListener("click", () => addBanned(b.dataset.p)));
    box.querySelectorAll(".sk_allow_btn").forEach(b =>
        b.addEventListener("click", () => addAllowed(b.dataset.p)));
}

function renderChips() {
    const cd = getCharData(getCurrentCharName());
    const g = getGlobal();

    setHidden("sk_char_banned_filter", cd.banned.length < SEARCH_THRESHOLD);
    setHidden("sk_global_banned_filter", g.banned.length < SEARCH_THRESHOLD);

    renderChipBox("sk_banned_list",        cd.banned,  "banned",  "char",   _filter.charBanned);
    renderChipBox("sk_allowed_list",       cd.allowed, "allowed", "char",   null);
    renderChipBox("sk_global_banned_list", g.banned,   "banned",  "global", _filter.globalBanned);
    renderChipBox("sk_global_allowed_list",g.allowed,  "allowed", "global", null);
}

function setHidden(id, hidden) {
    const el = document.getElementById(id);
    if (el) el.hidden = !!hidden;
}

// kind:  "banned" | "allowed"
// scope: "char"   | "global"
function renderChipBox(id, list, kind, scope, filter) {
    const box = document.getElementById(id);
    if (!box) return;

    const arr = filter ? applyFilterSort(list, filter.q, filter.sort) : [...list];

    if (!arr.length) {
        box.innerHTML = (filter && filter.q)
            ? `<div class="sk_list_empty">검색 결과 없음</div>`
            : `<div class="sk_list_empty">없음</div>`;
        return;
    }

    const moveArrow = scope === "char" ? "▲" : "▼";
    const moveTitle = scope === "char" ? "글로벌로 옮기기" : "현재 캐릭터로 옮기기";

    box.innerHTML = arr.map(p => {
        const esc = escapeHtml(p);
        return `<div class="sk_list_row">
            <span class="sk_list_text" title="${esc}">${esc}</span>
            <button class="sk_list_move" data-p="${esc}" title="${moveTitle}">${moveArrow}</button>
            <button class="sk_list_remove" data-p="${esc}" title="제거">×</button>
        </div>`;
    }).join("");

    box.querySelectorAll(".sk_list_remove").forEach(b =>
        b.addEventListener("click", () => {
            if (scope === "char") removeFrom(kind, b.dataset.p);
            else removeFromGlobal(kind, b.dataset.p);
        }));
    box.querySelectorAll(".sk_list_move").forEach(b =>
        b.addEventListener("click", () => {
            if (scope === "char") promoteToGlobal(b.dataset.p, kind);
            else demoteToCharacter(b.dataset.p, kind);
        }));
}

// ====================================================================
// Chat context-menu — right-click or long-press on AI messages
// ====================================================================
let _ctxMenu = null;
let _longPressTimer = null;

function wordAtCaret(range) {
    const node = range.startContainer;
    if (!node || node.nodeType !== Node.TEXT_NODE) return null;
    const text = node.textContent;
    const off = range.startOffset;
    let s = off, e = off;
    while (s > 0 && /[\p{L}\p{N}']/u.test(text[s - 1])) s--;
    while (e < text.length && /[\p{L}\p{N}']/u.test(text[e])) e++;
    return text.slice(s, e).trim() || null;
}

function phraseFromEvent(target, x, y) {
    const hl = target.closest?.(".slop-hl");
    if (hl) return hl.textContent.trim();
    const sel = window.getSelection();
    const selText = sel?.toString().trim().replace(/\s+/g, " ") ?? "";
    if (selText && selText.length < 120) return selText;
    let range = null;
    if (document.caretRangeFromPoint) {
        range = document.caretRangeFromPoint(x, y);
    } else if (document.caretPositionFromPoint) {
        const pos = document.caretPositionFromPoint(x, y);
        if (pos) { range = document.createRange(); range.setStart(pos.offsetNode, pos.offset); }
    }
    return range ? wordAtCaret(range) : null;
}

function showCtxMenu(x, y, phrase) {
    hideCtxMenu();
    if (!phrase) return;

    const menu = document.createElement("div");
    menu.className = "sk_ctx_menu";
    menu.innerHTML = `
        <span class="sk_ctx_phrase">"${escapeHtml(phrase)}"</span>
        <button class="sk_ctx_btn">추가</button>
    `;
    document.body.appendChild(menu);
    _ctxMenu = menu;

    const vw = window.innerWidth, vh = window.innerHeight;
    const mw = 220, mh = 72;
    const left = Math.max(8, Math.min(x, vw - mw - 8));
    const top  = (y + mh + 12 > vh) ? y - mh - 8 : y + 10;
    menu.style.left = `${left}px`;
    menu.style.top  = `${Math.max(8, top)}px`;

    menu.querySelector(".sk_ctx_btn").addEventListener("click", () => {
        addBanned(phrase);
        showCtxToast(`"${phrase}" 추가됨`);
        hideCtxMenu();
    });

    setTimeout(() => document.addEventListener("pointerdown", _ctxOutside, { capture: true, once: true }), 0);
}

function _ctxOutside(e) { if (!_ctxMenu?.contains(e.target)) hideCtxMenu(); }

function hideCtxMenu() { _ctxMenu?.remove(); _ctxMenu = null; }

function showCtxToast(msg) {
    const t = document.createElement("div");
    t.className = "sk_toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2000);
}

function initChatContextMenu() {
    const chat = document.getElementById("chat");
    if (!chat) return;

    chat.addEventListener("contextmenu", (e) => {
        const mes = e.target.closest(".mes");
        if (!mes || mes.getAttribute("is_user") === "true") return;
        e.preventDefault();
        const phrase = phraseFromEvent(e.target, e.clientX, e.clientY);
        if (phrase) showCtxMenu(e.clientX, e.clientY, phrase);
    });

    chat.addEventListener("touchstart", (e) => {
        const mes = e.target.closest(".mes");
        if (!mes || mes.getAttribute("is_user") === "true") return;
        const touch = e.touches[0];
        const target = e.target;
        _longPressTimer = setTimeout(() => {
            const phrase = phraseFromEvent(target, touch.clientX, touch.clientY);
            if (phrase) { navigator.vibrate?.(40); showCtxMenu(touch.clientX, touch.clientY, phrase); }
        }, 500);
    }, { passive: true });

    const cancelLp = () => clearTimeout(_longPressTimer);
    chat.addEventListener("touchmove",   cancelLp, { passive: true });
    chat.addEventListener("touchend",    cancelLp, { passive: true });
    chat.addEventListener("touchcancel", cancelLp, { passive: true });

    document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideCtxMenu(); });
}

// ====================================================================
// Floating window — opened from the wand (magic-wand) menu
// ====================================================================
function openWindow() {
    const win = document.getElementById(`${MODULE_NAME}_panel`);
    if (!win) return;
    const bd = document.getElementById("sk_backdrop");
    if (bd) bd.style.display = "block";
    win.style.display = "flex";
    renderPanel();
    const menu = document.getElementById("extensionsMenu");
    if (menu) menu.style.display = "none";
}

function closeWindow() {
    const win = document.getElementById(`${MODULE_NAME}_panel`);
    if (win) win.style.display = "none";
    const bd = document.getElementById("sk_backdrop");
    if (bd) bd.style.display = "none";
}

function addWandButton() {
    const menu = document.getElementById("extensionsMenu");
    if (!menu || document.getElementById("sk_wand_button")) return;
    const item = document.createElement("div");
    item.id = "sk_wand_button";
    item.className = "list-group-item flex-container flexGap5 interactable";
    item.tabIndex = 0;
    item.title = "AI 반복 킬러 열기";
    item.innerHTML = `<i class="fa-solid fa-broom"></i><span>AI 반복 킬러</span>`;
    item.addEventListener("click", openWindow);
    menu.appendChild(item);
}

// ====================================================================
// Init
// ====================================================================
jQuery(() => {
    try {
        const { eventSource, event_types } = ctx();

        getSettings();
        applyColor();
        buildPanel();
        applyTheme();
        initChatContextMenu();
        addWandButton();

        // Escape closes the window (unless typing in a field).
        document.addEventListener("keydown", (e) => {
            if (e.key !== "Escape") return;
            const t = e.target;
            if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
            closeWindow();
        });

        // MESSAGE_RECEIVED fires only for freshly generated replies (not on chat
        // load, not on abort), so we use it to mark which message is eligible for
        // auto-reroll. The reroll itself runs after GENERATION_ENDED, deferred via
        // setTimeout so the continue call isn't nested inside the generation pipeline.
        eventSource.on(event_types.MESSAGE_RECEIVED, (mesId) => { _lastFreshId = Number(mesId); });
        eventSource.on(event_types.GENERATION_ENDED, () => {
            restorePenalty();
            if (_lastFreshId === null) return;
            const id = _lastFreshId;
            _lastFreshId = null;
            setTimeout(() => maybeReroll(id), 0);
        });

        eventSource.on(event_types.CHARACTER_MESSAGE_RENDERED, (mesId) => {
            highlightMessage(mesId);
            renderRanking();
        });
        eventSource.on(event_types.MESSAGE_SWIPED, () => renderRanking());
        eventSource.on(event_types.CHAT_CHANGED, () => {
            _rerollCount.clear();
            renderPanel();
            setTimeout(refreshAllHighlights, 300);
        });

        setTimeout(() => { renderPanel(); refreshAllHighlights(); }, 500);
        console.log("[SlopKiller] loaded");
    } catch (err) {
        console.error("[SlopKiller] init failed:", err);
    }
});
