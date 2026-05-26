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

const DEFAULT_SETTINGS = Object.freeze({
    enabled: true,
    theme: "cream",     // system | mono | cream | peach | lilac
    minN: 2,            // shortest phrase length (words)
    maxN: 4,            // longest phrase length (words)
    threshold: 3,       // occurrences needed to count as slop
    scanDepth: 50,      // how many recent AI messages to scan
    injectEnabled: true,
    maxInject: 12,      // max phrases sent to the model
    injectTemplate:
        "[System note — writing variety]\n" +
        "STRICTLY FORBIDDEN — never write these phrases or any close variation, under any circumstance: {{banned}}.\n" +
        "Also avoid overusing these repeated phrases, or close paraphrases: {{slop}}.\n" +
        "Vary your sentence structure and reach for fresh wording and new sensory detail instead.",
    highlightEnabled: true,
    highlightColor: "#ff6b6b",
    penaltyEnabled: true,
    penaltyBoost: 0.3,  // added to freq/pres penalty on OpenAI-compatible backends
    autoReroll: true,   // re-generate (via continue) when a banned phrase appears
    rerollMax: 3,       // max continue attempts per message
    characters: {},     // charName -> { banned: [], allowed: [] }
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

// Old default templates — auto-upgraded to the current default on load.
const LEGACY_INJECT_TEMPLATES = new Set([
    "[System note — writing variety] Avoid reusing the following overused phrases, or any close paraphrase, in your next reply: {{phrases}}. Vary your sentence structure and reach for fresh wording and new sensory detail instead.",
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

// ====================================================================
// Context helpers
// ====================================================================
function ctx() { return SillyTavern.getContext(); }

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

function contentWordCount(words) { return words.filter(w => !STOPWORDS.has(w)).length; }

function computeCounts() {
    const s = getSettings();
    const chat = ctx().chat || [];
    const msgs = chat.filter(m =>
        m && !m.is_user && !m.is_system && typeof m.mes === "string" && m.mes.trim()
    ).slice(-s.scanDepth);

    const counts = new Map();
    for (const m of msgs) {
        const words = tokenize(m.mes);
        for (let n = s.minN; n <= s.maxN; n++) {
            for (let i = 0; i + n <= words.length; i++) {
                const gram = words.slice(i, i + n);
                if (contentWordCount(gram) < 2) continue;
                const key = gram.join(" ");
                counts.set(key, (counts.get(key) || 0) + 1);
            }
        }
    }
    return counts;
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
// overlaps an allowed phrase, so ✓ clears the whole run). Sorted by count.
function aboveThreshold() {
    const s = getSettings();
    const cd = getCharData(getCurrentCharName());
    const allowed = cd.allowed.map(x => x.toLowerCase());
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
    const banned = getCharData(getCurrentCharName()).banned.map(x => x.toLowerCase());
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
    const cd = getCharData(getCurrentCharName());
    const seen = new Set();
    const collect = (src, into) => {
        for (const p of src) {
            const k = String(p).toLowerCase().trim();
            if (k && !seen.has(k)) { seen.add(k); into.push(k); }
        }
    };
    const banned = [];
    collect(cd.banned, banned);
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
    const cd = getCharData(getCurrentCharName());
    const out = [];
    const push = (p) => {
        const k = String(p).toLowerCase().trim();
        if (k && !out.includes(k)) out.push(k);
    };
    cd.banned.forEach(push);
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

    const phrases = getCharData(getCurrentCharName()).banned.map(b => b.toLowerCase());
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

function buildPhraseRegex(phrases) {
    if (!phrases.length) return null;
    const alts = [...phrases].sort((a, b) => b.length - a.length).map(escapeRe).join("|");
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
// in style.css. "system" leaves the panel in native ST appearance.
// ====================================================================
function applyTheme() {
    const theme = getSettings().theme || "cream";
    document.body.setAttribute("data-sk-theme", theme);
    document.querySelectorAll("#slop_killer_panel .sk_theme_btn").forEach(btn => {
        btn.classList.toggle("sk_theme_active", btn.dataset.theme === theme);
    });
}

// ====================================================================
// Manual ban / allow actions
// ====================================================================
function normalizePhrase(p) { return String(p).toLowerCase().trim(); }

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

// ====================================================================
// Settings panel
// ====================================================================
function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function buildPanel() {
    if (document.getElementById(`${MODULE_NAME}_panel`)) return;
    const host = document.getElementById("extensions_settings2") || document.getElementById("extensions_settings");
    if (!host) return;

    const s = getSettings();
    const html = `
    <div id="${MODULE_NAME}_panel" class="extension_settings">
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>Slop Killer</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">

                <label class="checkbox_label">
                    <input id="sk_enabled" type="checkbox" ${s.enabled ? "checked" : ""}>
                    <span>확장 활성화</span>
                </label>

                <hr>
                <h4>테마</h4>
                <div class="sk_theme_picker">
                    <button class="sk_theme_btn" data-theme="system" title="System (기본)"></button>
                    <button class="sk_theme_btn" data-theme="mono"   title="Mono (흑백)"></button>
                    <button class="sk_theme_btn" data-theme="cream"  title="Cream (베이지)"></button>
                    <button class="sk_theme_btn" data-theme="peach"  title="Peach (피치)"></button>
                    <button class="sk_theme_btn" data-theme="lilac"  title="Lilac (연보라)"></button>
                </div>

                <hr>
                <h4>감지 설정</h4>
                <label>구절 길이 — 최소 <span id="sk_minN_val">${s.minN}</span> 단어</label>
                <input id="sk_minN" type="range" min="1" max="5" value="${s.minN}" class="sk_slider">
                <label>구절 길이 — 최대 <span id="sk_maxN_val">${s.maxN}</span> 단어</label>
                <input id="sk_maxN" type="range" min="1" max="6" value="${s.maxN}" class="sk_slider">
                <label>슬롭 판정 임계 — <span id="sk_threshold_val">${s.threshold}</span>회 이상</label>
                <input id="sk_threshold" type="range" min="2" max="15" value="${s.threshold}" class="sk_slider">
                <label>스캔 범위 — 최근 <span id="sk_scanDepth_val">${s.scanDepth}</span>개 메시지</label>
                <input id="sk_scanDepth" type="range" min="5" max="200" step="5" value="${s.scanDepth}" class="sk_slider">

                <hr>
                <h4>프롬프트 주입</h4>
                <label class="checkbox_label">
                    <input id="sk_injectEnabled" type="checkbox" ${s.injectEnabled ? "checked" : ""}>
                    <span>생성 직전 "이 표현 피하라" 자동 주입</span>
                </label>
                <label>주입 최대 개수 — <span id="sk_maxInject_val">${s.maxInject}</span>개</label>
                <input id="sk_maxInject" type="range" min="1" max="40" value="${s.maxInject}" class="sk_slider">
                <label>주입 문구 (템플릿)</label>
                <p class="sk_hint"><code>{{banned}}</code> 수동 금지어, <code>{{slop}}</code> 자동 감지어, <code>{{phrases}}</code> 둘 다. 목록이 비면 그 줄은 자동 생략됩니다.</p>
                <textarea id="sk_injectTemplate" class="text_pole sk_template" rows="4" spellcheck="false">${escapeHtml(s.injectTemplate)}</textarea>
                <button id="sk_injectReset" class="menu_button sk_reset_btn">기본 문구로 복원</button>

                <hr>
                <h4>반복 패널티 부스트</h4>
                <label class="checkbox_label">
                    <input id="sk_penaltyEnabled" type="checkbox" ${s.penaltyEnabled ? "checked" : ""}>
                    <span>슬롭 감지 시 frequency / presence penalty 자동 상승</span>
                </label>
                <p class="sk_hint">OpenAI 호환·DeepSeek 백엔드에서만 작동합니다. Gemini·Claude는 무시됩니다.</p>
                <label>부스트 강도 — <span id="sk_penaltyBoost_val">${s.penaltyBoost}</span></label>
                <input id="sk_penaltyBoost" type="range" min="0.1" max="1.0" step="0.1" value="${s.penaltyBoost}" class="sk_slider">

                <hr>
                <h4>자동 리롤 (금지어 강제)</h4>
                <label class="checkbox_label">
                    <input id="sk_autoReroll" type="checkbox" ${s.autoReroll ? "checked" : ""}>
                    <span>수동 금지어가 출력에 나오면 자동으로 다시 생성</span>
                </label>
                <p class="sk_hint">금지어 직전 문장까지 남기고 그 뒤만 이어쓰기(continue)로 재생성합니다. 토큰이 추가로 소모됩니다.</p>
                <label>최대 재시도 — <span id="sk_rerollMax_val">${s.rerollMax}</span>회</label>
                <input id="sk_rerollMax" type="range" min="1" max="5" value="${s.rerollMax}" class="sk_slider">

                <hr>
                <h4>하이라이트</h4>
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

                <hr>
                <h4>현재 캐릭터: <span id="sk_charname" style="color:var(--SmartThemeQuoteColor);"></span></h4>
                <p class="sk_hint">감지된 반복 표현 (빈도순). 🚫 = 금지어로 추가, ✓ = 이건 슬롭 아님(제외)</p>
                <div id="sk_ranking" class="sk_ranking"></div>
                <button id="sk_rescan" class="menu_button sk_rescan_btn">다시 스캔</button>

                <hr>
                <h4>등록된 금지어</h4>
                <p class="sk_hint">위 자동 감지 목록에서 🚫를 누르거나, 아래에 직접 입력해 추가할 수 있습니다. 출처와 무관하게 모두 여기 모입니다.</p>
                <div style="display:flex; gap:6px;">
                    <input id="sk_ban_input" type="text" class="text_pole" placeholder="금지할 표현 입력" style="flex:1;">
                    <button id="sk_ban_add" class="menu_button">추가</button>
                </div>
                <div id="sk_banned_list" class="sk_chips"></div>

                <h4 style="margin-top:10px;">허용어 (슬롭 아님)</h4>
                <div id="sk_allowed_list" class="sk_chips"></div>

            </div>
        </div>
    </div>`;

    host.insertAdjacentHTML("beforeend", html);
    bindPanel();
    renderPanel();
}

function bindPanel() {
    const s = getSettings();

    const bindCheckbox = (id, key, after) => {
        const el = document.getElementById(id);
        el.addEventListener("change", () => {
            s[key] = el.checked;
            save();
            after?.();
        });
    };

    const bindSlider = (id, key, parser = parseInt, after) => {
        const el = document.getElementById(id);
        const lbl = document.getElementById(`${id}_val`);
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

    const banInput = document.getElementById("sk_ban_input");
    const doAdd = () => { if (banInput.value.trim()) { addBanned(banInput.value); banInput.value = ""; } };
    document.getElementById("sk_ban_add").addEventListener("click", doAdd);
    banInput.addEventListener("keydown", (e) => { if (e.key === "Enter") doAdd(); });

    document.querySelectorAll("#slop_killer_panel .sk_theme_btn").forEach(btn => {
        btn.addEventListener("click", () => {
            s.theme = btn.dataset.theme;
            applyTheme();
            save();
        });
    });
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
        box.innerHTML = `<div class="sk_hint">감지된 반복 표현 없음 (임계 ${getSettings().threshold}회 이상)</div>`;
        return;
    }

    box.innerHTML = list.slice(0, 40).map(({ phrase, count }) => `
        <div class="sk_rank_row">
            <span class="sk_rank_count">×${count}</span>
            <span class="sk_rank_phrase">${escapeHtml(phrase)}</span>
            <button class="menu_button sk_ban_btn" data-p="${escapeHtml(phrase)}" title="금지어로 추가">🚫</button>
            <button class="menu_button sk_allow_btn" data-p="${escapeHtml(phrase)}" title="이건 슬롭 아님">✓</button>
        </div>`).join("");

    box.querySelectorAll(".sk_ban_btn").forEach(b =>
        b.addEventListener("click", () => addBanned(b.dataset.p)));
    box.querySelectorAll(".sk_allow_btn").forEach(b =>
        b.addEventListener("click", () => addAllowed(b.dataset.p)));
}

function renderChips() {
    const cd = getCharData(getCurrentCharName());
    renderChipBox("sk_banned_list", cd.banned, "banned");
    renderChipBox("sk_allowed_list", cd.allowed, "allowed");
}

function renderChipBox(id, arr, kind) {
    const box = document.getElementById(id);
    if (!box) return;
    if (!arr.length) { box.innerHTML = `<span class="sk_hint">없음</span>`; return; }
    box.innerHTML = arr.map(p =>
        `<span class="sk_chip">${escapeHtml(p)}<button data-p="${escapeHtml(p)}" data-kind="${kind}" title="제거">×</button></span>`
    ).join("");
    box.querySelectorAll("button").forEach(b =>
        b.addEventListener("click", () => removeFrom(b.dataset.kind, b.dataset.p)));
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
