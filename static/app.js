/* GFlava-Quant: Logik der Oberfläche. Läuft ohne Build-Schritt direkt im Browser. */
"use strict";

/* =====================================================================
   Zustand und Hilfsfunktionen
   ===================================================================== */
let FORMATS = {}, EXCLUDES = {};
let MODELS = [], MODEL_ROOTS_COUNT = 0, LOGS_COUNT = 0;
let CTQ_FOUND = false, CTQ_PATH = null;
let PRESETS = [];

// Aktuelle Auswahl. Dateipfade stehen in den Eingabefeldern, nicht hier.
const S = { format: 'int8_convrot', exclude: 'qwen21_custom' };
let selectedModel = null;      // Eintrag aus der Modell-Liste (oder null bei Pfad von Hand)
let MODEL_INFO = null;         // Ergebnis von /api/detect_model für die aktuelle Datei
let DETECTED_KEY = null;       // erkannter Layer-Schutz (oder null)
let DETECT_STATE = 'none';     // none | detecting | found | unknown
let START_ERROR = '';

function readStored(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; } }
function writeStored(key, value) { try { localStorage.setItem(key, value); } catch (e) {} }

let LANG = readStored('lang', 'de');
let THEME = readStored('theme', 'dark');
let MINIMAL = readStored('minimal', '0') === '1';

const $ = (id) => document.getElementById(id);

function t(key, ...args) {
  const dict = I18N[LANG] || I18N.de;
  const val = (dict[key] !== undefined) ? dict[key] : I18N.de[key];
  return typeof val === 'function' ? val(...args) : val;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function toast(text, ms) {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), ms || 2200);
}

async function api(path, body, method) {
  const opts = body === undefined && !method ? {} : {
    method: method || 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
  };
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (e) {}
  return { ok: res.ok, status: res.status, data };
}

function formatLabelFor(entry) { return LANG === 'en' ? (entry.label_en || entry.label) : entry.label; }

function formatBytes(bytes) {
  const locale = LANG === 'en' ? 'en-US' : 'de-DE';
  const gb = bytes / (1024 ** 3);
  if (gb >= 1) return gb.toLocaleString(locale, { maximumFractionDigits: 1 }) + ' GB';
  const mb = bytes / (1024 ** 2);
  return mb.toLocaleString(locale, { maximumFractionDigits: mb < 10 ? 1 : 0 }) + ' MB';
}

function inputPathValue() { return $('inputPath').value.trim().replace(/^"(.*)"$/, '$1'); }
function outputPathValue() { return $('outputPath').value.trim().replace(/^"(.*)"$/, '$1'); }

function applyStaticI18n() {
  document.title = 'GFlava-Quant';
  document.documentElement.lang = LANG;
  document.querySelectorAll('[data-i18n]').forEach((el) => { el.innerHTML = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-ph]').forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  document.querySelectorAll('[data-i18n-title]').forEach((el) => {
    const txt = t(el.dataset.i18nTitle);
    el.title = txt; el.setAttribute('aria-label', txt);
  });
}

/* =====================================================================
   Theme, Minimal-Modus, Hintergrund
   ===================================================================== */
const BG_PALETTES = {
  dark: { base: [10, 11, 17], stops: [[96, 32, 118], [36, 46, 124], [16, 96, 140]], strength: 0.5, dot: 'rgba(255,255,255,0.05)' },
  light: { base: [243, 244, 248], stops: [[206, 168, 232], [170, 184, 236], [152, 210, 232]], strength: 0.45, dot: 'rgba(20,24,40,0.07)' },
};

// Dreiecks-Mosaik als quantisierter Farbverlauf (Position und Helligkeit auf feste Stufen gerundet),
// dazu das Punktraster. Statisch: es wird nur beim Laden, Theme-Wechsel und Skalieren neu gezeichnet.
function drawBackground() {
  const cv = $('bgCanvas'); if (!cv) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const w = window.innerWidth, h = window.innerHeight;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pal = document.documentElement.getAttribute('data-theme') === 'light' ? BG_PALETTES.light : BG_PALETTES.dark;
  const lerp = (a, b, k) => a + (b - a) * k;
  const hash = (a, b) => { const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return x - Math.floor(x); };
  const rgb = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
  ctx.fillStyle = rgb(pal.base); ctx.fillRect(0, 0, w, h);
  const drawDots = () => { ctx.fillStyle = pal.dot; for (let y = 12; y < h; y += 24) for (let x = 12; x < w; x += 24) ctx.fillRect(x, y, 1.5, 1.5); };
  if (document.documentElement.hasAttribute('data-minimal')) { drawDots(); return; }
  const side = 64, th = side * Math.sqrt(3) / 2, HUE_STEPS = 7, LIGHT_STEPS = 4;
  for (let r = -1; r * th < h + th; r++) {
    for (let c = -2; c * side / 2 < w + side; c++) {
      const x = c * side / 2, y = r * th, up = ((c + r) % 2 + 2) % 2 === 0, cx = x + side / 2, cy = y + th / 2;
      const k0 = Math.round(Math.min(1, Math.max(0, (cx / w) * 0.55 + (1 - cy / h) * 0.45)) * HUE_STEPS) / HUE_STEPS;
      const col = k0 < 0.5 ? pal.stops[0].map((v, i) => lerp(v, pal.stops[1][i], k0 * 2)) : pal.stops[1].map((v, i) => lerp(v, pal.stops[2][i], (k0 - 0.5) * 2));
      const level = Math.round(hash(r, c) * LIGHT_STEPS) / LIGHT_STEPS;
      const vignette = 1 - 0.5 * Math.pow(Math.min(1, Math.max(0, cy / h)), 3);
      const k = pal.strength * (0.42 + 0.4 * level) * vignette;
      const fill = rgb(pal.base.map((v, i) => lerp(v, col[i], k)));
      ctx.beginPath();
      if (up) { ctx.moveTo(x, y + th); ctx.lineTo(x + side / 2, y); ctx.lineTo(x + side, y + th); }
      else { ctx.moveTo(x, y); ctx.lineTo(x + side, y); ctx.lineTo(x + side / 2, y + th); }
      ctx.closePath(); ctx.fillStyle = fill; ctx.strokeStyle = fill; ctx.lineWidth = 1; ctx.fill(); ctx.stroke();
    }
  }
  drawDots();
}
let bgResizeTimer = null;
window.addEventListener('resize', () => { clearTimeout(bgResizeTimer); bgResizeTimer = setTimeout(drawBackground, 150); });

function withoutTransitions(fn) {
  const root = document.documentElement;
  root.classList.add('no-transitions'); fn();
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove('no-transitions')));
}
function applyTheme() {
  if (THEME === 'light') document.documentElement.setAttribute('data-theme', 'light');
  else document.documentElement.removeAttribute('data-theme');
  drawBackground();
}
function toggleTheme() { THEME = THEME === 'light' ? 'dark' : 'light'; writeStored('theme', THEME); withoutTransitions(applyTheme); }
function updateMinimalButton() {
  const btn = $('minimalToggleBtn'), label = MINIMAL ? t('minimalOffTitle') : t('minimalOnTitle');
  btn.title = label; btn.setAttribute('aria-label', label); btn.setAttribute('aria-pressed', MINIMAL ? 'true' : 'false');
}
function applyMinimal() {
  if (MINIMAL) document.documentElement.setAttribute('data-minimal', ''); else document.documentElement.removeAttribute('data-minimal');
  updateMinimalButton(); drawBackground();
}
function toggleMinimal() { MINIMAL = !MINIMAL; writeStored('minimal', MINIMAL ? '1' : '0'); withoutTransitions(applyMinimal); }

/* =====================================================================
   Einstellungen: Modell-Ordner, ComfyUI-Pfad, ctq-Pfad, Systemcheck
   ===================================================================== */
let SETTINGS = { model_roots: [], comfyui_root: '', ctq_path_override: '' };
let CTQ_AUTO_PATH = null, LAST_SYSTEM_CHECKS = null, LAST_CHECKED_AT = null;

function settingsMsg(text, kind) {
  const el = $('settingsError');
  if (!text) { el.hidden = true; return; }
  el.textContent = text; el.className = 'settings-msg' + (kind === 'info' ? ' info' : ''); el.hidden = false;
}
function settingsIsOpen() { return $('settingsOverlay').classList.contains('open'); }
function updateCtqAutoHint() {
  $('settingsCtqAutoHint').textContent = CTQ_AUTO_PATH ? t('settingsCtqAutoFoundTemplate', CTQ_AUTO_PATH) : t('settingsCtqAutoMissing');
}
function renderRootList() {
  const list = $('settingsRootList');
  if (!SETTINGS.model_roots.length) { list.innerHTML = `<div class="modal-hint">${t('settingsNoRoots')}</div>`; return; }
  list.innerHTML = SETTINGS.model_roots.map((r, i) => `
    <div class="root-row">
      <svg class="root-row-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
      <div class="root-row-text"><div class="root-row-label">${escapeHtml(r.label)}</div><div class="root-row-path">${escapeHtml(r.path)}</div></div>
      <button type="button" class="secondary icon-btn" data-remove-idx="${i}" title="${escapeHtml(t('settingsRemoveRoot'))}" aria-label="${escapeHtml(t('settingsRemoveRoot'))}">&times;</button>
    </div>`).join('');
  list.querySelectorAll('[data-remove-idx]').forEach((btn) => btn.addEventListener('click', (e) => {
    SETTINGS.model_roots.splice(+e.currentTarget.dataset.removeIdx, 1); renderRootList();
  }));
}
function addModelRootFromPath(path) {
  const name = path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path;
  SETTINGS.model_roots.push({ key: '', label: name, path }); renderRootList();
}
async function openSettings(showFirstRunHint) {
  settingsMsg('');
  // Ergebnis der automatischen Prüfung beim Start gleich zeigen, statt eine leere Liste.
  if (LAST_SYSTEM_CHECKS) renderSystemCheck(LAST_SYSTEM_CHECKS); else $('systemCheckList').innerHTML = '';
  try {
    const { data } = await api('/api/settings');
    SETTINGS = { model_roots: (data.model_roots || []).map((r) => ({ ...r })), comfyui_root: data.comfyui_root || '', ctq_path_override: data.ctq_path_override || '' };
    CTQ_AUTO_PATH = data.ctq_auto_path || null;
  } catch (e) { SETTINGS = { model_roots: [], comfyui_root: '', ctq_path_override: '' }; CTQ_AUTO_PATH = null; }
  $('comfyuiRootInput').value = SETTINGS.comfyui_root;
  $('ctqOverrideInput').value = SETTINGS.ctq_path_override;
  updateCtqAutoHint(); renderRootList();
  $('settingsOverlay').classList.add('open');
  $('settingsBtn').setAttribute('aria-expanded', 'true');
  if (showFirstRunHint) settingsMsg(t('firstRunBanner'), 'info');
}
function closeSettings() {
  $('settingsOverlay').classList.remove('open');
  $('settingsBtn').setAttribute('aria-expanded', 'false');
}
async function saveSettings() {
  settingsMsg('');
  const payload = { model_roots: SETTINGS.model_roots, comfyui_root: $('comfyuiRootInput').value.trim(), ctq_path_override: $('ctqOverrideInput').value.trim() };
  try {
    const { ok, data } = await api('/api/settings', payload);
    if (!ok) { settingsMsg(data.error || t('settingsSaveError')); return; }
    closeSettings(); await loadConfig(); await loadModels();
  } catch (e) { settingsMsg(t('settingsSaveError')); }
}

const CHECK_ICON_SVG = {
  ok: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
  warn: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
  unknown: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.5-2 1.8-2 3.4"></path><line x1="12" y1="16.5" x2="12.01" y2="16.5"></line></svg>',
};
CHECK_ICON_SVG.missing = CHECK_ICON_SVG.warn;

function renderSystemCheck(checks) {
  LAST_SYSTEM_CHECKS = checks;
  const listEl = $('systemCheckList');
  listEl.innerHTML = checks.map((c) => {
    const detail = LANG === 'en' ? (c.detail_en || c.detail) : c.detail;
    const actions = [];
    if (c.can_install) actions.push(`<button type="button" class="primary" data-check-action="install" data-check-id="${escapeHtml(c.id)}">${t('settingsInstall')}</button>`);
    if (c.can_update) actions.push(`<button type="button" class="secondary" data-check-action="update" data-check-id="${escapeHtml(c.id)}">${t('settingsUpdate')}</button>`);
    if (c.id === 'app' && c.update_available && c.release_url) actions.push(`<button type="button" class="primary" data-open-url="${escapeHtml(c.release_url)}">${t('openReleaseBtn')}</button>`);
    return `<div class="check-row"><div class="check-icon ${escapeHtml(c.status)}">${CHECK_ICON_SVG[c.status] || CHECK_ICON_SVG.unknown}</div>
      <div class="check-row-text"><div class="check-row-label">${escapeHtml(formatLabelFor(c))}</div><div class="check-row-detail">${escapeHtml(detail)}</div></div>
      <div class="check-row-actions">${actions.join('')}</div></div>`;
  }).join('');
  listEl.querySelectorAll('[data-check-action]').forEach((btn) => btn.addEventListener('click', () => performCheckAction(btn.dataset.checkId, btn.dataset.checkAction, btn)));
  listEl.querySelectorAll('[data-open-url]').forEach((btn) => btn.addEventListener('click', () => window.open(btn.dataset.openUrl, '_blank', 'noopener')));
  renderCheckTime();
}
function renderCheckTime() {
  const el = $('systemCheckTime');
  if (!LAST_CHECKED_AT) { el.textContent = ''; return; }
  const time = new Date(LAST_CHECKED_AT * 1000).toLocaleTimeString(LANG === 'en' ? 'en-US' : 'de-DE', { hour: '2-digit', minute: '2-digit' });
  el.textContent = t('lastCheckedTemplate', time);
}
// silent: automatische Prüfung beim Start, zeigt nur bei Updates etwas an und darf ein
// höchstens eine Stunde altes Ergebnis vom Server nehmen. Sonst: Knopf, prüft immer neu.
async function runSystemCheck(silent) {
  const btn = $('runSystemCheckBtn');
  if (!silent) { $('systemCheckList').innerHTML = `<div class="modal-hint">${t('settingsChecking')}</div>`; btn.disabled = true; }
  try {
    const { ok, data } = await api('/api/system_check' + (silent ? '?cached=1' : ''));
    if (!ok) throw new Error('HTTP');
    LAST_CHECKED_AT = data.checked_at || null;
    renderSystemCheck(data.checks || []); renderUpdateNotice();
    if (!silent && !availableUpdates().length) toast(t('updatesNoneToast'));
  } catch (e) {
    if (!silent) $('systemCheckList').innerHTML = `<div class="modal-hint" style="color:var(--err)">${t('settingsCheckFailed')}</div>`;
  } finally { btn.disabled = false; }
}

/* ---- Update-Hinweis: Pille in der Kopfzeile und Banner über der Seite ---- */
const availableUpdates = () => (LAST_SYSTEM_CHECKS || []).filter((c) => c.update_available);
// Ausgeblendet wird nur genau dieser Stand. Kommt eine neuere Version, erscheint das Banner wieder.
const updateSignature = (list) => list.map((c) => c.id + ':' + (c.latest_version || '')).join('|');
function renderUpdateNotice() {
  const list = availableUpdates(), pill = $('updatePill'), banner = $('updateBanner');
  pill.hidden = !list.length;
  if (!list.length) { banner.hidden = true; return; }
  $('updatePillText').textContent = t('updatePillTemplate', list.length);
  pill.title = list.map((c) => formatLabelFor(c)).join(', ');
  $('updateBannerText').innerHTML = '<b>' + escapeHtml(t('updateBannerTemplate', list.length)) + '</b> '
    + list.map((c) => escapeHtml(formatLabelFor(c) + ': ' + (LANG === 'en' ? (c.detail_en || c.detail) : c.detail))).join(' · ');
  banner.hidden = readStored('updateDismissed', '') === updateSignature(list);
}
async function showUpdates() {
  if (!settingsIsOpen()) await openSettings(false);
  // Nur das Menü scrollen; scrollIntoView würde auch die Seite dahinter verschieben.
  $('settingsOverlay').scrollTo({ top: $('updatesSection').offsetTop - 8, behavior: 'smooth' });
}
async function performCheckAction(id, action, btn) {
  settingsMsg(''); btn.disabled = true; btn.textContent = t('settingsWorking');
  try {
    const { ok, data } = await api('/api/system_check/action', { id, action });
    if (!ok) settingsMsg(data.error || t('settingsActionFailed'));
  } catch (e) { settingsMsg(t('settingsActionFailed')); }
  await runSystemCheck();
  if (id === 'ctq') await loadConfig();
}
// Echte native Windows-Dialoge: der Server öffnet sie per PowerShell (siehe /api/browse_native).
async function pickNative(body) {
  try {
    const { ok, data } = await api('/api/browse_native', body);
    if (!ok) { settingsMsg(data.error || t('settingsActionFailed')); return null; }
    return data.path || null;
  } catch (e) { settingsMsg(t('settingsActionFailed')); return null; }
}
const pickNativeFolder = (title, initialDir) => pickNative({ mode: 'folder', title, initial_dir: initialDir || '', select_hint: t('folderPickerHint') });
const pickNativeFile = (title, filter, initialDir) => pickNative({ mode: 'file', title, filter, initial_dir: initialDir || '' });

function renderCtqStatus() {
  const el = $('ctqStatus'), txt = $('ctqStatusText');
  el.classList.remove('ok', 'missing');
  if (CTQ_FOUND) { txt.textContent = t('ctqFoundShort'); el.title = t('ctqFound') + CTQ_PATH; el.classList.add('ok'); }
  else { txt.textContent = t('ctqMissingShort'); el.title = t('ctqMissing'); el.classList.add('missing'); }
}
/* =====================================================================
   Format (Schritt 2) und Größenschätzung
   ===================================================================== */
// Effektive Bits pro Gewicht inklusive Skalen-Aufwand. Nur Näherung: daraus wird die Größenschätzung.
const FORMAT_BITS = { int8_convrot: 8, int8_tensorwise: 8, int8_block: 8.25, fp8: 8, nvfp4: 4.5, mxfp8: 8.25 };
// Feste Reihenfolge der Karten (Flask liefert JSON-Schlüssel alphabetisch sortiert).
const FORMAT_ORDER = ['int8_convrot', 'int8_tensorwise', 'int8_block', 'fp8', 'nvfp4', 'mxfp8'];
const bitsOf = (key) => FORMAT_BITS[key] || 8;
const pctOf = (key) => Math.round(bitsOf(key) / 16 * 100);

function fmtText(key) {
  const f = (I18N[LANG] || I18N.de).fmt || {}, fd = I18N.de.fmt[key] || {};
  return Object.assign({}, fd, f[key] || {});
}
function fmtName(key) { const x = fmtText(key); return x.name || (FORMATS[key] ? formatLabelFor(FORMATS[key]) : key); }

function renderFormatCards() {
  const grid = $('formatGrid');
  const keys = FORMAT_ORDER.filter((k) => FORMATS[k]).concat(Object.keys(FORMATS).filter((k) => !FORMAT_ORDER.includes(k)));
  grid.innerHTML = keys.map((key) => {
    const x = fmtText(key), f = FORMATS[key], on = Math.round(bitsOf(key));
    const cells = Array.from({ length: 16 }, (_, i) => `<i class="${i < on ? 'on' : ''}"></i>`).join('');
    return `<button type="button" class="format-card" data-format="${escapeHtml(key)}" aria-pressed="${S.format === key}">
      <div class="fc-top"><span class="fc-name">${escapeHtml(x.name || key)}</span><span class="fc-size">${escapeHtml(t('sizeApprox', pctOf(key) + ' %'))}</span></div>
      <div class="fc-bits" aria-hidden="true">${cells}</div>
      <p class="fc-desc">${escapeHtml(x.desc || '')}</p>
      <p class="fc-hw">${f && f.verified ? `<span class="badge ok">${escapeHtml(t('tagRecommended'))}</span>` : ''}${escapeHtml(x.hw || '')}</p>
    </button>`;
  }).join('');
  grid.querySelectorAll('[data-format]').forEach((btn) => btn.addEventListener('click', () => setFormat(btn.dataset.format)));
}

function renderGroupsizeSeg() {
  const seg = $('groupsizeSeg'), cur = String($('convrotGroupsizeVal').value);
  seg.innerHTML = ['4', '16', '64', '256', '1024'].map((v) => `<button type="button" data-gs="${v}" aria-pressed="${cur === v}">${v}</button>`).join('');
  seg.querySelectorAll('[data-gs]').forEach((b) => b.addEventListener('click', () => { $('convrotGroupsizeVal').value = b.dataset.gs; renderGroupsizeSeg(); onSettingsChanged(); }));
}

function renderFormatDetail() {
  const key = S.format, f = FORMATS[key];
  if (!f) return;
  const badge = f.verified ? `<span class="badge ok">${t('badgeVerified')}</span>` : `<span class="badge warn">${t('badgeExperimental')}</span>`;
  $('formatHint').innerHTML = badge + ' ' + (f.verified ? t('formatVerifiedText') : t('formatUnverifiedText'));
  $('groupsizeRow').hidden = key !== 'int8_convrot';
  $('blockSizeRow').hidden = key !== 'int8_block';
}

function renderGlossary() {
  $('glossaryList').innerHTML = (t('glossary') || []).map(([term, text]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(text)}</dd></div>`).join('');
}

function setFormat(key) {
  if (!FORMATS[key]) return;
  S.format = key; START_ERROR = '';
  $('formatGrid').querySelectorAll('[data-format]').forEach((b) => b.setAttribute('aria-pressed', b.dataset.format === key));
  renderFormatDetail();
  onSettingsChanged();
}

// Bit-Raster und Größenschätzung im Hero.
function updateShrink() {
  const key = S.format, bits = bitsOf(key);
  const src = $('srcCells'), tgt = $('tgtCells');
  if (!src.children.length) {
    src.innerHTML = Array.from({ length: 16 }, () => '<i class="cell"></i>').join('');
    tgt.innerHTML = Array.from({ length: 16 }, () => '<i class="cell"></i>').join('');
  }
  [...tgt.children].forEach((c, i) => c.classList.toggle('on', i < Math.round(bits)));
  $('srcName').textContent = t('shrinkOriginal');
  $('tgtName').textContent = fmtName(key);
  const info = MODEL_INFO, size = info && info.size_bytes, b16 = info && info.bytes_16bit_2d;
  const reliable = !!(size && b16 && b16 / size > 0.5);
  if (reliable) {
    const after = size - b16 * (1 - bits / 16);
    $('sizeBefore').textContent = formatBytes(size);
    $('sizeAfter').textContent = t('sizeApprox', formatBytes(after));
    $('shrinkNote').textContent = t('shrinkNoteModel');
  } else if (size) {
    $('sizeBefore').textContent = formatBytes(size);
    $('sizeAfter').textContent = t('sizeApprox', pctOf(key) + ' %');
    $('shrinkNote').textContent = t('shrinkNoteNo16');
  } else {
    $('sizeBefore').textContent = '100 %';
    $('sizeAfter').textContent = t('sizeApprox', pctOf(key) + ' %');
    $('shrinkNote').textContent = t('shrinkNoteNoModel');
  }
  updateRecommend();
}

function recommendTarget() { return { format: 'int8_convrot', exclude: DETECTED_KEY || null }; }
function recommendIsActive() {
  if (!hasModel()) return false;
  const r = recommendTarget();
  return S.format === r.format && (!r.exclude || S.exclude === r.exclude);
}
function updateRecommend() {
  const btn = $('recommendBtn'), note = $('recommendNote');
  const has = hasModel(), active = recommendIsActive();
  btn.disabled = !has;
  btn.classList.toggle('on', active);
  btn.textContent = active ? t('recommendActive') : t('recommendBtn');
  if (!has) note.textContent = t('recommendNeedModel');
  else if (DETECT_STATE === 'detecting') note.textContent = t('typeDetecting');
  else if (DETECTED_KEY && EXCLUDES[DETECTED_KEY]) note.textContent = t('recommendDetected', EXCLUDES[DETECTED_KEY].chip || formatLabelFor(EXCLUDES[DETECTED_KEY]));
  else note.textContent = t('recommendUndetected');
}
function applyRecommended() {
  if (!hasModel()) return;
  const r = recommendTarget();
  S.format = r.format; START_ERROR = '';
  if (r.exclude) S.exclude = r.exclude;
  renderFormatCards(); renderFormatDetail(); renderChips(); updateExcludeHint();
  onSettingsChanged();
}

/* =====================================================================
   Layer-Schutz (Schritt 3)
   ===================================================================== */
const CHIP_GROUPS = [
  ['groupImage', ['qwen21_custom', 'flux1_custom', 'sdxl_custom', 'ctq_qwen', 'ctq_zimage', 'ctq_zimage_refiner', 'ctq_boogu', 'ctq_anima', 'ctq_lens', 'ctq_flux2', 'ctq_distillation_large', 'ctq_distillation_small', 'ctq_nerf_large', 'ctq_nerf_small', 'ctq_radiance', 'ctq_krea2', 'ctq_ideogram4']],
  ['groupVideo', ['ctq_wan', 'ctq_hunyuan', 'ctq_minimaxh3', 'ctq_ltxv2']],
  ['groupText', ['ctq_gemma4', 'ctq_qwen_vlm', 'ctq_t5xxl', 'ctq_mistral', 'ctq_visual', 'ctq_generic_text']],
];
const CHIP_LABEL_DE = { ctq_nerf_large: 'NeRF (groß)', ctq_distillation_large: 'Chroma (groß)' };
const MODEL_TYPE_CLASS = {
  ctq_anima: 'type-anima', ctq_flux2: 'type-flux2', flux1_custom: 'type-flux1', sdxl_custom: 'type-sdxl',
  qwen21_custom: 'type-qwen', ctq_qwen: 'type-qwen', ctq_zimage: 'type-zimage', ctq_zimage_refiner: 'type-zimage',
  ctq_wan: 'type-video', ctq_hunyuan: 'type-video', ctq_minimaxh3: 'type-video', ctq_ltxv2: 'type-video',
};
function chipLabel(key) {
  const v = EXCLUDES[key]; if (!v) return key;
  if (key === 'none') return t('excludeNoneChip');
  if (key === 'custom') return t('excludeCustomChip');
  if (LANG === 'de' && CHIP_LABEL_DE[key]) return CHIP_LABEL_DE[key];
  return v.chip || formatLabelFor(v);
}
function chipHTML(key) {
  const v = EXCLUDES[key]; if (!v) return '';
  const tip = key === 'none' || key === 'custom' ? '' : formatLabelFor(v);
  return `<button type="button" class="chip" data-excl="${escapeHtml(key)}" title="${escapeHtml(tip)}" aria-pressed="${S.exclude === key}">${escapeHtml(chipLabel(key))}</button>`;
}
function renderChips() {
  $('chipGroups').innerHTML = CHIP_GROUPS.map(([titleKey, keys]) =>
    `<div class="chip-group"><div class="chip-group-title">${escapeHtml(t(titleKey))}</div><div class="chips">${keys.map(chipHTML).join('')}</div></div>`).join('');
  $('chipsOther').innerHTML = ['none', 'custom'].map(chipHTML).join('');
  document.querySelectorAll('[data-excl]').forEach((btn) => btn.addEventListener('click', () => setExclude(btn.dataset.excl)));
  $('customRegexRow').hidden = S.exclude !== 'custom';
}
function setExclude(key) {
  if (!EXCLUDES[key]) return;
  S.exclude = key; START_ERROR = '';
  document.querySelectorAll('[data-excl]').forEach((b) => b.setAttribute('aria-pressed', b.dataset.excl === key));
  $('customRegexRow').hidden = key !== 'custom';
  updateExcludeHint(); onSettingsChanged();
}

function updateExcludeBanner() {
  const el = $('excludeStatus');
  el.className = 'detect-banner';
  if (DETECT_STATE === 'detecting') el.textContent = t('excludeBannerDetecting');
  else if (DETECT_STATE === 'found' && DETECTED_KEY && EXCLUDES[DETECTED_KEY]) {
    el.classList.add('found'); el.innerHTML = t('excludeBannerFound', escapeHtml(chipLabel(DETECTED_KEY)));
  } else if (DETECT_STATE === 'unknown') { el.classList.add('unknown'); el.textContent = t('excludeBannerUnknown'); }
  else el.textContent = t('excludeBannerNoModel');
}
function updateExcludeHint() {
  updateExcludeBanner();
  const preset = EXCLUDES[S.exclude], hint = $('excludeHint');
  if (!preset) { hint.textContent = ''; updateAdvancedHint(); return; }
  if (preset.kind === 'regex') {
    const badge = preset.verified ? `<span class="badge ok">${t('badgeVerified')}</span>` : `<span class="badge warn">${t('badgeCommunityUnverified')}</span>`;
    hint.innerHTML = badge + ' ' + t('excludeActivePattern') + '<code>' + escapeHtml(preset.value) + '</code>' + t('excludeUnquantizedNote') + (preset.verified ? '' : t('excludeCheckComfyUI'));
  } else if (preset.kind === 'builtin') {
    const badge = preset.verified ? `<span class="badge ok">${t('badgeEndToEnd')}</span> ` : '';
    let desc;
    if (preset.filter_desc) {
      const note = LANG === 'en' ? preset.filter_desc_note_en : preset.filter_desc_note_de;
      desc = '<code>' + escapeHtml(preset.value) + '</code>: ' + escapeHtml(preset.filter_desc) + (note ? ' ' + escapeHtml(note) : '') + '.';
    } else desc = t('excludeUsesBuiltin') + '<code>' + escapeHtml(preset.value) + '</code>.';
    hint.innerHTML = badge + desc + t('excludeFullDetails');
  } else if (preset.kind === 'none') hint.textContent = t('excludeNoneText');
  else if (preset.kind === 'custom') hint.textContent = t('excludeCustomText');
  else hint.textContent = '';
  updateAdvancedHint();
}
function updateAdvancedHint() {
  const preset = EXCLUDES[S.exclude], box = $('advancedPresetHint');
  if (!preset || (preset.kind !== 'builtin' && preset.kind !== 'regex')) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = preset.kind === 'builtin' ? t('advancedBuiltinText', escapeHtml(preset.value)) : t('advancedRegexText');
}

/* =====================================================================
   Modellauswahl (Schritt 1)
   ===================================================================== */
let comboVisibleItems = [], comboActiveIndex = -1, modelDetectSeq = 0, detectTimer = null;

const hasModel = () => !!inputPathValue();
const modelDisplayName = (m) => (m.subdir ? m.subdir + '/' : '') + m.filename;
const modelMatches = (m, q) => !q || (modelDisplayName(m) + ' ' + m.root_label).toLowerCase().includes(q.toLowerCase());

function highlightMatch(text, query) {
  if (!query) return escapeHtml(text);
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return escapeHtml(text);
  return escapeHtml(text.slice(0, idx)) + '<mark>' + escapeHtml(text.slice(idx, idx + query.length)) + '</mark>' + escapeHtml(text.slice(idx + query.length));
}

async function loadModels() {
  const hint = $('modelSelectHint');
  hint.textContent = t('modelSearching');
  try {
    const { ok, data } = await api('/api/models');
    if (!ok) throw new Error('scan');
    MODELS = data.models || []; MODEL_ROOTS_COUNT = (data.roots || []).length;
    if (selectedModel) {
      const again = MODELS.find((m) => m.path === selectedModel.path);
      if (again) selectedModel = again;
      else { selectedModel = null; $('modelSearch').value = ''; }
    }
    hint.textContent = MODELS.length ? t('modelsFoundTemplate', MODELS.length, MODEL_ROOTS_COUNT) : t('modelsNoneFound');
  } catch (e) { hint.textContent = t('modelsScanError'); showManualPath(true); }
  updateStatsBar();
}

function renderComboPanel(query) {
  const panel = $('modelComboPanel'); panel.innerHTML = '';
  const filtered = MODELS.filter((m) => modelMatches(m, query));
  comboVisibleItems = filtered; comboActiveIndex = filtered.length ? 0 : -1;
  if (!filtered.length) {
    const empty = document.createElement('div'); empty.className = 'combo-empty';
    empty.textContent = MODELS.length ? t('comboNoMatches', query) : t('comboNoModels'); panel.appendChild(empty); return;
  }
  let lastGroup = null;
  filtered.forEach((m, idx) => {
    if (m.root_label !== lastGroup) {
      lastGroup = m.root_label;
      const label = document.createElement('div'); label.className = 'combo-group-label'; label.textContent = m.root_label; panel.appendChild(label);
    }
    const item = document.createElement('div');
    item.className = 'combo-item'; item.dataset.idx = idx; item.setAttribute('role', 'option');
    item.innerHTML = `<span>${highlightMatch(modelDisplayName(m), query)}</span><span class="combo-item-size">${m.size_mb != null ? escapeHtml(m.size_mb) + ' MB' : ''}</span>`;
    item.addEventListener('mousedown', (e) => { e.preventDefault(); selectModel(m); });
    item.addEventListener('mouseenter', () => { comboActiveIndex = idx; highlightActiveItem(); });
    panel.appendChild(item);
  });
  highlightActiveItem();
}
function highlightActiveItem() {
  document.querySelectorAll('#modelComboPanel .combo-item').forEach((el) => {
    const active = Number(el.dataset.idx) === comboActiveIndex;
    el.classList.toggle('active', active); el.setAttribute('aria-selected', active ? 'true' : 'false');
    if (active) el.scrollIntoView({ block: 'nearest' });
  });
}
// Beim Anklicken des Feldes mit schon gewähltem Modell die ganze Liste zeigen (nicht nur den einen Treffer).
function openComboPanel(showAll) {
  const search = $('modelSearch'), value = search.value.trim();
  const isSelectedName = !!selectedModel && value === modelDisplayName(selectedModel);
  renderComboPanel(showAll === true && isSelectedName ? '' : value);
  $('modelComboPanel').hidden = false; search.setAttribute('aria-expanded', 'true');
}
function closeComboPanel() { $('modelComboPanel').hidden = true; $('modelSearch').setAttribute('aria-expanded', 'false'); }

function showManualPath(show) {
  $('manualPathRow').hidden = !show;
  $('manualPathToggle').textContent = show ? t('manualToggleHide') : t('manualToggleShow');
}
function showOutputRow(show) {
  $('outputPathRow').hidden = !show;
  $('outputToggle').textContent = show ? t('outputToggleHide') : t('outputToggleShow');
}

// Name der Ausgabedatei, wenn das Feld leer bleibt (gleiche Regel wie auf dem Server).
function autoOutputPath() {
  const p = inputPathValue(); if (!p) return '';
  const i = Math.max(p.lastIndexOf('\\'), p.lastIndexOf('/')), dot = p.lastIndexOf('.');
  const base = dot > i ? p.slice(0, dot) : p, ext = dot > i ? p.slice(dot) : '';
  return base + '_' + S.format + ext;
}
function updateModelPreview() {
  const el = $('modelPathPreview'), path = inputPathValue();
  if (!path) { el.hidden = true; $('outputPathHint').innerHTML = t('outputAutoHintNone'); return; }
  const out = outputPathValue() || autoOutputPath();
  const root = selectedModel ? ` <span class="badge info">${escapeHtml(selectedModel.root_label)}</span>` : '';
  el.innerHTML = t('fullPathLabel') + '<code>' + escapeHtml(path) + '</code>' + root + '<br>' + t('outputAutoHint', escapeHtml(out));
  el.hidden = false;
  $('outputPathHint').innerHTML = t('outputAutoHint', escapeHtml(autoOutputPath()));
}

function setDetectState(state, key) {
  DETECT_STATE = state; DETECTED_KEY = key || null;
  const pill = $('modelTypePill');
  if (state === 'detecting') {
    pill.className = 'pill loading'; pill.innerHTML = '<span class="dot"></span><span>' + escapeHtml(t('typeDetecting')) + '</span>'; pill.hidden = false;
  } else if (state === 'found' && key && EXCLUDES[key]) {
    pill.className = 'pill ' + (MODEL_TYPE_CLASS[key] || 'type-other');
    pill.innerHTML = '<span class="dot"></span><span>' + escapeHtml(t('typeDetected', chipLabel(key))) + '</span>'; pill.hidden = false;
  } else pill.hidden = true;
  updateExcludeBanner(); updateRecommend();
}

// Liest den Header der gewählten Datei (Architektur + Größen) und setzt den passenden Layer-Schutz.
async function detectModel(path) {
  const seq = ++modelDetectSeq;
  MODEL_INFO = null; setDetectState('detecting'); updateShrink();
  try {
    const { ok, data } = await api('/api/detect_model?path=' + encodeURIComponent(path));
    if (seq !== modelDetectSeq) return;           // inzwischen wurde etwas anderes gewählt
    if (!ok) { setDetectState('none'); updateShrink(); return; }
    MODEL_INFO = data;
    if (data.preset_key && EXCLUDES[data.preset_key]) {
      setDetectState('found', data.preset_key);
      S.exclude = data.preset_key; renderChips(); updateExcludeHint();
    } else setDetectState('unknown');
    updateShrink(); onSettingsChanged();
  } catch (e) { if (seq === modelDetectSeq) { setDetectState('none'); updateShrink(); } }
}

function selectModel(m) {
  selectedModel = m; START_ERROR = '';
  $('modelSearch').value = modelDisplayName(m);
  $('inputPath').value = m.path;
  closeComboPanel(); updateModelPreview(); onSettingsChanged();
  detectModel(m.path);
}

// Pfad von Hand: Datei erst nach kurzer Pause prüfen, nicht bei jedem Tastendruck.
function onManualPathInput() {
  selectedModel = null; START_ERROR = '';
  if ($('modelSearch').value) $('modelSearch').value = '';
  clearTimeout(detectTimer);
  const p = inputPathValue();
  modelDetectSeq++;
  if (!p || !/\.safetensors$/i.test(p)) { MODEL_INFO = null; setDetectState('none'); updateShrink(); }
  else detectTimer = setTimeout(() => detectModel(p), 450);
  updateModelPreview(); onSettingsChanged();
}

/* =====================================================================
   Voreinstellungen
   ===================================================================== */
function getSettingsSnapshot() {
  return {
    format: S.format, exclude_preset: S.exclude, low_memory_mode: $('lowMemoryMode').value,
    convrot_groupsize: Number($('convrotGroupsizeVal').value) || 256, block_size: Number($('blockSize').value) || 128,
    custom_regex: $('customRegex').value, extra_args: $('extraArgs').value,
  };
}
function applySettingsSnapshot(s) {
  if (FORMATS[s.format]) S.format = s.format;
  if (EXCLUDES[s.exclude_preset]) S.exclude = s.exclude_preset;
  $('lowMemoryMode').value = s.low_memory_mode || 'auto';
  $('convrotGroupsizeVal').value = String(s.convrot_groupsize || 256);
  $('blockSize').value = s.block_size || 128;
  $('customRegex').value = s.custom_regex || '';
  $('extraArgs').value = s.extra_args || '';
  START_ERROR = '';
  renderFormatCards(); renderFormatDetail(); renderGroupsizeSeg(); renderChips(); updateExcludeHint(); onSettingsChanged();
}
const sameSnapshot = (a, b) => ['format', 'exclude_preset', 'low_memory_mode', 'convrot_groupsize', 'block_size', 'custom_regex', 'extra_args'].every((k) => String(a[k] ?? '') === String(b[k] ?? ''));

function presetMsg(text, ok) {
  const el = $('presetMsg');
  if (!text) { el.hidden = true; return; }
  el.textContent = text; el.className = 'inline-msg' + (ok ? ' ok' : ''); el.hidden = false;
}
async function loadPresets() {
  try { const { data } = await api('/api/presets'); PRESETS = data.presets || []; } catch (e) { PRESETS = []; }
  renderPresets();
}
function renderPresets() {
  const box = $('presetChips'), now = getSettingsSnapshot();
  if (!PRESETS.length) { box.innerHTML = `<span class="hint">${escapeHtml(t('presetNone'))}</span>`; return; }
  box.innerHTML = PRESETS.map((p, i) => {
    const active = sameSnapshot(now, p.settings);
    return `<span class="preset-chip${active ? ' active' : ''}"><button type="button" class="chip" data-preset="${i}" aria-pressed="${active}">${escapeHtml(p.name)}</button><button type="button" class="del" data-preset-del="${i}" title="${escapeHtml(t('presetDeleteTitle'))}" aria-label="${escapeHtml(t('presetDeleteTitle') + ': ' + p.name)}">&times;</button></span>`;
  }).join('');
  box.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
    const p = PRESETS[+b.dataset.preset]; applySettingsSnapshot(p.settings); presetMsg(t('presetLoaded', p.name), true);
  }));
  box.querySelectorAll('[data-preset-del]').forEach((b) => b.addEventListener('click', async () => {
    const p = PRESETS[+b.dataset.presetDel];
    if (!confirm(t('presetDeleteConfirm', p.name))) return;
    const { ok, data } = await api('/api/presets/delete', { name: p.name });
    if (!ok) presetMsg(data.error || t('presetError')); else { presetMsg(''); await loadPresets(); }
  }));
}
function toggleSaveRow(show) {
  $('presetSaveRow').hidden = !show; $('presetSaveToggle').hidden = show;
  if (show) { $('presetName').value = ''; $('presetName').focus(); }
}
async function savePreset() {
  const name = $('presetName').value.trim();
  if (!name) { presetMsg(t('presetNameMissing')); return; }
  const { ok, data } = await api('/api/presets', { name, settings: getSettingsSnapshot() });
  if (!ok) { presetMsg(data.error || t('presetError')); return; }
  toggleSaveRow(false); presetMsg(t('presetSaved', name), true); await loadPresets();
}

/* =====================================================================
   Befehlsvorschau
   ===================================================================== */
let SHELL = 'cmd', previewTimer = null, previewSeq = 0, LAST_CMD_TEXT = '';

function shellQuote(x) {
  if (x === '') return SHELL === 'ps' ? "''" : '""';
  if (!/[\s"'&|<>()^%!;$`,]/.test(x)) return x;
  return SHELL === 'ps' ? "'" + x.replace(/'/g, "''") + "'" : '"' + x.replace(/"/g, '\\"') + '"';
}
function requestBody() {
  const s = getSettingsSnapshot();
  return {
    input_path: inputPathValue(), output_path: outputPathValue(), format: s.format, exclude_preset: s.exclude_preset,
    custom_regex: s.custom_regex, convrot_groupsize: s.convrot_groupsize, block_size: s.block_size,
    low_memory_mode: s.low_memory_mode, extra_args: s.extra_args,
  };
}
function schedulePreview() {
  if (!$('previewDetails').open) return;
  clearTimeout(previewTimer); previewTimer = setTimeout(refreshPreview, 250);
}
async function refreshPreview() {
  const seq = ++previewSeq;
  try {
    const { ok, data } = await api('/api/preview', requestBody());
    if (seq !== previewSeq) return;
    if (!ok) { $('previewCmd').textContent = ''; $('previewNote').textContent = t('previewError') + (data.error || ''); LAST_CMD_TEXT = ''; return; }
    LAST_CMD_TEXT = [data.program].concat(data.args.map(shellQuote)).join(' ');
    $('previewCmd').textContent = LAST_CMD_TEXT;
    const notes = [];
    if (data.input_missing) notes.push(t('previewNeedModel'));
    if (data.low_memory_pending) notes.push(t('previewLowAuto'));
    $('previewNote').textContent = notes.join(' ');
  } catch (e) { if (seq === previewSeq) $('previewNote').textContent = t('previewError') + e.message; }
}
async function copyCommand() {
  if (!LAST_CMD_TEXT) await refreshPreview();
  try { await navigator.clipboard.writeText(LAST_CMD_TEXT); toast(t('previewCopied')); }
  catch (e) { toast(t('copyFailed')); }
}
function setShell(sh) {
  SHELL = sh; $('shellCmd').setAttribute('aria-pressed', sh === 'cmd'); $('shellPs').setAttribute('aria-pressed', sh === 'ps'); refreshPreview();
}

/* =====================================================================
   Logs, ctq-Hilfe, Statistik
   ===================================================================== */
async function loadLogsList() {
  const box = $('logsList');
  try {
    const { data } = await api('/api/logs');
    const logs = data.logs || []; LOGS_COUNT = logs.length; box.innerHTML = '';
    if (!logs.length) box.innerHTML = `<p class="hint">${escapeHtml(t('logsNone'))}</p>`;
    else {
      const locale = LANG === 'en' ? 'en-US' : 'de-DE';
      logs.forEach((entry) => {
        const row = document.createElement('div'); row.className = 'log-row';
        const info = document.createElement('span');
        info.innerHTML = '<span class="when">' + escapeHtml(new Date(entry.mtime * 1000).toLocaleString(locale)) + '</span> &nbsp; ' + escapeHtml(entry.name) + ' <span class="when">(' + Math.max(1, Math.round(entry.size / 1024)) + ' KB)</span>';
        const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'secondary'; btn.textContent = t('logsView');
        btn.addEventListener('click', () => viewLog(entry.name));
        row.appendChild(info); row.appendChild(btn); box.appendChild(row);
      });
    }
  } catch (e) { box.innerHTML = `<p class="hint">${escapeHtml(t('logsError'))}</p>`; }
  updateStatsBar();
}
async function viewLog(name) {
  const viewer = $('logViewer'); viewer.hidden = false; viewer.textContent = t('logsLoading', name);
  try { const res = await fetch('/api/logs/' + encodeURIComponent(name)); viewer.textContent = res.ok ? await res.text() : t('logsLoadError'); }
  catch (e) { viewer.textContent = t('logsLoadError'); }
  viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function updateStatsBar() { $('statsBar').textContent = t('statsBarTemplate', MODEL_ROOTS_COUNT, MODELS.length, LOGS_COUNT); }
async function showCtqHelp() {
  const btn = $('ctqHelpBtn'), out = $('ctqHelpOutput');
  if (!out.hidden) { out.hidden = true; btn.textContent = t('ctqHelpShow'); return; }
  btn.textContent = t('ctqHelpLoading');
  const { data } = await api('/api/ctq_help');
  out.textContent = data.help || t('errorStatusTemplate', data.error || ''); out.hidden = false; btn.textContent = t('ctqHelpHide');
}

async function loadConfig() {
  const { data: cfg } = await api('/api/config');
  FORMATS = cfg.formats || {}; EXCLUDES = cfg.excludes || {};
  CTQ_FOUND = !!cfg.ctq_found; CTQ_PATH = cfg.ctq_path;
  if (!FORMATS[S.format]) S.format = Object.keys(FORMATS)[0];
  if (!EXCLUDES[S.exclude]) S.exclude = 'none';
  renderCtqStatus(); renderFormatCards(); renderFormatDetail(); renderGroupsizeSeg(); renderChips(); updateExcludeHint(); updateShrink(); updateActionBar();
}
/* =====================================================================
   Warteschlange, Auftragsfenster, Start
   ===================================================================== */
const ACTIVE_STATES = ['queued', 'starting', 'running'];
const isFinal = (s) => !ACTIVE_STATES.includes(s);

let QUEUE = [], RUNNING = null, QUEUE_ACTIVE = false, STARTING = false;
let FOCUS = null, FOCUS_PINNED = false, focusGeneration = 0, FOCUS_STATUS_CLASS = '';
let timerInterval = null;

function renderQueue() {
  const sec = $('queueSection'), list = $('queueList');
  const active = QUEUE.filter((j) => !isFinal(j.status));
  const finished = QUEUE.filter((j) => isFinal(j.status)).slice(0, 6);
  const shown = active.concat(finished);
  sec.hidden = shown.length === 0;
  if (!shown.length) { list.innerHTML = ''; return; }
  list.innerHTML = shown.map((j) => {
    let label, cls;
    if (j.status === 'queued') { label = t('qWaiting', j.position || '?'); cls = 'waiting'; }
    else if (j.status === 'starting') { label = t('qStarting'); cls = 'running'; }
    else if (j.status === 'running') { label = t('qRunning', j.progress || 0); cls = 'running'; }
    else if (j.status === 'done') { label = t('qDone'); cls = 'done'; }
    else if (j.status === 'cancelled') { label = t('qCancelled'); cls = 'cancelled'; }
    else { label = t('qError'); cls = 'error'; }
    const sub = j.status === 'error' ? (j.error || '') : (j.output_path || '');
    const buttons = [];
    if (j.log_file || j.status === 'running' || j.status === 'starting') buttons.push(`<button type="button" class="queue-btn" data-q-log="${escapeHtml(j.id)}">${t('qLog')}</button>`);
    if (!isFinal(j.status)) buttons.push(`<button type="button" class="queue-btn" data-q-cancel="${escapeHtml(j.id)}" data-q-state="${escapeHtml(j.status)}">${j.status === 'queued' ? t('qRemove') : t('qCancel')}</button>`);
    const bar = j.status === 'running' ? `<div class="q-bar" aria-hidden="true"><i style="width:${Number(j.progress) || 0}%"></i></div>` : '';
    return `<li class="queue-item ${j.status === 'running' || j.status === 'starting' ? 'running' : ''}">
      <div class="q-main"><div class="q-name">${escapeHtml(j.name || '')} <small>${escapeHtml(fmtName(j.format))}</small></div><div class="q-sub">${escapeHtml(sub)}</div></div>
      <div class="q-side"><span class="q-status ${cls}">${escapeHtml(label)}</span><div class="q-actions">${buttons.join('')}</div></div>${bar}</li>`;
  }).join('');
  list.querySelectorAll('[data-q-log]').forEach((b) => b.addEventListener('click', () => {
    const job = QUEUE.find((j) => j.id === b.dataset.qLog);
    setFocus(b.dataset.qLog, !!job && isFinal(job.status), true);
  }));
  list.querySelectorAll('[data-q-cancel]').forEach((b) => b.addEventListener('click', () => cancelJob(b.dataset.qCancel, b.dataset.qState)));
}

async function refreshQueue() {
  if (document.hidden) return;
  try {
    const { ok, data } = await api('/api/queue');
    if (!ok) return;
    QUEUE = data.jobs || [];
  } catch (e) { return; }
  RUNNING = QUEUE.find((j) => j.status === 'running' || j.status === 'starting') || null;
  QUEUE_ACTIVE = QUEUE.some((j) => !isFinal(j.status));
  renderQueue(); updateActionBar();
  const focusJob = FOCUS && QUEUE.find((j) => j.id === FOCUS);
  if (RUNNING && RUNNING.id !== FOCUS && (!FOCUS || (!FOCUS_PINNED && (!focusJob || isFinal(focusJob.status))))) setFocus(RUNNING.id, false, false);
}

function updateTimer() {
  const el = $('drawerTimer');
  if (!RUNNING) { el.textContent = ''; return; }
  const secs = RUNNING.started_at ? Math.max(0, Math.floor(Date.now() / 1000 - RUNNING.started_at)) : 0;
  el.textContent = t('durationLabel') + Math.floor(secs / 60) + ':' + String(secs % 60).padStart(2, '0');
}

function updateActionBar() {
  const summary = $('actionSummary'), path = inputPathValue();
  if (START_ERROR) summary.innerHTML = '<span class="errline">' + escapeHtml(START_ERROR) + '</span>';
  else if (!path) summary.textContent = t('chooseModelFirst');
  else if (!CTQ_FOUND) summary.innerHTML = '<span class="warnline">' + escapeHtml(t('ctqMissingAction')) + '</span>';
  else summary.innerHTML = '<b>' + escapeHtml(path.split(/[\\/]/).pop()) + '</b> &rarr; ' + escapeHtml(fmtName(S.format));
  const run = $('runBtn');
  run.textContent = QUEUE_ACTIVE ? t('runBtnQueue') : t('runBtn');
  run.disabled = !path || !CTQ_FOUND || STARTING;
  $('actionProgress').hidden = !RUNNING;
  $('progressBar').style.width = (RUNNING ? (RUNNING.progress || 0) : 0) + '%';
  $('cancelJobBarBtn').hidden = !RUNNING;
  updateTimer();
}

/* ---- Auftragsfenster ---- */
const drawerIsOpen = () => $('jobDrawer').classList.contains('open');
function openDrawer() { $('jobDrawer').classList.add('open'); $('drawerReopenBtn').style.display = 'none'; $('drawerReopenBtn').hidden = true; }
function minimizeDrawer() {
  $('jobDrawer').classList.remove('open');
  if (FOCUS_PINNED) FOCUS_PINNED = false;
  if (FOCUS) { $('drawerReopenBtn').hidden = false; $('drawerReopenBtn').style.display = 'inline-flex'; }
}
function setStatusLine(text, cls) {
  const el = $('drawerStatusLine');
  el.textContent = text; el.className = 'job-drawer-status ' + (cls || '');
  $('jobDrawer').classList.toggle('running', cls === 'status-running');
  $('drawerReopenText').textContent = text;
  $('drawerReopenBtn').className = 'drawer-reopen-pill ' + (cls || '');
  if (!drawerIsOpen() && FOCUS) { $('drawerReopenBtn').hidden = false; $('drawerReopenBtn').style.display = 'inline-flex'; }
}

function setFocus(id, pinned, openIt) {
  FOCUS = id; FOCUS_PINNED = !!pinned;
  $('log').textContent = ''; $('jobLogLink').hidden = true;
  setStatusLine(t('startingStatus'), '');
  if (openIt) openDrawer();
  pollFocus(id);
}

function pollFocus(jobId) {
  // Eigene Generation statt setInterval: der nächste Abruf wird erst nach Abschluss des vorigen geplant,
  // damit sich überlappende Antworten nicht gegenseitig mit veralteten Daten überschreiben.
  const mine = ++focusGeneration;
  async function tick() {
    if (mine !== focusGeneration) return;
    let res;
    try { res = await api('/api/status/' + jobId); } catch (e) { if (mine === focusGeneration) setTimeout(tick, 1500); return; }
    if (mine !== focusGeneration) return;
    if (res.status === 404) { setStatusLine(t('errorStatusTemplate', res.data.error || ''), 'status-error'); return; }
    const d = res.data;
    $('drawerName').textContent = (d.name || '') + (d.format ? ' · ' + fmtName(d.format) : '');
    const logEl = $('log'), stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 30;
    logEl.textContent = (d.log || []).join('\n'); if (stick) logEl.scrollTop = logEl.scrollHeight;
    const link = $('jobLogLink');
    if (d.log_file) {
      link.hidden = false; link.innerHTML = escapeHtml(t('fullLogSavingAs')) + '<a href="#" id="jobLogLinkA">' + escapeHtml(d.log_file) + '</a>';
      $('jobLogLinkA').addEventListener('click', (e) => { e.preventDefault(); viewLog(d.log_file); });
    }
    const active = !isFinal(d.status);
    $('cancelJobDrawerBtn').hidden = !active;
    if (d.status === 'queued') {
      const q = QUEUE.find((j) => j.id === jobId);
      setStatusLine(t('waitingStatusTemplate', q && q.position ? q.position : '?'), 'status-running'); setTimeout(tick, 1500);
    } else if (d.status === 'running' || d.status === 'starting') {
      setStatusLine(t('runningStatusTemplate', d.progress || 0), 'status-running'); setTimeout(tick, 1200);
    } else {
      const map = { done: ['doneStatusTemplate', d.output_path, 'status-done'], error: ['errorStatusTemplate', d.error, 'status-error'], cancelled: ['cancelledStatus', null, 'status-cancelled'] };
      const m = map[d.status];
      if (m) setStatusLine(m[1] === null ? t(m[0]) : t(m[0], m[1]), m[2]);
      loadLogsList(); refreshQueue();
    }
  }
  tick();
}

async function cancelJob(jobId, state) {
  if (!jobId) return;
  if (state !== 'queued' && !confirm(t('cancelConfirm'))) return;
  const btns = document.querySelectorAll('.cancel-job-btn');
  btns.forEach((b) => { b.disabled = true; b.textContent = t('cancellingBtn'); });
  try {
    const { ok, data } = await api('/api/cancel/' + jobId, {});
    if (!ok) alert(data.error || t('cancelFailed'));
  } catch (e) { alert(t('cancelFailed')); }
  btns.forEach((b) => { b.disabled = false; b.textContent = t('cancelJobBtn'); });
  refreshQueue();
}

async function startJob() {
  if (STARTING) return;
  STARTING = true; updateActionBar();
  try {
    const { ok, data } = await api('/api/quantize', requestBody());
    if (!ok) { START_ERROR = data.error || t('startFailed'); toast(t('startFailed') + START_ERROR, 6000); return; }
    START_ERROR = '';
    await refreshQueue();
    if (!data.queued) setFocus(data.job_id, false, true);
    else {
      const q = QUEUE.find((j) => j.id === data.job_id);
      toast(t('queuedToast', q && q.position ? q.position : data.position - 1));
    }
  } catch (e) { START_ERROR = String((e && e.message) || e); toast(t('startFailed') + START_ERROR, 6000); }
  finally { STARTING = false; updateActionBar(); }
}

// Fenster verschieben und in der Größe ändern.
(function setupJobDrawerDragAndResize() {
  const drawer = $('jobDrawer'), header = $('jobDrawerHeader'), handle = $('jobDrawerResizeHandle');
  const clamp = (v, min, max) => Math.min(Math.max(v, min), max);
  function pin() {
    const rect = drawer.getBoundingClientRect();
    drawer.style.left = rect.left + 'px'; drawer.style.top = rect.top + 'px'; drawer.classList.add('placed'); return rect;
  }
  let dragging = false, sx = 0, sy = 0, sl = 0, st = 0, resizing = false, rw = 0, rh = 0;
  header.addEventListener('mousedown', (e) => {
    if (e.target.closest('button')) return;
    dragging = true; const r = pin(); document.body.classList.add('drawer-noselect'); sx = e.clientX; sy = e.clientY; sl = r.left; st = r.top; e.preventDefault();
  });
  handle.addEventListener('mousedown', (e) => {
    resizing = true; const r = pin(); sx = e.clientX; sy = e.clientY; rw = r.width; rh = r.height; document.body.classList.add('drawer-noselect'); e.preventDefault(); e.stopPropagation();
  });
  window.addEventListener('mousemove', (e) => {
    if (dragging) {
      drawer.style.left = clamp(sl + e.clientX - sx, 0, Math.max(window.innerWidth - drawer.offsetWidth, 0)) + 'px';
      drawer.style.top = clamp(st + e.clientY - sy, 0, Math.max(window.innerHeight - drawer.offsetHeight, 0)) + 'px';
    } else if (resizing) {
      const rect = drawer.getBoundingClientRect();
      drawer.style.maxWidth = 'none'; drawer.style.maxHeight = 'none';
      drawer.style.width = clamp(rw + e.clientX - sx, 300, window.innerWidth - rect.left - 12) + 'px';
      drawer.style.height = clamp(rh + e.clientY - sy, 180, window.innerHeight - rect.top - 12) + 'px';
    }
  });
  window.addEventListener('mouseup', () => { if (dragging || resizing) { dragging = resizing = false; document.body.classList.remove('drawer-noselect'); } });
})();

/* =====================================================================
   Gemeinsame Aktualisierung, Sprache, Start
   ===================================================================== */
function onSettingsChanged() {
  updateModelPreview(); updateShrink(); renderPresets(); updateActionBar(); schedulePreview();
}

function applyLanguage() {
  applyStaticI18n(); renderCtqStatus(); updateMinimalButton();
  renderFormatCards(); renderFormatDetail(); renderGlossary(); renderChips(); updateExcludeHint();
  showManualPath(!$('manualPathRow').hidden); showOutputRow(!$('outputPathRow').hidden);
  $('modelSelectHint').textContent = MODELS.length ? t('modelsFoundTemplate', MODELS.length, MODEL_ROOTS_COUNT) : t('modelsNoneFound');
  if (DETECT_STATE === 'found') setDetectState('found', DETECTED_KEY);
  updateModelPreview(); updateShrink(); renderPresets(); renderQueue(); updateActionBar(); updateStatsBar(); loadLogsList(); refreshPreview();
  if (LAST_SYSTEM_CHECKS) renderSystemCheck(LAST_SYSTEM_CHECKS);
  renderUpdateNotice();
  if (settingsIsOpen()) { renderRootList(); updateCtqAutoHint(); }
}
function setLanguage(lang) { LANG = lang; writeStored('lang', LANG); applyLanguage(); }

function bindEvents() {
  $('runBtn').addEventListener('click', startJob);
  $('cancelJobBarBtn').addEventListener('click', () => RUNNING && cancelJob(RUNNING.id, RUNNING.status));
  $('cancelJobDrawerBtn').addEventListener('click', () => {
    const job = QUEUE.find((j) => j.id === FOCUS); cancelJob(FOCUS, job ? job.status : 'running');
  });
  $('drawerCloseBtn').addEventListener('click', minimizeDrawer);
  $('drawerReopenBtn').addEventListener('click', openDrawer);
  $('ctqHelpBtn').addEventListener('click', showCtqHelp);
  $('refreshModelsBtn').addEventListener('click', loadModels);
  $('manualPathToggle').addEventListener('click', () => showManualPath($('manualPathRow').hidden));
  $('outputToggle').addEventListener('click', () => showOutputRow($('outputPathRow').hidden));
  $('inputPath').addEventListener('input', onManualPathInput);
  $('outputPath').addEventListener('input', () => { START_ERROR = ''; updateModelPreview(); onSettingsChanged(); });
  $('recommendBtn').addEventListener('click', applyRecommended);

  const search = $('modelSearch');
  search.addEventListener('focus', () => { search.select(); openComboPanel(true); });
  search.addEventListener('input', () => openComboPanel(false));
  search.addEventListener('keydown', (e) => {
    const open = !$('modelComboPanel').hidden;
    if (e.key === 'ArrowDown') { e.preventDefault(); if (!open) { openComboPanel(true); return; } if (comboActiveIndex < comboVisibleItems.length - 1) comboActiveIndex++; highlightActiveItem(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); if (comboActiveIndex > 0) comboActiveIndex--; highlightActiveItem(); }
    else if (e.key === 'Enter') { e.preventDefault(); if (comboActiveIndex >= 0 && comboVisibleItems[comboActiveIndex]) selectModel(comboVisibleItems[comboActiveIndex]); }
    else if (e.key === 'Escape') { closeComboPanel(); search.blur(); }
  });
  search.addEventListener('blur', () => setTimeout(() => { closeComboPanel(); if (selectedModel) search.value = modelDisplayName(selectedModel); }, 150));

  ['blockSize', 'customRegex', 'extraArgs'].forEach((id) => $(id).addEventListener('input', () => { START_ERROR = ''; onSettingsChanged(); }));
  $('lowMemoryMode').addEventListener('change', onSettingsChanged);

  $('presetSaveToggle').addEventListener('click', () => toggleSaveRow(true));
  $('presetSaveCancel').addEventListener('click', () => toggleSaveRow(false));
  $('presetSaveConfirm').addEventListener('click', savePreset);
  $('presetName').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); savePreset(); } else if (e.key === 'Escape') toggleSaveRow(false); });

  $('previewDetails').addEventListener('toggle', () => { if ($('previewDetails').open) refreshPreview(); });
  $('copyCmdBtn').addEventListener('click', copyCommand);
  $('shellCmd').addEventListener('click', () => setShell('cmd'));
  $('shellPs').addEventListener('click', () => setShell('ps'));

  $('themeToggleBtn').addEventListener('click', toggleTheme);
  $('minimalToggleBtn').addEventListener('click', toggleMinimal);
  $('langSelect').value = LANG;
  $('langSelect').addEventListener('change', (e) => setLanguage(e.target.value));

  $('settingsBtn').addEventListener('click', () => { if (settingsIsOpen()) closeSettings(); else openSettings(false); });
  $('ctqStatus').addEventListener('click', () => { if (!CTQ_FOUND) openSettings(false); });
  $('settingsCloseBtn').addEventListener('click', closeSettings);
  $('settingsCancelBtn').addEventListener('click', closeSettings);
  $('settingsSaveBtn').addEventListener('click', saveSettings);
  document.addEventListener('mousedown', (e) => { if (settingsIsOpen() && !e.target.closest('.settings-anchor')) closeSettings(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && settingsIsOpen()) closeSettings(); });
  $('addRootBtn').addEventListener('click', async () => { const p = await pickNativeFolder(t('settingsAddRoot'), ''); if (p) addModelRootFromPath(p); });
  $('browseComfyuiBtn').addEventListener('click', async () => { const p = await pickNativeFolder(t('settingsComfyuiHeading'), $('comfyuiRootInput').value); if (p) $('comfyuiRootInput').value = p; });
  $('browseCtqBtn').addEventListener('click', async () => { const p = await pickNativeFile(t('settingsCtqHeading'), 'ctq executable|ctq.exe|All files|*.*', $('ctqOverrideInput').value); if (p) $('ctqOverrideInput').value = p; });
  $('runSystemCheckBtn').addEventListener('click', () => runSystemCheck(false));
  $('autoUpdateCheck').checked = readStored('autoUpdateCheck', '1') === '1';
  $('autoUpdateCheck').addEventListener('change', (e) => writeStored('autoUpdateCheck', e.target.checked ? '1' : '0'));
  $('updatePill').addEventListener('click', showUpdates);
  $('updateShowBtn').addEventListener('click', showUpdates);
  $('updateDismissBtn').addEventListener('click', () => { writeStored('updateDismissed', updateSignature(availableUpdates())); $('updateBanner').hidden = true; });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshQueue(); });
}

async function init() {
  applyTheme(); applyMinimal(); applyStaticI18n();
  showManualPath(false); showOutputRow(false); renderGlossary(); bindEvents();
  updateModelPreview(); updateShrink(); updateActionBar(); updateExcludeBanner();
  try { await loadConfig(); } catch (e) { $('modelSelectHint').textContent = t('modelsScanError'); }
  loadModels(); loadLogsList(); loadPresets(); refreshQueue();
  setInterval(refreshQueue, 2000);
  setInterval(updateTimer, 500);
  try {
    const { data } = await api('/api/settings');
    if (data.first_run && !readStored('setupSeen', '')) { writeStored('setupSeen', '1'); openSettings(true); }
  } catch (e) {}
  if (readStored('autoUpdateCheck', '1') === '1') runSystemCheck(true);
}
init();
