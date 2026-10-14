// Tests sortModels() of static/app.js (model list sorting by name or date).
//   node tests/model_sort.test.js
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'static', 'app.js'), 'utf8');
const block = src.slice(src.indexOf('const modelDisplayName'), src.indexOf('const modelMatches'));
const sortModels = new Function('readStored', block + '; return sortModels;')((key, fallback) => fallback);

const models = [
  { filename: 'flux_b.safetensors', subdir: '', root_label: 'Diffusion', mtime: 300 },
  { filename: 'anima.safetensors', subdir: 'old', root_label: 'Diffusion', mtime: 100 },
  { filename: 'Qwen_10.safetensors', subdir: '', root_label: 'Other', mtime: 200 },
  { filename: 'Qwen_2.safetensors', subdir: '', root_label: 'Other', mtime: 200 },   // same date as Qwen_10
  { filename: 'no_date.safetensors', subdir: '', root_label: 'Other' },              // no mtime: counts as oldest
];
const names = (list) => list.map((m) => m.filename.replace('.safetensors', '')).join(', ');

const expected = {
  // by displayed name (subfolder included, so "old/anima" sorts under "o"), case-insensitive, numbers in natural order
  'name-asc': 'flux_b, no_date, anima, Qwen_2, Qwen_10',
  'name-desc': 'Qwen_10, Qwen_2, anima, no_date, flux_b',
  // oldest first / newest first; equal dates always fall back to the name A-Z
  'date-asc': 'no_date, anima, Qwen_2, Qwen_10, flux_b',
  'date-desc': 'flux_b, Qwen_2, Qwen_10, anima, no_date',
};

let failed = 0;
for (const [mode, wanted] of Object.entries(expected)) {
  const actual = names(sortModels(models, mode));
  if (actual !== wanted) { failed++; console.log(`FAIL ${mode}: expected "${wanted}", got "${actual}"`); }
}
// the original list must not be reordered (the list is shared with the rest of the UI)
if (names(models) !== 'flux_b, anima, Qwen_10, Qwen_2, no_date') { failed++; console.log('FAIL: sortModels changed its input'); }
console.log(failed ? `${failed} FAILED` : `all ${Object.keys(expected).length + 1} model sorting checks passed`);
process.exit(failed ? 1 : 0);
