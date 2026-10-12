// Tests the shellQuote() function of static/app.js for cmd, PowerShell and bash.
//   node tests/shell_quote.test.js
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'static', 'app.js'), 'utf8');
const fn = src.slice(src.indexOf('function shellQuote'), src.indexOf('function requestBody'));
const quoter = (shell) => new Function('SHELL', fn + '; return shellQuote;')(shell);

const expectations = [
  // [shell, input, expected]
  ['cmd', 'C:\\models\\a.safetensors', 'C:\\models\\a.safetensors'],   // backslashes are plain path characters
  ['ps', 'C:\\models\\a.safetensors', 'C:\\models\\a.safetensors'],
  ['sh', 'C:\\models\\a.safetensors', "'C:\\models\\a.safetensors'"],  // bash would eat them
  ['sh', '/home/me/a.safetensors', '/home/me/a.safetensors'],
  ['sh', '/home/me/my models/a.safetensors', "'/home/me/my models/a.safetensors'"],
  ['sh', '(img_in|txt_in)', "'(img_in|txt_in)'"],
  ['sh', "it's", "'it'\\''s'"],
  ['sh', 'input_blocks\\.0\\.0', "'input_blocks\\.0\\.0'"],
  ['sh', '', "''"],
  ['cmd', '', '""'],
  ['ps', "it's", "'it''s'"],
  ['cmd', '/home/me/my models/a', '"/home/me/my models/a"'],
  ['sh', '--int8', '--int8'],
];

let failed = 0;
for (const [shell, input, expected] of expectations) {
  const actual = quoter(shell)(input);
  if (actual !== expected) {
    failed++;
    console.log(`FAIL [${shell}] ${JSON.stringify(input)}: expected ${expected}, got ${actual}`);
  }
}
console.log(failed ? `${failed} FAILED` : `all ${expectations.length} shell quoting checks passed`);
process.exit(failed ? 1 : 0);
