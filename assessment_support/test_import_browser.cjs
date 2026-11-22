// Exercise the actual import and persistence functions with browser localStorage.
// The profile and synthetic drafts exist only in a temporary directory.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');

const browser = process.env.EAGE_TEST_BROWSER || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const source = fs.readFileSync('EAGE_pythonSandBox/app.js', 'utf8');
const workspaceStart = source.indexOf('function persist()');
const workspaceEnd = source.indexOf('function renderNav()', workspaceStart);
const importStart = source.indexOf("$('import-work').onchange=");
const importEnd = source.indexOf("$('upload').onchange=", importStart);
assert.ok(workspaceStart >= 0 && workspaceEnd > workspaceStart && importStart >= 0 && importEnd > importStart);
const workspace = source.slice(workspaceStart, workspaceEnd);
const handler = source.slice(importStart, importEnd);
const html = `<!doctype html><meta charset="utf-8"><title>pending</title>
<input id="import-work" type="file"><textarea id="code">current code</textarea><textarea id="notes">current note</textarea>
<input id="stdin"><div id="workspace-title">Workspace</div><div id="saved"></div><select id="workspace-select"></select><div id="workspace-note"></div>
<script>
const $ = id => document.getElementById(id);
let saved = {'lesson::draft':{code:'old',notes:'old'},'other::draft':{code:'other old',notes:'old'}};
const course={source_digest:'current-version'}, STORAGE='eage-test', lesson={id:'lesson',workspaces:[],codes:[],assessment:false,project:false};
let active='draft', error='';
const key=()=> 'lesson::draft';
const appendText=text=>error=text;
window.confirm=()=>true;
${workspace}
${handler}
(async()=>{
  const before=JSON.stringify(saved); localStorage.setItem(STORAGE,before);
  const original=Storage.prototype.setItem;
  Storage.prototype.setItem=function(){throw new DOMException('quota exceeded','QuotaExceededError');};
  const imported={format:STORAGE,course:'older-version',drafts:{'other::draft':{code:'new',notes:'note'},'fresh::draft':{code:'new',notes:'note'}}};
  await $('import-work').onchange({target:{files:[{text:async()=>JSON.stringify(imported)}],value:'selected'}});
  const failed=JSON.stringify(saved)===before && localStorage.getItem(STORAGE)===before && $('code').value==='current code' && error.includes('quota exceeded');
  Storage.prototype.setItem=original;
  persist();
  const stored=JSON.parse(localStorage.getItem(STORAGE));
  const recovered=stored['other::draft'].code==='other old' && stored['lesson::draft'].code==='current code' && !('fresh::draft' in stored);
  document.title=failed&&recovered?'PASS':'FAIL';
  document.body.dataset.result=JSON.stringify({failed,recovered,error});
})().catch(e=>{document.title='ERROR';document.body.dataset.result=e.stack;});
</script>`;
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'eage-import-browser-'));
try {
  const page = path.join(temp, 'test.html');
  fs.writeFileSync(page, html);
  const run = spawnSync(browser, ['--headless=new', '--disable-gpu', '--no-first-run', '--disable-background-networking',
    '--user-data-dir=' + path.join(temp, 'profile'), '--virtual-time-budget=3000', '--dump-dom', pathToFileURL(page).href],
    {encoding:'utf8', timeout:30000, windowsHide:true, maxBuffer:4*1024*1024});
  if (run.error) throw run.error;
  assert.equal(run.status, 0, run.stderr);
  assert.match(run.stdout, /<title>PASS<\/title>/, run.stderr + '\n' + run.stdout);
  console.log('Browser localStorage: failed import and subsequent save preserved the original drafts.');
} finally {
  fs.rmSync(temp, {recursive:true, force:true});
}
