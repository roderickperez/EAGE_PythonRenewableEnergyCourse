const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('EAGE_pythonSandBox/app.js','utf8');
const start = source.indexOf("$('import-work').onchange=");
const end = source.indexOf("$('upload').onchange=",start);
assert.ok(start >= 0 && end > start);
const elements = {'import-work':{}, code:{value:'current code'}, notes:{value:'current note'}, stdin:{value:''},
  'workspace-title':{textContent:'Workspace'}, saved:{classList:{add(){},remove(){}}},
  'workspace-select':{replaceChildren(){}}, 'workspace-note':{}};
const initial = {'lesson::draft':{code:'old',notes:'old'}, 'other::draft':{code:'other old',notes:'old'}};
let message, allow = false, fail = false, writes = [], errors=[];
const context = { $: id=>elements[id], saved:structuredClone(initial), course:{source_digest:'current-version'},
  key:()=> 'lesson::draft', active:'draft', STORAGE:'format',
  lesson:{id:'lesson',workspaces:[],codes:[],assessment:false,project:false},
  document:{createElement:()=>({})},
  confirm: text => { message=text; return allow; }, appendText: text => errors.push(text),
  localStorage:{setItem:(_key,value)=>{if(fail)throw Error('quota exceeded');writes.push(JSON.parse(value));}} };
vm.createContext(context);
const workspaceStart = source.indexOf('function persist()');
const workspaceEnd = source.indexOf('function renderNav()', workspaceStart);
assert.ok(workspaceStart >= 0 && workspaceEnd > workspaceStart);
vm.runInContext(source.slice(workspaceStart,workspaceEnd),context);
vm.runInContext(source.slice(start,end),context);
function event(data){return {target:{files:[{text:async()=>JSON.stringify(data)}],value:'selected'}};}
(async()=>{
 const data={format:'format',course:'older-version',drafts:{'other::draft':{code:'new',notes:'note'},'fresh::draft':{code:'new',notes:'note'}}};
 await elements['import-work'].onchange(event(data));
 assert.match(message,/different course version/);assert.match(message,/1 existing draft/);assert.match(message,/other::draft/);assert.equal(writes.length,0);
 allow=true;fail=true;await elements['import-work'].onchange(event(data));
 assert.equal(context.saved['other::draft'].code,'other old');assert.match(errors.at(-1),/quota exceeded/);assert.equal(elements.code.value,'current code');
 // Reproduce the original second step: a later successful save must not
 // persist the rejected import into a different, inactive draft.
 fail=false;context.persist();
 assert.equal(writes.at(-1)['other::draft'].code,'other old');
 assert.equal(writes.at(-1)['lesson::draft'].code,'current code');
 assert.equal(writes.at(-1)['fresh::draft'],undefined);
 await elements['import-work'].onchange(event(data));
 assert.equal(context.saved['other::draft'].code,'new');assert.equal(writes.at(-1)['lesson::draft'].code,'current code');
 const activeData={format:'format',course:'current-version',drafts:{'lesson::draft':{code:'imported active',notes:'imported note',stdin:'sample'}}};
 await elements['import-work'].onchange(event(activeData));
 assert.equal(elements.code.value,'imported active');assert.equal(elements.notes.value,'imported note');assert.equal(elements.stdin.value,'sample');
 assert.equal(writes.at(-1)['lesson::draft'].code,'imported active');
 allow=false;await elements['import-work'].onchange(event({...data,course:undefined}));assert.match(message,/version is missing/);
 console.log('Import confirmation, cancellation, storage failure, and actual workspace switching passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
