const $ = id => document.getElementById(id);
const STORAGE = 'eage.sandbox.v2';
const LAYOUT_STORAGE = 'eage.sandbox.layout';

// Layout State Management
let layoutState = {
  sidebarCollapsed: false,
  workspaceCollapsed: false,
  workspaceZoomed: false
};
try {
  const storedLayout = JSON.parse(localStorage.getItem(LAYOUT_STORAGE) || '{}');
  if (typeof storedLayout.sidebarCollapsed === 'boolean') layoutState.sidebarCollapsed = storedLayout.sidebarCollapsed;
  if (typeof storedLayout.workspaceCollapsed === 'boolean') layoutState.workspaceCollapsed = storedLayout.workspaceCollapsed;
  if (typeof storedLayout.workspaceZoomed === 'boolean') layoutState.workspaceZoomed = storedLayout.workspaceZoomed;
} catch { /* storage fallback */ }

function saveLayout() {
  try {
    localStorage.setItem(LAYOUT_STORAGE, JSON.stringify(layoutState));
  } catch { /* storage fallback */ }
}

function applyLayout() {
  const layout = $('layout');
  if (!layout) return;

  if (layoutState.workspaceZoomed) {
    layoutState.workspaceCollapsed = false;
  }

  layout.classList.toggle('sidebar-collapsed', layoutState.sidebarCollapsed);
  layout.classList.toggle('workspace-collapsed', layoutState.workspaceCollapsed);
  layout.classList.toggle('workspace-zoomed', layoutState.workspaceZoomed);

  // Update Topbar buttons
  $('toggle-sidebar')?.classList.toggle('active', !layoutState.sidebarCollapsed);
  $('toggle-sidebar')?.setAttribute('aria-pressed', String(!layoutState.sidebarCollapsed));

  $('toggle-workspace')?.classList.toggle('active', !layoutState.workspaceCollapsed);
  $('toggle-workspace')?.setAttribute('aria-pressed', String(!layoutState.workspaceCollapsed));

  $('zoom-workspace')?.classList.toggle('active', layoutState.workspaceZoomed);
  $('zoom-workspace')?.setAttribute('aria-pressed', String(layoutState.workspaceZoomed));
  const zoomText = $('zoom-text');
  if (zoomText) zoomText.textContent = layoutState.workspaceZoomed ? 'Zoom Out' : 'Zoom In';

  // Update Lab action buttons
  const labZoomBtn = $('lab-zoom-btn');
  if (labZoomBtn) {
    labZoomBtn.textContent = layoutState.workspaceZoomed ? '⤡ Restore' : '⤢ Zoom';
    labZoomBtn.title = layoutState.workspaceZoomed ? 'Zoom out to side-by-side view' : 'Zoom in to full workspace';
  }

  // Update Reading focus button
  $('toggle-reading-focus')?.classList.toggle('active', layoutState.workspaceCollapsed);

  // Update edge restore tabs
  if ($('sidebar-restore-tab')) {
    $('sidebar-restore-tab').style.display = layoutState.sidebarCollapsed ? 'inline-flex' : 'none';
  }
  if ($('workspace-restore-tab')) {
    $('workspace-restore-tab').style.display = (layoutState.workspaceCollapsed && !layoutState.workspaceZoomed) ? 'inline-flex' : 'none';
  }
  if ($('reopen-sidebar-btn')) {
    $('reopen-sidebar-btn').style.display = layoutState.sidebarCollapsed ? 'inline-flex' : 'none';
  }

  saveLayout();
}

function toggleSidebar(force) {
  layoutState.sidebarCollapsed = (force !== undefined) ? !force : !layoutState.sidebarCollapsed;
  if (layoutState.workspaceZoomed && !layoutState.sidebarCollapsed) {
    layoutState.workspaceZoomed = false;
  }
  applyLayout();
}

function toggleWorkspace(force) {
  layoutState.workspaceCollapsed = (force !== undefined) ? !force : !layoutState.workspaceCollapsed;
  if (layoutState.workspaceCollapsed) {
    layoutState.workspaceZoomed = false;
  }
  applyLayout();
}

function toggleWorkspaceZoom(force) {
  layoutState.workspaceZoomed = (force !== undefined) ? !!force : !layoutState.workspaceZoomed;
  if (layoutState.workspaceZoomed) {
    layoutState.workspaceCollapsed = false;
  }
  applyLayout();
}

function ensureWorkspaceVisible() {
  if (layoutState.workspaceCollapsed) {
    layoutState.workspaceCollapsed = false;
    applyLayout();
  }
}

let course, lesson, active = 'draft', worker, busy = false, timer, draftTimer;
let saved = {}, workerReady = false;
try { saved = JSON.parse(localStorage.getItem(STORAGE) || '{}'); } catch { /* local storage may be disabled */ }
const status = text => { $('python-status').textContent = text; };
function key(id = active) { return `${lesson.id}::${id}`; }
function persist() {
  if (!lesson) return;
  saved[key()] = {code: $('code').value, notes: $('notes').value, stdin: $('stdin').value, title: $('workspace-title').textContent};
  try { localStorage.setItem(STORAGE, JSON.stringify(saved)); $('saved').textContent = 'Saved on this browser'; }
  catch { $('saved').textContent = 'Storage full or unavailable — export your work'; }
}
function entries() {
  return [{id:'draft', title:'My lesson workspace'}, ...lesson.workspaces,
    ...lesson.codes.filter(c=>c.runnable).map(c => ({...c, title:`${c.id} · ${c.title}${c.solution ? ' (practice answer)' : ''}`}))];
}
function selectWork(id, initial) {
  persist(); active = id;
  const entry = entries().find(x => x.id === id) || {title:id};
  const draft = saved[key()];
  $('workspace-title').textContent = entry.title;
  $('code').value = draft?.code ?? initial ?? entry.code ?? '# Write your Python here.\n';
  $('notes').value = draft?.notes || ''; $('stdin').value = draft?.stdin || '';
  const options = entries();
  if (!options.some(x => x.id === id)) options.push({id, title:entry.title});
  // Include custom setup drafts imported or saved in earlier visits.
  for (const [k, v] of Object.entries(saved)) if (k.startsWith(lesson.id+'::')) {
    const local = k.slice(lesson.id.length+2);
    if (!options.some(x => x.id === local)) options.push({id:local,title:v.title || local});
  }
  $('workspace-select').replaceChildren(...options.map(e => { const o=document.createElement('option'); o.value=e.id;o.textContent=e.title;return o; }));
  $('workspace-select').value = id;
  $('workspace-note').textContent = (lesson.assessment || lesson.project)
    ? 'Your own answer only. Add your code and written reasoning, then export it for submission.'
    : 'Runs share variables within this lesson. Use preceding examples when a cell depends on earlier setup.';
  persist();
}
function renderNav() {
  const term=$('search').value.toLowerCase(); const groups = new Map();
  for(const p of course.pages) if ((p.title+' '+p.group).toLowerCase().includes(term)) {
    if(!groups.has(p.group)) groups.set(p.group,[]);groups.get(p.group).push(p);
  }
  $('lessons').replaceChildren(...[...groups].map(([name,pages]) => {
    const d=document.createElement('details');d.open=!!term || pages.some(p=>p.id===lesson?.id);
    const s=document.createElement('summary');s.textContent=name;d.append(s);
    for(const p of pages){const b=document.createElement('button');b.textContent=p.title;b.classList.toggle('active',p.id===lesson?.id);b.onclick=()=>navigate(p.id);d.append(b);}return d;
  }));
}
function navigate(id) { location.hash = 'lesson='+encodeURIComponent(id); }
async function route() {
  const params = new URLSearchParams(location.hash.slice(1));
  const next=course.pages.find(p=>p.id===params.get('lesson')) || course.pages[0];
  if(lesson?.id!==next.id){
    if(busy){location.hash='lesson='+encodeURIComponent(lesson.id);appendText('Stop the current program before changing lessons.','error');return;}
    persist();lesson=next;active='draft';
    $('output').replaceChildren();
    $('breadcrumb').textContent=lesson.group;
    $('lesson').innerHTML=lesson.html;
    if(lesson.assessment || lesson.project){const banner=document.createElement('p');banner.className='assessment-banner';banner.textContent='Questions and requirements only. Use the dedicated workspace buttons to write and save your own answers. No reference solution is included.';$('lesson').prepend(banner);}
    if(lesson.project){const note=document.createElement('p');note.className='callout';note.textContent='Sandbox dataset location: data/solar.csv, data/wind.csv, data/hydro.csv, data/geothermal.csv and data/demand.csv. These five separate files are loaded automatically into Python when you run code. Each project section has its own saved draft; the main workspace can hold your complete program.';$('lesson').prepend(note);}
    // Avoid persisting the preceding lesson's editor into this new lesson.
    $('code').value=saved[key()]?.code ?? '# Write your Python here.\n';
    $('notes').value=saved[key()]?.notes || ''; $('stdin').value=saved[key()]?.stdin || '';
    selectWork('draft');renderNav();$('reading').scrollTop=0;
    const outline=document.createElement('select');outline.setAttribute('aria-label','Jump to a section');
    const first=document.createElement('option');first.value='';first.textContent='On this page · jump to a section';outline.append(first);
    for(const h of $('lesson').querySelectorAll('h2,h3')) if(h.id){const o=document.createElement('option');o.value=h.id;o.textContent=h.textContent;outline.append(o);}
    outline.onchange=()=>$(outline.value)?.scrollIntoView({block:'start'});$('lesson-outline').replaceChildren(outline);
    if(workerReady) worker.postMessage({type:'context',path:lesson.id});
    if(window.MathJax?.typesetPromise) await window.MathJax.typesetPromise([$('lesson')]).catch(()=>{});
  }
  if(params.get('anchor')) document.getElementById(params.get('anchor'))?.scrollIntoView({block:'start'});
}
function appendText(text, className='') {const p=document.createElement('pre');p.textContent=text;p.className=className;$('output').append(p);}
function download(name, data, type='text/plain') {
  const url=URL.createObjectURL(new Blob([data],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function locked(value){busy=value;$('run').disabled=value;$('stop').disabled=!value;}
function createWorker(){
  worker?.terminate();workerReady=false;
  worker=new Worker('python-worker.js',{type:'module'});status('Downloading Python…');
  worker.onmessage=({data:m})=>{
    if(m.type==='status')status(m.text);
    if(m.type==='ready'){workerReady=true;status('Python ready');}
    if(m.type==='stdout')appendText(m.text);
    if(m.type==='stderr')appendText(m.text,'error');
    if(m.type==='image'){const image=document.createElement('img');image.src=m.data;image.alt='Figure produced by your Python code';$('output').append(image);}
    if(m.type==='html'){const frame=document.createElement('iframe');frame.title='Python HTML output';frame.setAttribute('sandbox','allow-scripts');frame.srcdoc=m.html;$('output').append(frame);}
    if(m.type==='download')download(m.name,m.bytes,'application/octet-stream');
    if(m.type==='done'){clearTimeout(timer);locked(false);status('Python ready');appendText(m.error || 'Finished.',m.error?'error':'muted');}
    if(m.type==='fatal'){clearTimeout(timer);locked(false);status('Python could not start');appendText(m.error+' Try Run again to reconnect.','error');worker.terminate();worker=null;workerReady=false;}
  };
  worker.onerror=e=>{clearTimeout(timer);locked(false);appendText('Python worker error: '+e.message,'error');worker?.terminate();worker=null;workerReady=false;status('Python stopped');};
  worker.postMessage({type:'init',assets:course.assets,path:lesson.id});
}
function run(){
  persist();if(busy)return;
  if(/\binput\s*\(/.test($('code').value) && !$('stdin').value){
    document.querySelector('details.tools').open=true;$('stdin').focus();
    appendText('This program asks for input. Enter one answer per line in Input values, then run it.','muted');return;
  }
  $('output').replaceChildren();locked(true);
  if(!worker)createWorker();
  worker.postMessage({type:'run',code:$('code').value,stdin:$('stdin').value,path:lesson.id});
  timer=setTimeout(()=>appendText('This run is taking longer than usual. You can stop it at any time.','muted'),60000);
}
$('run').onclick=run;
$('stop').onclick=()=>{worker?.terminate();worker=null;workerReady=false;clearTimeout(timer);locked(false);status('Stopped — Run starts a fresh Python session');appendText('Stopped. Drafts kept; runtime variables and uploads cleared.','muted');};
$('reset-runtime').onclick=()=>{if(busy)$('stop').click();else{worker?.terminate();worker=null;workerReady=false;status('Python will restart on the next run');appendText('Runtime cleared. Saved drafts are unchanged.','muted');}};
$('reset-code').onclick=()=>{if(confirm('Reset this draft to its starting code? Other drafts are kept.')){$('code').value=entries().find(x=>x.id===active)?.code || '# Write your Python here.\n';$('notes').value='';persist();}};
$('code').onkeydown=e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();run();}if(e.key==='Tab'){e.preventDefault();const el=e.target;el.setRangeText('    ',el.selectionStart,el.selectionEnd,'end');persist();}};
for(const id of ['code','notes','stdin']) $(id).oninput=()=>{clearTimeout(draftTimer);draftTimer=setTimeout(persist,250);};
window.addEventListener('beforeunload',persist);
$('lesson').onclick=e=>{
  const b=e.target.closest('button');if(!b)return;
  ensureWorkspaceVisible();
  if(b.dataset.workspace)selectWork(b.dataset.workspace);
  if(b.dataset.code)selectWork(b.dataset.code);
  if(b.dataset.setup){const index=lesson.codes.findIndex(c=>c.id===b.dataset.setup);const code=lesson.codes.slice(0,index+1).filter(c=>!c.solution&&c.runnable).map(c=>c.code).join('\n\n');selectWork('setup-'+b.dataset.setup,code);}
  if(window.innerWidth<1050)$('code').scrollIntoView({block:'center'});
};
$('workspace-select').onchange=e=>selectWork(e.target.value);
$('own-work').onclick=()=>{ensureWorkspaceVisible();selectWork('draft');};$('search').oninput=renderNav;
document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>navigate(b.dataset.jump));
$('help-button').onclick=()=>$('help').showModal();
$('clear-output').onclick=()=>$('output').replaceChildren();
$('download-code').onclick=()=>download(active+'.py',$('code').value);
$('export-work').onclick=()=>{persist();download('eage-participant-work.json',JSON.stringify({format:STORAGE,course:course.source_digest,drafts:saved},null,2),'application/json');};
$('import-work').onchange=async e=>{try{const data=JSON.parse(await e.target.files[0].text());if(data.format!==STORAGE||!data.drafts)throw Error('Not a sandbox work export');for(const [k,v] of Object.entries(data.drafts)){if(!k.includes('::')||typeof v.code!=='string'||typeof v.notes!=='string')throw Error('Invalid draft');}if(confirm('Import these drafts? Matching saved drafts will be replaced.')){persist();Object.assign(saved,data.drafts);localStorage.setItem(STORAGE,JSON.stringify(saved));const d=saved[key()];if(d){$('code').value=d.code;$('notes').value=d.notes;$('stdin').value=d.stdin||'';}selectWork(active);}}catch(err){appendText('Import failed: '+err.message,'error');}e.target.value='';};
$('upload').onchange=async e=>{if(busy){appendText('Stop or finish the run before uploading.','error');return;}if(!worker)createWorker();for(const file of e.target.files){const bytes=await file.arrayBuffer();worker.postMessage({type:'upload',name:file.name,bytes},[bytes]);}e.target.value='';};
$('download-file').onclick=()=>{if(!worker || busy){appendText('Run Python first and wait for it to finish.','error');return;}worker.postMessage({type:'download',path:$('download-path').value});};

// Layout toggle listeners
$('toggle-sidebar').onclick=()=>toggleSidebar();
$('collapse-sidebar').onclick=()=>toggleSidebar(false);
$('sidebar-restore-tab').onclick=()=>toggleSidebar(true);
$('reopen-sidebar-btn').onclick=()=>toggleSidebar(true);

$('toggle-workspace').onclick=()=>toggleWorkspace();
$('lab-collapse-btn').onclick=()=>toggleWorkspace(false);
$('workspace-restore-tab').onclick=()=>toggleWorkspace(true);
$('toggle-reading-focus').onclick=()=>toggleWorkspace(layoutState.workspaceCollapsed);

$('zoom-workspace').onclick=()=>toggleWorkspaceZoom();
$('lab-zoom-btn').onclick=()=>toggleWorkspaceZoom();

// Keyboard Shortcuts: Alt+S (Sidebar), Alt+W (Workspace), Alt+Z (Zoom)
window.addEventListener('keydown', e => {
  if (e.altKey && (e.key === 's' || e.key === 'S')) {
    e.preventDefault();
    toggleSidebar();
  } else if (e.altKey && (e.key === 'w' || e.key === 'W')) {
    e.preventDefault();
    toggleWorkspace();
  } else if (e.altKey && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault();
    toggleWorkspaceZoom();
  } else if (e.key === 'Escape' && layoutState.workspaceZoomed) {
    toggleWorkspaceZoom(false);
  }
});

// Apply initial layout state
applyLayout();
window.addEventListener('hashchange',()=>route());
try{const response=await fetch('generated/course.json');if(!response.ok)throw Error('Course content has not been built. Run ./start_sandbox.sh from the repository.');course=await response.json();$('book-link').href=course.book_url;await route();}catch(e){$('lesson').textContent=e.message;status('Course loading failed');}
