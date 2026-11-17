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

  // Update reading focus button


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

function setWorkspaceMode(notebook) {
  const frame = $('workspace-notebook');
  $('lab').classList.toggle('show-notebook', notebook);
  frame.hidden = !notebook;
  $('workspace-mode-editor').setAttribute('aria-pressed', String(!notebook));
  $('workspace-mode-notebook').setAttribute('aria-pressed', String(notebook));
  if (notebook && !frame.hasAttribute('src')) {
    frame.src = 'jupyterlite/lab/index.html?path=Start%20here.ipynb';
  }
}
$('workspace-mode-editor').onclick = () => setWorkspaceMode(false);
$('workspace-mode-notebook').onclick = () => setWorkspaceMode(true);

let course, lesson, active = 'draft', worker, busy = false, timer, draftTimer, lessonLab;
let saved = {}, workerReady = false;
try { saved = JSON.parse(localStorage.getItem(STORAGE) || '{}'); } catch { /* local storage may be disabled */ }
const status = (text, state) => { const indicator = $('python-status'); indicator.dataset.state = state; indicator.title = text; indicator.setAttribute('aria-label', text); indicator.querySelector('.sr-only').textContent = text; };
function key(id = active) { return `${lesson.id}::${id}`; }
function restoreWorkbookDrafts() {
  for (const [id, oldKey] of Object.entries(lesson?.legacy_drafts ?? {})) {
    if (saved[oldKey] && !saved[key(id)]) saved[key(id)] = { ...saved[oldKey] };
  }
}
function persist() {
  if (!lesson) return;
  saved[key()] = {code: $('code').value, notes: $('notes').value, stdin: $('stdin').value, title: $('workspace-title').textContent};
  try { localStorage.setItem(STORAGE, JSON.stringify(saved)); $('saved').textContent = 'Saved on this browser'; $('saved').classList.remove('save-error'); }
  catch { $('saved').textContent = 'Storage full or unavailable — export your work'; $('saved').classList.add('save-error'); }
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
  const isActive = entry => entry.id === lesson?.id;
  for(const p of course.pages) if ((p.title+' '+p.group).toLowerCase().includes(term)) {
    if(!groups.has(p.group)) groups.set(p.group,[]);groups.get(p.group).push(p);
  }
  $('lessons').replaceChildren(...[...groups].map(([name,pages]) => {
    const d=document.createElement('details');d.open=!!term || pages.some(isActive);
    const s=document.createElement('summary');s.textContent=name;d.append(s);
    for(const p of pages){const b=document.createElement('button');b.textContent=p.title;b.classList.toggle('active',isActive(p));b.onclick=()=>navigate(p.id);d.append(b);}return d;
  }));
}
function navigate(id) { location.hash = 'lesson='+encodeURIComponent(id); }
async function route() {
  const params = new URLSearchParams(location.hash.slice(1));
  const next=course.pages.find(p=>p.id===params.get('lesson')) || course.pages[0];
  if(lesson?.id!==next.id){
    if(busy){location.hash='lesson='+encodeURIComponent(lesson.id);appendText('Stop the current program before changing lessons.','error');return;}
    stopMedia();persist();lessonLab?.destroy();lessonLab=null;lesson=next;active='draft';
    restoreWorkbookDrafts();
    $('output').replaceChildren();
    $('breadcrumb').textContent=lesson.group;
    const notebookPaths = {
      "section0/introRenewableEnergy.md": "EAGE_PythonRenewableEnergyCourse/section0/introRenewableEnergy.ipynb",
      "section5/download-to-database.md": "EAGE_PythonRenewableEnergyCourse/section5/download-to-database.ipynb",
      "section5/world-energy-download.md": "EAGE_PythonRenewableEnergyCourse/section5/world-energy-download.ipynb",
      "section5/SQL_Pandas.ipynb": "EAGE_PythonRenewableEnergyCourse/section5/SQL_Pandas.ipynb",
      "section6/hydroelectricEnergy.md": "EAGE_PythonRenewableEnergyCourse/section6/hydroelectricEnergy.ipynb",
      "section6/solarEnergy.md": "EAGE_PythonRenewableEnergyCourse/section6/solarEnergy.ipynb",
      "section6/windEnergy.md": "EAGE_PythonRenewableEnergyCourse/section6/windEnergy.ipynb",
      "section6/geothermalEnergy.md": "EAGE_PythonRenewableEnergyCourse/section6/geothermalEnergy.ipynb"
    };
    const notebookPath = lesson.id.startsWith('section7/workbook/')
      ? 'EAGE_PythonRenewableEnergyCourse/' + lesson.id.replace(/\.md$/, '.ipynb')
      : notebookPaths[lesson.id];
    const colabButton = $('lesson-colab');
    colabButton.disabled = !notebookPath;
    colabButton.title = notebookPath ? 'Open this lesson in Google Colab' : 'No Google Colab notebook available for this lesson';
    colabButton.onclick = notebookPath ? () => window.open('https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/' + notebookPath, '_blank', 'noopener,noreferrer') : null;
    $('lesson').innerHTML=lesson.html;
    for (const [anchor, path, title, labPath = 'hydro-lab'] of [
      ['hydraulic-head-explorer', 'head.html', 'Hydraulic head explorer'],
      ['net-head-explorer', 'balance.html?demo=net-head', 'Net head explorer'],
      ['flow-volume-explorer', 'balance.html?demo=flow-volume', 'Flow and volume explorer'],
      ['hydraulic-power-explorer', 'balance.html?demo=hydraulic-power', 'Hydraulic power explorer'],
      ['electrical-power-explorer', 'balance.html?demo=electrical-power', 'Electrical power explorer'],
      ['hydraulic-loss-explorer', 'balance.html?demo=hydraulic-loss', 'Hydraulic losses explorer'],
      ['turbine-operation-explorer', 'balance.html?demo=turbine-operation', 'Turbine families and operation explorer'],
      ['pumped-storage-explorer', 'balance.html?demo=pumped-storage', 'Pumped storage explorer'],
      ['flow-duration-explorer', 'balance.html?demo=flow-duration', 'Hydrograph and flow-duration curve explorer'],
      ['solar-horizontal-explorer', 'resource.html?demo=horizontal', 'Horizontal irradiance explorer', 'solar-lab'],
      ['solar-poa-explorer', 'resource.html?demo=poa', 'Plane-of-array irradiance explorer', 'solar-lab'],
      ['solar-irradiation-explorer', 'resource.html?demo=irradiation', 'Interval irradiation explorer', 'solar-lab'],
      ['solar-tilt-explorer', 'resource.html?demo=tilt', 'Tilted-module irradiance explorer', 'solar-lab'],
      ['solar-incidence-explorer', 'resource.html?demo=incidence', 'Solar incidence angle explorer', 'solar-lab'],
      ['solar-iv-explorer', 'resource.html?demo=iv', 'Photovoltaic current and power curve explorer', 'solar-lab'],
      ['solar-operating-point-explorer', 'resource.html?demo=operating-point', 'Photovoltaic operating point and DC power explorer', 'solar-lab'],
      ['solar-fill-factor-explorer', 'resource.html?demo=fill-factor', 'Photovoltaic fill factor explorer', 'solar-lab'],
      ['solar-dc-temperature-explorer', 'resource.html?demo=dc-temperature', 'Temperature-corrected solar DC power explorer', 'solar-lab'],
      ['solar-cell-temperature-explorer', 'resource.html?demo=cell-temperature', 'Solar cell temperature and NOCT explorer', 'solar-lab'],
      ['solar-inverter-clipping-explorer', 'resource.html?demo=inverter-clipping', 'Solar inverter clipping and loss explorer', 'solar-lab'],
      ['solar-thermal-collector-explorer', 'resource.html?demo=thermal-collector', 'Solar thermal collector heat-balance explorer', 'solar-lab'],
      ['wind-cube-law-explorer', 'aerodynamics.html?demo=cube-law', 'Available wind power and the cube law explorer', 'wind-lab'],
      ['wind-betz-explorer', 'aerodynamics.html?demo=betz', 'Rotor extraction and Betz limit explorer', 'wind-lab'],
      ['wind-air-density-explorer', 'aerodynamics.html?demo=air-density', 'Dry-air density and available wind power explorer', 'wind-lab'],
      ['wind-shear-explorer', 'aerodynamics.html?demo=shear', 'Hub-height wind shear explorer', 'wind-lab'],
      ['wind-turbulence-explorer', 'aerodynamics.html?demo=turbulence', 'Turbulence and averaging explorer', 'wind-lab'],
      ['wind-electrical-curve-explorer', 'aerodynamics.html?demo=electrical-curve', 'Electrical wind turbine power curve explorer', 'wind-lab'],
      ['wind-annual-energy-explorer', 'aerodynamics.html?demo=annual-energy', 'From wind resource to annual energy explorer', 'wind-lab'],
      ['wind-weibull-explorer', 'aerodynamics.html?demo=weibull', 'Weibull wind resource and bin integration explorer', 'wind-lab'],
      ['wind-losses-explorer', 'aerodynamics.html?demo=losses', 'Wind energy losses and uncertainty explorer', 'wind-lab'],
      ['wind-direction-explorer', 'aerodynamics.html?demo=direction', 'Circular wind direction explorer', 'wind-lab'],
      ['geo-darcy-explorer', 'equations.html?demo=darcy', 'Darcy flux explorer', 'geothermal-lab'],
      ['geo-gradient-explorer', 'equations.html?demo=gradient', 'Geothermal gradient and heat flux explorer', 'geothermal-lab'],
      ['geo-heat-explorer', 'equations.html?demo=heat', 'Thermal power explorer', 'geothermal-lab'],
      ['geo-carnot-explorer', 'equations.html?demo=carnot', 'Carnot efficiency explorer', 'geothermal-lab'],
      ['geo-conversion-explorer', 'equations.html?demo=conversion', 'Gross and net power explorer', 'geothermal-lab'],
      ['geo-pump-explorer', 'equations.html?demo=pump', 'Pumping power explorer', 'geothermal-lab'],
      ['geo-heatpump-explorer', 'equations.html?demo=heatpump', 'Heat pump explorer', 'geothermal-lab'],
      ['geo-storage-explorer', 'equations.html?demo=storage', 'Reservoir stored heat explorer', 'geothermal-lab'],
      ['geo-decline-explorer', 'equations.html?demo=decline', 'Production decline explorer', 'geothermal-lab'],
      ['geo-energy-explorer', 'equations.html?demo=energy', 'Energy and capacity factor explorer', 'geothermal-lab'],
      ['geo-worked-explorer', 'equations.html?demo=conversion', 'Worked calculation explorer', 'geothermal-lab'],
      ['geo-compact-gradient-explorer', 'equations.html?demo=gradient', 'Gradient and Fourier law explorer', 'geothermal-lab'],
      ['geo-compact-heat-explorer', 'equations.html?demo=heat', 'Thermal power and enthalpy explorer', 'geothermal-lab'],
      ['geo-compact-efficiency-explorer', 'equations.html?demo=conversion', 'Conversion and net power explorer', 'geothermal-lab'],
      ['geo-compact-carnot-explorer', 'equations.html?demo=carnot', 'Carnot limit explorer', 'geothermal-lab'],
      ['geo-compact-energy-explorer', 'equations.html?demo=energy', 'Energy and availability explorer', 'geothermal-lab'],
      ['geo-compact-decline-explorer', 'equations.html?demo=decline', 'Annual decline explorer', 'geothermal-lab'],
    ]) {
      const spot = $(anchor);
      if (!spot) continue;
      const frame = document.createElement('iframe');
      frame.className = { 'hydro-lab': 'hydraulic-head-explorer', 'solar-lab': 'solar-resource-explorer',
        'wind-lab': 'wind-equation-explorer', 'geothermal-lab': 'geothermal-equation-explorer' }[labPath];
      frame.title = title;
      frame.loading = 'lazy';
      frame.src = `generated/${labPath}/${path}`;
      spot.after(frame);
    }
    const lab = {
      'section6/hydroelectricEnergy.md': {
        anchor: '#physical-description-water-elevation-and-electricity', id: 'hydro-3d-lab',
        mount: async host => (await import('./generated/hydro-lab/hydro-lab.js'))
          .mountHydroLab(host, undefined, { assetBase: 'generated/hydro-lab/', standaloneUrl: '' }),
      },
      'section6/solarEnergy.md': {
        anchor: '#physical-description-what-solar-energy-systems-do', id: 'solar-3d-lab',
        mount: async host => (await import('./generated/solar-lab/solar-lab.js')).mountSolarLab(host),
      },
      'section6/windEnergy.md': {
        anchor: '#physical-description-extracting-energy-from-moving-air', id: 'wind-3d-lab',
        mount: async host => (await import('./generated/wind-lab/wind-lab.js')).mountWindLab(host),
      },
      'section6/geothermalEnergy.md': {
        anchor: '#physical-description-useful-heat-from-the-subsurface', id: 'geothermal-3d-lab',
        mount: async host => (await import('./generated/geothermal-lab/geothermal-lab.js')).mountGeothermalLab(host),
      },
      'section8/projectIntro.md': {
        anchor: '#load-the-separate-input-datasets', id: 'project-3d-map',
        mount: async host => (await import('./generated/project-map/project-map.js')).mountProjectMap(host),
      },
      'section8/wind-project.md': {
        anchor: '#how-to-use-the-explorer', id: 'wind-farm-project',
        mount: async host => (await import('./generated/wind-lab/wind-project.js?v=wind-title-1')).mountWindProject(host),
      },
      'section8/solar-project.md': {
        anchor: '#how-to-use-the-explorer', id: 'solar-farm-project',
        mount: async host => (await import('./generated/solar-lab/solar-project.js?v=solar-project-1')).mountSolarProject(host),
      },
    }[lesson.id];
    if(lab){
      const content = $('lesson');
      const host = document.createElement('div');
      host.id = lab.id;
      host.className = 'hydro-3d-lab';
      (content.querySelector(lab.anchor) || content.firstElementChild)?.before(host);
      lab.mount(host).then(instance => {
        if (host.isConnected) lessonLab = instance;
        else instance.destroy();
      }).catch(error => {
        host.className = 'callout';
        host.textContent = 'The interactive lab could not load. The lesson equations and examples remain available below.';
        console.error('Course lab failed to load', error);
      });
    }
    setupMedia();
    if(lesson.assessment || lesson.project){const banner=document.createElement('p');banner.className='assessment-banner';banner.textContent='Questions and requirements only. Use the dedicated workspace buttons to write and save your own answers. No reference solution is included.';$('lesson').prepend(banner);}
    if(lesson.project){const note=document.createElement('p');note.className='callout';note.textContent='Sandbox dataset location: data/solar.csv, data/wind.csv, data/hydro.csv, data/geothermal.csv and data/demand.csv. These five separate files are loaded automatically into Python when you run code. Each project section has its own saved draft; the main workspace can hold your complete program.';$('lesson').prepend(note);}
    // Avoid persisting the preceding lesson's editor into this new lesson.
    $('code').value=saved[key()]?.code ?? '# Write your Python here.\n';
    $('notes').value=saved[key()]?.notes || ''; $('stdin').value=saved[key()]?.stdin || '';
    selectWork('draft');$('reading').scrollTop=0;
    const outline=document.createElement('select');outline.setAttribute('aria-label','Jump to a section');
    const first=document.createElement('option');first.value='';first.textContent='On this page · jump to a section';outline.append(first);
    for(const h of $('lesson').querySelectorAll('h2,h3')) if(h.id){const o=document.createElement('option');o.value=h.id;o.textContent=h.textContent;outline.append(o);}
    outline.onchange=()=>$(outline.value)?.scrollIntoView({block:'start'});$('lesson-outline').replaceChildren(outline);
    if(workerReady) worker.postMessage({type:'context',path:lesson.id});
    if(window.MathJax?.typesetPromise) await window.MathJax.typesetPromise([$('lesson')]).catch(()=>{});
  }
  renderNav();
  if(params.get('anchor')) document.getElementById(params.get('anchor'))?.scrollIntoView({block:'start'});
}
// Media URLs come only from the validated build manifest. One player owns each queue.
let audioQueue = [], audioIndex = 0;
const microphone = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4m-3 0h6"/></svg>';
function stopMedia(){
  for(const id of ['audio-player','video-player']) { const player=$(id); player.pause(); player.removeAttribute('src'); player.load(); }
  audioQueue=[]; $('media-player').hidden=true; $('media-status').textContent='';
}
function playAudio(items){
  if(!items.length)return; stopMedia(); audioQueue=items; audioIndex=0;
  $('media-player').hidden=false; $('audio-player').hidden=false; $('video-player').hidden=true; playNextAudio();
}
function playNextAudio(){
  const item=audioQueue[audioIndex]; if(!item){$('media-status').textContent='Audio finished.'; return;}
  const player=$('audio-player'); player.src=item.url; $('media-status').textContent=item.title;
  player.play().catch(()=>{$('media-status').textContent='Press play to listen: '+item.title;});
}
$('audio-player').addEventListener('ended',()=>{audioIndex++;playNextAudio();});
$('audio-player').addEventListener('error',()=>{if(audioQueue.length) $('media-status').textContent='Audio could not be loaded.';});
function setupMedia(){
  const lang=document.documentElement.lang === 'es' ? 'es' : 'en';
  const configured=lesson.media?.audio || {};
  const audio=configured.en || configured.es ? (configured[lang] || {}) : (lang === 'en' ? configured : {});
  const headings=[...$('lesson').querySelectorAll('h2[id],h3[id],h4[id]')].filter(heading =>
    !['head-flow-and-the-hydraulic-energy-balance', 'water-allocation-hydrology-and-storage',
      'solar-resource-and-measurement-definitions', 'pv-conversion-definitions-equations-and-limits',
      'deriving-the-main-aerodynamic-relationships', 'resource-characterization-density-height-and-variability'].includes(heading.id) &&
    (heading.tagName === 'H2' || lesson.id === 'section6/geothermalEnergy.md' || audio[heading.id]));
  const items=[];
  if(audio.intro) items.push({url:audio.intro,title:lesson.title});
  for(const heading of headings){
    const title=heading.textContent.trim(), url=audio[heading.id];
    const button=document.createElement('button'); button.type='button';button.className='section-audio';button.innerHTML=microphone;
    button.disabled=!url;button.title=url?'Play audio: '+title:'Audio unavailable: '+title;button.setAttribute('aria-label',button.title);
    if(url){const item={url,title};items.push(item);button.onclick=()=>playAudio([item]);}
    heading.append(button);
  }
  const all=$('lesson-audio'); all.disabled=!items.length || headings.some(h=>!audio[h.id]);
  all.onclick=()=>playAudio(items);
  const video=$('lesson-video');const videoUrl=typeof lesson.media?.video === 'string' ? (lang === 'en' ? lesson.media.video : null) : lesson.media?.video?.[lang];video.disabled=!videoUrl;
  video.onclick=()=>{if(!videoUrl)return;stopMedia();$('media-player').hidden=false;$('audio-player').hidden=true;const player=$('video-player');player.hidden=false;player.src=videoUrl;player.play().catch(()=>{$('media-status').textContent='Press play to watch the video.';});};
}
function appendText(text, className='') {const p=document.createElement('pre');p.textContent=text;p.className=className;$('output').append(p);}
function download(name, data, type='text/plain') {
  const url=URL.createObjectURL(new Blob([data],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function locked(value){busy=value;$('run').disabled=value;$('stop').disabled=!value;}
function createWorker(){
  worker?.terminate();workerReady=false;
  worker=new Worker('python-worker.js',{type:'module'});status('Downloading Python…', 'loading');
  worker.onmessage=({data:m})=>{
    if(m.type==='status')status(m.text, /^loading|^downloading/i.test(m.text) ? 'loading' : (workerReady ? 'ready' : 'loading'));
    if(m.type==='ready'){workerReady=true;status('Python ready', 'ready');}
    if(m.type==='stdout')appendText(m.text);
    if(m.type==='stderr')appendText(m.text,'error');
    if(m.type==='image'){const image=document.createElement('img');image.src=m.data;image.alt='Figure produced by your Python code';$('output').append(image);}
    if(m.type==='html'){const frame=document.createElement('iframe');frame.title='Python HTML output';frame.setAttribute('sandbox','allow-scripts');frame.srcdoc=m.html;$('output').append(frame);}
    if(m.type==='download')download(m.name,m.bytes,'application/octet-stream');
    if(m.type==='done'){clearTimeout(timer);locked(false);status('Python ready', 'ready');appendText(m.error || 'Finished.',m.error?'error':'muted');}
    if(m.type==='fatal'){clearTimeout(timer);locked(false);status('Python could not start', 'offline');appendText(m.error+' Try Run again to reconnect.','error');worker.terminate();worker=null;workerReady=false;}
  };
  worker.onerror=e=>{clearTimeout(timer);locked(false);appendText('Python worker error: '+e.message,'error');worker?.terminate();worker=null;workerReady=false;status('Python stopped', 'offline');};
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
$('stop').onclick=()=>{worker?.terminate();worker=null;workerReady=false;clearTimeout(timer);locked(false);status('Stopped — Run starts a fresh Python session', 'offline');appendText('Stopped. Drafts kept; runtime variables and uploads cleared.','muted');};
$('reset-runtime').onclick=()=>{if(busy)$('stop').click();else{worker?.terminate();worker=null;workerReady=false;status('Python will restart on the next run', 'offline');appendText('Runtime cleared. Saved drafts are unchanged.','muted');}};
$('reset-code').onclick=()=>{if(confirm('Reset this draft to its starting code? Other drafts are kept.')){$('code').value=entries().find(x=>x.id===active)?.code || '# Write your Python here.\n';$('notes').value='';persist();}};
$('code').onkeydown=e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();run();}if(e.key==='Tab'){e.preventDefault();const el=e.target;el.setRangeText('    ',el.selectionStart,el.selectionEnd,'end');persist();}};
for(const id of ['code','notes','stdin']) $(id).oninput=()=>{clearTimeout(draftTimer);draftTimer=setTimeout(persist,250);};
window.addEventListener('beforeunload',persist);
$('lesson').onclick=e=>{
  const b=e.target.closest('button');if(!b || !(b.dataset.workspace || b.dataset.code || b.dataset.setup))return;
  ensureWorkspaceVisible();
  setWorkspaceMode(false);
  if(b.dataset.workspace)selectWork(b.dataset.workspace);
  if(b.dataset.code)selectWork(b.dataset.code);
  if(b.dataset.setup){const index=lesson.codes.findIndex(c=>c.id===b.dataset.setup);const code=lesson.codes.slice(0,index+1).filter(c=>!c.solution&&c.runnable).map(c=>c.code).join('\n\n');selectWork('setup-'+b.dataset.setup,code);}
  if(window.innerWidth<1050)$('code').scrollIntoView({block:'center'});
};
$('workspace-select').onchange=e=>selectWork(e.target.value);
$('search').oninput=renderNav;
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

$('toggle-workspace').onclick=()=>{toggleWorkspace();if(window.innerWidth<=1050&&!layoutState.workspaceCollapsed)$('lab').scrollIntoView({block:'start',behavior:'smooth'});};

$('zoom-workspace').onclick=()=>{toggleWorkspaceZoom();if(window.innerWidth<=1050&&layoutState.workspaceZoomed)$('lab').scrollIntoView({block:'start',behavior:'smooth'});};

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
setWorkspaceMode(true);
window.addEventListener('hashchange',()=>route());
try{if(window.refreshCourseSession)await window.refreshCourseSession();const response=await fetch('generated/course.json',{cache:'no-store'});if(!response.ok)throw Error('Course content could not be loaded. If using the private course, sign in again; for local use, run ./start_sandbox.sh.');course=await response.json();$('book-link').href=course.book_url;await route();}catch(e){$('lesson').textContent=e.message;status('Course loading failed', 'offline');}

// Confirm before leaving the course for the main menu.
$('home-menu').addEventListener('click', () => {
  if (window.confirm('Return to the main menu?')) window.location.assign('https://roderickperezaltamar.com/#courses');
});
