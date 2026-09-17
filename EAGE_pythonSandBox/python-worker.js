// Python lives in a worker so Stop can always terminate a runaway program.
// Pinned together with the documented package ABI; never mix fallback versions.
const INDEX = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
let py, ready, assets=[], context='', globals, queue=Promise.resolve();
let outputCharacters=0;
const send=(type, fields={})=>self.postMessage({type,...fields});
function stream(type,text){
  const previous=outputCharacters;outputCharacters+=text.length;
  if(previous<150000)send(type,{text:text.slice(0,150000-previous)});
  if(previous<150000 && outputCharacters>=150000)send('stderr',{text:'Output truncated at 150,000 characters. Stop the program if it is still running.'});
}
async function fetchAsset(asset){
  const target='/course/'+asset.path;
  if(py.FS.analyzePath(target).exists)return;
  const response=await fetch(new URL(asset.url,self.location.href));
  if(!response.ok)throw Error(`Cannot load course dataset: ${asset.path}`);
  const bytes=new Uint8Array(await response.arrayBuffer());
  py.FS.mkdirTree(target.slice(0,target.lastIndexOf('/')));py.FS.writeFile(target,bytes);
}
async function setContext(path){
  if(context===path)return;
  context=path;
  globals?.destroy();globals=py.runPython("{'__name__': '__main__'}");
  const cwd='/course/'+path.slice(0,path.lastIndexOf('/'));
  py.FS.mkdirTree(cwd);py.FS.chdir(cwd);
}
async function boot(message){
  assets=message.assets;
  const {loadPyodide}=await import(INDEX+'pyodide.mjs');
  py=await loadPyodide({indexURL:INDEX,stdout:text=>stream('stdout',text),stderr:text=>stream('stderr',text)});
  py.FS.mkdirTree('/course');py.FS.mkdirTree('/uploads');
  // Both root-detection conventions used by the book resolve to the same files.
  py.FS.symlink('/course','/home/pyodide/EAGE_PythonRenewableEnergyCourse');
  py.FS.symlink('/course','/course/EAGE_PythonRenewableEnergyCourse');
  await fetchAsset(assets.find(a=>a.path==='myst.yml'));
  await setContext(message.path);
  await py.runPythonAsync("import os, builtins\nos.environ['MPLBACKEND']='Agg'");
  send('ready');
}
const DISPLAY_SETUP = `
import sys as _sys, io as _io, base64 as _base64
from js import postMessage as _post
from pyodide.ffi import to_js as _to_js
def _emit(kind, **kw):
    _post(_to_js(dict(type=kind, **kw)))
def _display(obj):
    if hasattr(obj, '_repr_html_'):
        value = obj._repr_html_()
        if value:
            _emit('html', html=str(value))
            return
    print(obj)
def _flush_figures(*args, **kwargs):
    if 'matplotlib.pyplot' not in _sys.modules:
        return
    plt = _sys.modules['matplotlib.pyplot']
    for number in plt.get_fignums():
        buf = _io.BytesIO()
        plt.figure(number).savefig(buf, format='png', dpi=110, bbox_inches='tight')
        _emit('image', data='data:image/png;base64,' + _base64.b64encode(buf.getvalue()).decode())
    plt.close('all')
if 'matplotlib.pyplot' in _sys.modules:
    _sys.modules['matplotlib.pyplot'].show = _flush_figures
if 'plotly.io' in _sys.modules:
    def _plotly_show(fig, *args, **kwargs):
        _emit('html', html=fig.to_html(include_plotlyjs='cdn', full_html=True))
    _sys.modules['plotly.io'].show = _plotly_show
display = _display
`;
async function run(message){
  try{
    outputCharacters=0;
    if(message.fresh)context='';
    await setContext(message.path);
    if(/^\s*[%!]/m.test(message.code))throw Error('This cell contains Jupyter magic or a terminal command. Use the book’s local/Colab environment, or remove the installation line before running ordinary Python here.');
    send('status',{text:'Loading course data…'});
    // Small public teaching datasets; lazy-load larger spreadsheets only when the
    // relevant lesson needs them. No instructor directory is in this manifest.
    const needed=assets.filter(a=>/\.(csv|db|nc|geojson|xlsx)$/.test(a.path) && (
      a.path.startsWith('section8/data/') || a.path.startsWith('data/examples/') ||
      (message.path.startsWith('section5/') && (a.path.endsWith('.db')||a.path.endsWith('.xlsx'))) ||
      (message.path.startsWith('section4/') && !a.path.endsWith('.db')) ||
      message.code.includes(a.path.split('/').at(-1))));
    await Promise.all(needed.map(fetchAsset));
    send('status',{text:'Loading Python packages…'});
    const source=message.code;
    await py.loadPackagesFromImports(source,{messageCallback:text=>send('status',{text}),errorCallback:text=>send('stderr',{text})});
    if(/xarray|to_netcdf|open_dataset/.test(source))await py.loadPackage('scipy');
    // A few pure-Python packages are outside the core distribution.
    const extras=[['plotly','plotly==6.3.0'],['folium','folium==0.20.0'],['openpyxl','openpyxl==3.1.5'],['seaborn','seaborn==0.13.2']];
    for(const [module,spec] of extras)if(new RegExp('(?:import|from)\\s+'+module+'\\b').test(source) || (module==='openpyxl' && /read_excel|ExcelFile/.test(source))){
      await py.loadPackage('micropip');const pip=py.pyimport('micropip');try{await pip.install(spec);}finally{pip.destroy();}
    }
    if(/matplotlib|seaborn/.test(source)){
      await py.loadPackage('matplotlib');await py.runPythonAsync("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot");
    }
    if(/plotly/.test(source))await py.runPythonAsync('import plotly.io');
    await py.runPythonAsync(DISPLAY_SETUP,{globals});
    const inputs=message.stdin.split('\n');let at=0;
    py.setStdin({stdin:()=>at<inputs.length?inputs[at++]:null});
    send('status',{text:'Running Python…'});
    const result=await py.runPythonAsync(source,{globals});
    if(result!==undefined && result!==null){
      globals.set('_cell_result',result);await py.runPythonAsync('_display(_cell_result)',{globals});
      if(result.destroy)result.destroy();
    }
    await py.runPythonAsync('_flush_figures()',{globals});
    send('done');
  }catch(error){send('done',{error:String(error.message || error)});}
}
self.onmessage=({data:message})=>{
  if(message.type==='init'){ready=boot(message);ready.catch(error=>send('fatal',{error:String(error)}));return;}
  // Serializing prevents uploads, context resets and runs from racing each other.
  queue=queue.then(async()=>{
    await ready;
    if(message.type==='run')await run(message);
    if(message.type==='context')await setContext(message.path);
    if(message.type==='upload'){
      const name=message.name.replace(/[^\w. -]/g,'_');py.FS.writeFile('/uploads/'+name,new Uint8Array(message.bytes));send('stdout',{text:'Uploaded /uploads/'+name});
    }
    if(message.type==='download'){
      const bytes=py.FS.readFile(message.path);send('download',{name:message.path.split('/').at(-1),bytes});
    }
  }).catch(error=>send('stderr',{text:String(error.message || error)}));
};
