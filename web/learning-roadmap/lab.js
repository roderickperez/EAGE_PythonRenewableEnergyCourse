import {stations} from './stations.js';
import {startWorld} from './scene.js';
const el=id=>document.getElementById(id);
const key='eage-roadmap-checkpoints-v1';
let stored;
try {stored=JSON.parse(localStorage.getItem(key));} catch {}
const earned=new Set(stations.filter(s=>Array.isArray(stored)&&stored.includes(s.id)).map(s=>s.id));
let world,current=null,near=null;
const tasks={
  hydro:['Adjust head to deliver about 1 MW.',(v,p)=>p>=.95&&p<=1.05],
  solar:['Find an evening hour when the panel stops exporting.',(v,p)=>v>=18&&p===0],
  wind:['Find the wind speed that reaches 3 MW.',(v,p)=>p>=3],
  geothermal:['Run the plant at 30 kg/s and measure its output.',v=>v===30]
};
const samples=new Map();
function locationCard(site) {
  near=site;
  if(current)return;
  el('place').textContent=site?.name||'Energy crossing';
  el('inspect').hidden=!site;
  el('inspect').textContent=site?.plant?'Inspect plant':'Explore Python Island';
  el('hint').textContent=site?'Walk across the bridges or choose a destination.':'Follow a bridge to an energy site.';
}
function inspect(site) {
  current=site;
  el('console').hidden=!site;
  el('drive').hidden=!!site;
  document.querySelector('footer').hidden=!!site;
  el('mode').textContent=site?'Field station':'Island expedition';
  if(!site){locationCard(near);return;}
  el('place').textContent=site.name;
  el('hint').textContent=site.plant?'Drag to orbit · Pinch or scroll to zoom':'Open the field notes for your first Python exercise.';
  el('instrument').hidden=!site.plant;
  if(!site.plant)return;
  const [name,min,max,value,unit]=site.plant.control;
  const slider=el('setting');Object.assign(slider,{min,max,value});
  el('variable').textContent=name;
  slider.oninput=()=>{
    el('value').textContent=slider.value+' '+unit;
    const value=Number(slider.value);
    const output=site.plant.set(value);
    el('reading').textContent=output;
    const [task,meetsTarget]=tasks[site.id];
    if(meetsTarget(value,parseFloat(output)))samples.set(site.id,value+' '+unit+' → '+output);
    el('mission').textContent=samples.has(site.id)?'Reading recorded · '+samples.get(site.id):task;
  };
  slider.oninput();
}
function node(tag,text){const n=document.createElement(tag);n.textContent=text;return n;}
function link(text,href){const a=node('a',text);a.href=href;return a;}
function quizLink(s){return link(s.name+' course quiz ↗','../../index.html#lesson='+encodeURIComponent('section7/renewableEnergyquizzes/'+s.quiz+'.md'));}
function dialog(title){
  el('dialog-title').textContent=title;
  el('contents').replaceChildren();
  if(!el('drawer').open)el('drawer').showModal();
  world?.pause(true);
  return el('contents');
}
function routes(){
  const list=dialog('Destinations');
  list.append(node('p',earned.size===5?'Expedition complete · 5 checkpoints earned':earned.size+' of 5 checkpoints earned'));
  for(const s of stations){
    const b=node('button',s.name+(earned.has(s.id)?'   ✓':'   →'));
    b.onclick=()=>{el('drawer').close();if(world)world.travel(s.id);else notes(s);};
    list.append(b);
  }
}
function notes(s){
  const list=dialog(s.name+' field notes');
  list.append(node('p',s.description),node('p',s.task));
  if(s.id==='python'){
    list.append(node('pre','power_mw = [2, 4, 3]\nenergy_mwh = sum(power_mw) * 1\nprint(energy_mwh)  # 9 MWh'),link('Open the Python Course ↗','https://roderickperezaltamar.com/courses/eage-python-course.html'));
  }else list.append(link('Open full course lab ↗','../'+s.id+'-lab/index.html'));
  list.append(node('h3',s.question));
  const feedback=node('p','');feedback.setAttribute('role','status');
  const options=s.options.map((option,i)=>{
    const b=node('button',option);b.onclick=()=>{
      feedback.textContent=(i===s.correct?'Checkpoint earned. ':'Try again. ')+s.why;
      if(i!==s.correct)return;
      earned.add(s.id);
      try{localStorage.setItem(key,JSON.stringify([...earned]));}catch{}
      options.forEach(o=>o.disabled=true);
      if(earned.size===5){list.append(node('h3','Expedition complete'),...stations.map(quizLink));}
    };
    return b;
  });
  list.append(...options,feedback,quizLink(s));
}
try{
  world=startWorld(el('world'),stations,locationCard,inspect,text=>{el('hint').textContent=text;});
}catch(error){
  console.error(error);
  el('hint').textContent='3D rendering is unavailable in this browser. Open Destinations for the course labs and quizzes.';
  el('drive').hidden=true;
}
el('routes').onclick=routes;
el('inspect').onclick=()=>world.inspect();
el('back').onclick=()=>world.leave();
el('study').onclick=()=>notes(current);
el('close').onclick=()=>el('drawer').close();
el('drawer').addEventListener('close',()=>world?.pause(false));
const pad=el('stick');
function move(e){
  const r=pad.getBoundingClientRect();
  const x=(e.clientX-r.left-r.width/2)/(r.width/2),y=(e.clientY-r.top-r.height/2)/(r.height/2);
  const length=Math.max(1,Math.hypot(x,y));
  world?.stick.set(x/length,y/length);
  pad.firstElementChild.style.transform='translate('+x/length*21+'px,'+y/length*21+'px)';
}
pad.onpointerdown=e=>{pad.setPointerCapture(e.pointerId);move(e);};
pad.onpointermove=e=>{if(pad.hasPointerCapture(e.pointerId))move(e);};
pad.onlostpointercapture=()=>{world?.stick.set(0,0);pad.firstElementChild.style.transform='';};
