import {stations} from './stations.js';
import {drawMap} from './scene.js';

const el = id => document.getElementById(id);
const saveKey = 'eage-roadmap-checkpoints-v1';
let saved;
try { saved = JSON.parse(localStorage.getItem(saveKey)); } catch {}
const earned = new Set(stations.filter(s=>Array.isArray(saved) && saved.includes(s.id)).map(s=>s.id));
let selected;
function link(label, href) {
  const a = document.createElement('a');
  a.textContent = label;
  a.href = href;
  return a;
}
function courseQuiz(s) {
  return link(s.name+' quiz', '../../index.html#lesson='+encodeURIComponent('section7/renewableEnergyquizzes/'+s.quiz+'.md'));
}
const buttons = stations.map(s=>{
  const b=document.createElement('button');
  b.textContent=s.name;
  if(s.id==='python'){
    const img=new Image();
    img.src='python.svg';
    img.alt='';
    b.prepend(img);
  }
  b.addEventListener('click',()=>choose(s.id));
  el('markers').append(b);
  el('quiz-links').append(courseQuiz(s));
  return b;
});
function progress() {
  buttons.forEach((b,i)=>{
    b.classList.toggle('earned',earned.has(stations[i].id));
    b.setAttribute('aria-label',stations[i].name+(earned.has(stations[i].id)?', checkpoint earned':''));
  });
  el('progress').value=earned.size;
  el('count').textContent=earned.size+' of 5 checkpoints earned';
  el('finish').hidden=earned.size!==5;
}
function choose(id) {
  selected=stations.find(s=>s.id===id);
  buttons.forEach((b,i)=>b.setAttribute('aria-pressed',String(stations[i]===selected)));
  el('name').textContent=selected.name;
  el('description').textContent=selected.description;
  el('task').textContent=selected.task;
  el('enter').textContent=selected.id==='python'?'Explore Python Island':'Explore '+selected.name;
  el('activity').replaceChildren();
  el('checkpoint').replaceChildren();
  el('station').hidden=true;
}
function openStation() {
  const s=selected;
  el('station').hidden=false;
  el('station-title').textContent=s.name;
  const activity=el('activity');
  activity.replaceChildren();
  if(s.id==='python') {
    const pre=document.createElement('pre');
    pre.textContent='power_mw = [2, 4, 3]  # three one-hour intervals\nenergy_mwh = sum(power_mw) * 1\nprint(energy_mwh)     # 9 MWh';
    activity.append(pre,link('Open the Python Course ↗','https://roderickperezaltamar.com/courses/eage-python-course.html'));
  } else {
    const frame=document.createElement('iframe');
    frame.title=s.name+' interactive course lab';
    frame.src='../'+s.id+'-lab/index.html';
    activity.append(frame,link('Open full-size lab ↗',frame.src));
  }
  const question=document.createElement('h3');
  question.textContent=s.question;
  const choices=document.createElement('div');
  choices.className='choices';
  const feedback=document.createElement('p');
  feedback.setAttribute('role','status');
  const back=document.createElement('button');
  back.textContent='Return to island map';
  back.hidden=true;
  back.onclick=()=>{
    const next=stations.find(candidate=>!earned.has(candidate.id))||s;
    choose(next.id);
    if(earned.size===stations.length) el('finish').scrollIntoView();
    else buttons[stations.indexOf(next)].focus();
  };
  s.options.forEach((option,i)=>{
    const b=document.createElement('button');
    b.textContent=option;
    choices.append(b);
    b.onclick=()=>{
      if(i!==s.correct) { feedback.textContent='Try again. '+s.why; return; }
      earned.add(s.id);
      try {localStorage.setItem(saveKey,JSON.stringify([...earned]));} catch {}
      feedback.textContent='Checkpoint earned. '+s.why;
      back.hidden=false;
      for(const child of choices.children) child.disabled=true;
      progress();
    };
  });
  el('checkpoint').replaceChildren(question,choices,feedback,courseQuiz(s),back);
  el('station-title').focus();
}
el('enter').addEventListener('click',openStation);
choose(stations[0].id);
progress();
drawMap(el('terrain'),stations,buttons,choose);
