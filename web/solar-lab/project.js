import { SITES, PANEL_COST, SALVAGE, PRICE, conditions, declination, calculate } from './project-model.js';
import { mountProjectScene } from './project-scene.js';
import css from './project.css';

export function mountSolarProject(host) {
  const style=document.createElement('style');style.textContent=css;host.append(style);
  const root=document.createElement('section');root.className='solar-project';host.append(root);
  root.innerHTML=`<h2>Solar farm layout explorer</h2><p class="sp-intro">Choose a location on the spinning globe, then build a solar farm and explore how its orientation and weather affect generation.</p>
    <div class="sp-layout"><div class="sp-scene" tabindex="0" role="application" aria-label="3D solar farm. Use arrow keys to choose a cell and Enter to add or remove a panel."></div>
    <div class="sp-controls"><div class="sp-section" data-section="site"><label>Choose a location<select data-in="site">${SITES.map((s,i)=>`<option value="${i}">${s.name}</option>`).join('')}<option value="custom">Custom coordinates</option></select></label>
      <div class="sp-row"><label>Latitude (°)<input data-in="lat" type="number" min="-90" max="90" step="0.01" value="48.208"></label><label>Longitude (°)<input data-in="lon" type="number" min="-180" max="180" step="0.01" value="16.374"></label></div>
      <p class="sp-note">The gold marker shows your selected site. Enter coordinates for any location; the globe is schematic.</p><button data-action="start">Explore this site →</button></div>
      <div class="sp-section" data-section="farm" hidden><div class="sp-card"><span>Available budget</span><strong data-out="balance"></strong><small>Panel $1,000 · removal returns $500 · generation earns $0.10/kWh</small></div>
      <div class="sp-row"><label>Panel tilt <output data-out="tilt"></output><input data-in="tilt" type="range" min="0" max="90" value="30"></label><label>Azimuth <output data-out="azimuth"></output><input data-in="azimuth" type="range" min="0" max="359" value="180"></label></div>
      <label>Shading / soiling loss <output data-out="shading"></output><input data-in="shading" type="range" min="0" max="80" value="5"></label>
      <div class="sp-row"><label>Scenario date<input data-in="date" type="date" value="2025-06-21"></label><label>Solar time <output data-out="hour"></output><input data-in="hour" type="range" min="0" max="23.95" step="0.05" value="8"></label></div>
      <div class="sp-button-row"><button data-action="play">Pause simulation</button><button data-action="change">Change location</button></div><p class="sp-time" data-out="weather"></p><small>24 simulated hours per 60 visible seconds. Solar time, not local clock time. Weather is synthetic.</small></div></div></div>
    <p class="sp-status" role="status" aria-live="polite">Select a location and start the explorer.</p><div class="sp-section" data-section="results" hidden>
    <div class="sp-metrics"><div><span>Panels</span><strong data-out="count">0</strong></div><div><span>Current AC power</span><strong data-out="power">0 kW</strong></div><div><span>Energy generated</span><strong data-out="energy">0 kWh</strong></div><div><span>Day estimate</span><strong data-out="daily">0 kWh</strong></div><div><span>Sun elevation</span><strong data-out="elevation">0°</strong></div></div>
    <h3>Power over the solar day</h3><svg class="sp-chart" viewBox="0 0 600 170" preserveAspectRatio="none" role="img" aria-label="Estimated AC power over the current solar day with the current time marked"></svg><p class="sp-note">The curve uses the current date, layout and weather for the full day. The gold line marks current solar time; cumulative energy tracks the actual evolving simulation.</p></div>`;
  const $=s=>root.querySelector(s), input=n=>$(`[data-in="${n}"]`), out=n=>$(`[data-out="${n}"]`), section=n=>$(`[data-section="${n}"]`),status=$('.sp-status');
  let site={...SITES[0]}, ids=[], selected=0, balance=5000,energy=0,playing=true,started=false,last=performance.now(),variation=0,siteDate='2025-06-21',hour=8,scene;
  const settings=()=>({tilt:+input('tilt').value,azimuth:+input('azimuth').value,shading:+input('shading').value,count:ids.length});
  const date=()=>new Date(siteDate+'T12:00:00Z');
  const weather=()=>conditions(date(),site.lat,variation);
  const money=v=>'$'+Math.floor(v).toLocaleString('en-US');
  try{scene=mountProjectScene($('.sp-scene'),id=>{selected=id;toggle();});scene.setSite(site);}catch(e){$('.sp-scene').textContent='The 3D view requires WebGL. Controls and results remain available.';}
  function syncSite(){
    const lat=Number(input('lat').value),lon=Number(input('lon').value);
    if(!input('lat').value||!input('lon').value||!Number.isFinite(lat)||!Number.isFinite(lon)||Math.abs(lat)>90||Math.abs(lon)>180){status.textContent='Enter a latitude from −90 to 90 and longitude from −180 to 180.';return false;}
    site={name:input('site').value==='custom'?'Custom location':SITES[+input('site').value].name,lat,lon};scene?.setSite(site);return true;
  }
  input('site').onchange=()=>{if(input('site').value!=='custom'){const s=SITES[+input('site').value];input('lat').value=s.lat;input('lon').value=s.lon;}syncSite();};
  for(const k of ['lat','lon'])input(k).oninput=()=>{input('site').value='custom';syncSite();};
  function graph(result){const samples=result.day.samples, max=Math.max(.1,...samples.map(s=>s.ac/1000))*1.1;
    const x=h=>40+h/24*540,y=p=>140-p/max*115;
    const line=samples.map((s,i)=>`${i?'L':'M'}${x((i+.5)/12).toFixed(1)} ${y(s.ac/1000).toFixed(1)}`).join(' ');
    $('.sp-chart').innerHTML=`<path d="M40 12V140H580" fill="none" stroke="#9aafb8"/><path d="${line}" fill="none" stroke="#26839b" stroke-width="3"/><path d="M${x(hour)} 12V140" stroke="#db9e26" stroke-width="2" stroke-dasharray="5 4"/><text x="37" y="17" text-anchor="end">${max.toFixed(1)}</text><text x="37" y="143" text-anchor="end">0</text><text x="41" y="158">00</text><text x="305" y="158" text-anchor="middle">12 · solar time</text><text x="580" y="158" text-anchor="end">24</text><text x="45" y="25">kW</text>`;}
  function render(){if(!started)return;const set=settings(),w=weather(),result=calculate(site,date(),hour,set,w);
    out('tilt').textContent=`${set.tilt}°`;out('azimuth').textContent=`${set.azimuth}° from north`;out('shading').textContent=`${set.shading}%`;out('hour').textContent=`${String(Math.floor(hour)).padStart(2,'0')}:${String(Math.floor((hour%1)*60)).padStart(2,'0')}`;
    out('balance').textContent=money(balance);out('count').textContent=ids.length;out('power').textContent=(result.current.ac/1000).toFixed(2)+' kW';out('energy').textContent=energy.toFixed(2)+' kWh';out('daily').textContent=result.day.energy.toFixed(2)+' kWh';out('elevation').textContent=result.current.elevation.toFixed(1)+'°';
    out('weather').textContent=`${site.name} · ${siteDate} · synthetic cloud cover ${(w.clouds*100).toFixed(0)}% · air ${w.ambient.toFixed(1)} °C`;graph(result);scene?.update(ids,set,site,date(),hour,declination(date()),selected);}
  function toggle(){if(!started)return;if(ids.includes(selected)){ids=ids.filter(i=>i!==selected);balance+=SALVAGE;status.textContent=`Panel removed from cell ${selected+1}; ${money(SALVAGE)} recovered.`;}
    else if(balance<PANEL_COST)status.textContent='Insufficient budget; wait for generation revenue.';
    else{ids.push(selected);balance-=PANEL_COST;status.textContent=`Panel installed at cell ${selected+1}.`; }render();}
  $('[data-action="start"]').onclick=()=>{if(!syncSite())return;started=true;section('site').hidden=true;section('farm').hidden=false;section('results').hidden=false;scene?.start();last=performance.now();status.textContent='Click a grid cell to buy a panel. Use arrow keys and Enter for keyboard placement.';render();};
  $('[data-action="change"]').onclick=()=>{window.location.reload();};
  $('[data-action="play"]').onclick=e=>{playing=!playing;last=performance.now();e.currentTarget.textContent=playing?'Pause simulation':'Play simulation';};
  for(const k of ['tilt','azimuth','shading'])input(k).oninput=render;
  input('hour').oninput=()=>{hour=+input('hour').value;render();};input('date').onchange=()=>{if(input('date').value){siteDate=input('date').value;render();}};
  $('.sp-scene').onkeydown=e=>{if(!started)return;if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter',' '].includes(e.key))e.preventDefault();if(e.key==='ArrowLeft')selected=Math.max(0,selected-1);if(e.key==='ArrowRight')selected=Math.min(99,selected+1);if(e.key==='ArrowUp')selected=Math.max(0,selected-10);if(e.key==='ArrowDown')selected=Math.min(99,selected+10);if(e.key==='Enter'||e.key===' ')toggle();else {status.textContent=`Selected cell ${selected+1}. Press Enter to add or remove a panel.`;render();}};
  const timer=setInterval(()=>{const now=performance.now(),dt=Math.min((now-last)/1000,1);last=now;if(!started||!playing||document.hidden)return;
    const result=calculate(site,date(),hour,settings(),weather());const hours=dt*24/60;const generated=result.current.ac/1000*hours;energy+=generated;balance+=generated*PRICE;hour+=hours;
    if(hour>=24){hour-=24;const d=date();d.setUTCDate(d.getUTCDate()+1);siteDate=d.toISOString().slice(0,10);input('date').value=siteDate;variation=Math.max(-.12,Math.min(.12,.65*variation+(Math.random()-.5)*.08));}input('hour').value=hour;render();},250);
  return {destroy(){clearInterval(timer);scene?.destroy();root.remove();style.remove();}};
}
const host=document.getElementById('project');if(host)mountSolarProject(host);
