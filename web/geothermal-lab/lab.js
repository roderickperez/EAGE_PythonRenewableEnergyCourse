import { updateMotionControl } from '../motion-control.js';
import shared from '../hydro-lab/styles.css';
import layout from '../solar-lab/styles.css';
import styles from './styles.css';
import { defaults, geothermal } from './model.js';
import { mountGeothermalScene } from './scene.js';

const groups = [
  ['Reservoir and wells', [
    ['reservoir','Reservoir temperature T<sub>res</sub>',65,220,1,'°C'],
    ['reinjection','Target reinjection temperature T<sub>r</sub>',15,180,1,'°C'],
    ['flow','Brine mass flow ṁ',0,150,1,'kg/s'],
    ['wellLoss','Temperature drop in production well',0,40,1,'K'],
  ]],
  ['Heat engine and internal loads', [
    ['sink','Cooling sink temperature T<sub>c</sub>',5,45,1,'°C'],
    ['efficiency','Requested gross conversion efficiency η',0,30,.5,'%'],
    ['cp','Brine specific heat c<sub>p</sub>',3000,4600,10,'J/(kg·K)'],
    ['parasitic','Internal loads as a share of gross output',0,70,1,'%'],
    ['pressure','Pump pressure rise Δp',0,15,.1,'MPa'],
    ['density','Brine density ρ<sub>f</sub>',800,1200,5,'kg/m³'],
    ['pumpEfficiency','Pump efficiency η<sub>pump</sub>',30,95,1,'%'],
    ['auxiliaries','Other plant loads when circulating',0,4,.05,'MW'],
  ]],
  ['Energy and production decline', [
    ['rating','Net export rating P<sub>r,net</sub>',.1,20,.1,'MW'],
    ['hours','Time at current conditions Δt',.25,48,.25,'h'],
    ['availability','Annual availability a',0,100,1,'%'],
    ['decline','Annual net-power decline d',0,20,.1,'%'],
    ['years','Project duration',1,40,1,'years'],
  ]],
  ['Geology and stored heat', [
    ['surface','Surface and inventory reference temperature T<sub>0</sub>',0,35,1,'°C'],
    ['depth','Depth z',.5,6,.1,'km'],
    ['gradient','Temperature gradient G<sub>T</sub>',10,75,1,'K/km'],
    ['conductivity','Rock thermal conductivity k<sub>T</sub>',.5,5,.1,'W/(m·K)'],
    ['permeability','Permeability k<sub>perm</sub>',1,300,1,'mD'],
    ['viscosity','Fluid viscosity μ',.1,2,.1,'mPa·s'],
    ['pressureGradient','Pressure-gradient magnitude',0,30,1,'kPa/m'],
    ['volume','Bulk reservoir volume V',.5,30,.5,'million m³'],
    ['porosity','Reservoir porosity φ',0,35,1,'%'],
    ['rockDensity','Rock density ρ<sub>r</sub>',2000,3200,10,'kg/m³'],
    ['rockCp','Rock specific heat c<sub>r</sub>',600,1400,10,'J/(kg·K)'],
  ]],
  ['Direct heat and heat-pump comparison', [
    ['heatPumpPower','Electrical input to heat pump',0,5,.05,'MW'],
    ['cop','Heating coefficient of performance',1,6,.1,''],
  ]],
];
const fields = groups.flatMap(([, values]) => values);
const f = (value, digits = 2) => value.toFixed(digits);

function controls(items) {
  return items.map(([key,label,min,max,step])=>`<div class="control">
    <div class="control-line"><label for="${key}">${label}</label><output id="${key}-value" aria-hidden="true"></output></div>
    <input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${defaults[key]}">
  </div>`).join('');
}

function chart(svg, annual) {
  const max = Math.max(1, annual[0] / 1000);
  const stride = 455 / annual.length;
  const bars = annual.map((mwh,i) => {
    const height = mwh / 1000 / max * 115;
    return `<rect x="${54 + i*stride}" y="${143-height}" width="${stride*.76}" height="${height}" rx="2" fill="#087660"/>`;
  }).join('');
  svg.innerHTML = `<path class="axis" fill="none" d="M53 22V143H510"/><text x="5" y="30">${f(max,1)} GWh</text>
    <text x="20" y="143">0</text><text x="56" y="163">Year 1</text>
    ${annual.length > 1 ? `<text x="510" y="163" text-anchor="end">Year ${annual.length}</text>` : ''}${bars}`;
}

export function mountGeothermalLab(container) {
  const root = container.shadowRoot ?? container.attachShadow({ mode:'open' });
  const sheet = document.createElement('style');
  sheet.textContent = shared + layout + styles;
  const body = document.createElement('div');
  body.innerHTML = `<article class="hydro-lab solar-lab geothermal-lab" aria-labelledby="geo-title">
    <header class="lab-head"><span class="lab-kicker">Interactive engineering diagram</span>
      <h2 id="geo-title">From underground heat to grid electricity</h2>
      <p>Adjust the reservoir and follow the heat through a binary-cycle geothermal plant.</p>
    </header>
    <div class="solar-control-panel">
      <fieldset><legend>Reservoir and wells</legend><div class="solar-controls">${controls(groups[0][1])}</div></fieldset>
      ${groups.slice(1,3).map(([name,items],i)=>`<details><summary>${name}</summary>
        ${i===0 ? '<label class="geo-load">Internal load model <select id="loadMode"><option value="fraction">Fraction of gross output</option><option value="explicit">Hydraulic pump and auxiliary loads</option></select></label>' : ''}
        <div class="solar-controls">${controls(items)}</div></details>`).join('')}
    </div>
    <div class="solar-stage">
      <figure class="engineering-card">
        <div id="geo-scene" aria-label="Isometric geothermal reservoir, wells, binary-cycle plant and grid connection">
          <div class="plant-tooltip" id="geo-tooltip" hidden></div>
          <button class="motion-toggle" id="geo-play" type="button"></button>
        </div>
        <figcaption>Drag to orbit. Hover or tap to identify components.</figcaption>
        <ul class="geo-legend"><li><i class="hot"></i>Hot production brine</li><li><i class="cool"></i>Cooled return brine</li>
          <li><i class="vapor"></i>Working-fluid vapor</li><li><i class="liquid"></i>Condensed working fluid</li>
          <li><i class="power"></i>Electrical output</li></ul>
      </figure>
      <section class="metrics-card" aria-label="Geothermal plant output">
        <div class="power-card"><span class="eyebrow">Net electrical export</span><strong id="geo-power"></strong>
          <span class="status" id="geo-status"></span></div><div class="metric-list" id="geo-metrics"></div>
      </section>
    </div>
    <div class="solar-analysis">
      <section class="chart-card"><h3>Annual electricity over the project lifetime</h3>
        <p>Annual generation after availability and the selected net-power decline.</p>
        <svg id="geo-chart" viewBox="0 0 530 177" role="img" aria-label="Annual net exported electricity in gigawatt-hours"></svg></section>
      <details open><summary>Live power and energy equations</summary><div id="geo-equations"></div></details>
      ${groups.slice(3).map(([name,items],i)=>`<details><summary>${name}</summary><div class="solar-controls">${controls(items)}</div><div id="geo-extra-${i}"></div></details>`).join('')}
      <p class="assumptions">The model assumes pressurized liquid brine, constant fluid properties and a fixed cooling sink. Return temperature is bounded by the sink and production temperatures when useful heat is available. The ground section is schematic. Depth, heat flux and stored heat are independent geology estimates. Annual decline is a scenario with assumed operating hours.</p>
    </div>
  </article>`;
  root.replaceChildren(sheet,body);
  const input = { ...defaults };
  const get = id => root.getElementById(id);
  let animate = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let scene;
  try { scene=mountGeothermalScene(get('geo-scene'),get('geo-tooltip')); }
  catch { get('geo-scene').textContent='The 3D view requires WebGL. Controls, equations and charts are available below.'; }
  const eq = value => `<div class="equation">${value}</div>`;
  function render() {
    const result = geothermal(input);
    const p = result;
    get('geo-power').textContent = f(p.exported,3) + ' MW';
    const status = get('geo-status');
    status.className = `status ${p.exported && !p.curtailed ? 'open' : 'clipped'}`;
    status.textContent = p.imported ? `Importing ${f(p.imported,3)} MW for plant loads`
      : p.curtailed ? 'Limited by net export rating'
      : p.exported ? 'Exporting to grid' : 'No net export';
    get('geo-metrics').innerHTML = [
      [`Electricity over ${input.hours} h`,f(p.interval,3)+' MWh'],
      ['Heat recovered from brine',f(p.thermal,2)+' MW thermal'],
      ['Gross electrical power',f(p.gross,3)+' MW'],
      ['Production / reinjection',`${f(p.production,0)} / ${f(p.reinjection,0)} °C`],
      ['Year-one electricity',f(p.annual[0]/1000,2)+' GWh'],
      ['Lifetime electricity',f(p.lifetime/1000,2)+' GWh'],
    ].map(([label,value])=>`<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join('');
    const loads = input.loadMode === 'fraction'
      ? `P<sub>par</sub> = f<sub>par</sub>P<sub>gross</sub> = ${input.parasitic}% × ${f(p.gross,3)} ≈ ${f(p.parasitic,3)} MW`
      : `Q = ṁ/ρ<sub>f</sub> = ${f(p.volumeFlow,4)} m³/s · P<sub>pump</sub> = Δp Q/η<sub>pump</sub> = ${f(p.pump,3)} MW · P<sub>par</sub> = P<sub>pump</sub> + P<sub>aux</sub> = ${f(p.parasitic,3)} MW`;
    get('geo-equations').innerHTML = [
      `T<sub>p</sub> = T<sub>res</sub> − well temperature drop = ${input.reservoir} − ${input.wellLoss} = ${p.production} °C · ΔT = T<sub>p</sub> − T<sub>r</sub> = ${p.delta} K`,
      `h<sub>p</sub> − h<sub>r</sub> ≈ c<sub>p</sub>ΔT = ${f(p.enthalpy/1000)} kJ/kg · P<sub>th</sub> = ṁ(h<sub>p</sub> − h<sub>r</sub>) = ${f(p.thermal)} MW thermal`,
      `η<sub>C</sub> = 1 − T<sub>c</sub>/T<sub>p</sub> for a hot source above the sink = ${f(p.carnot*100,1)}%. Use Kelvin. The reversible bound for a cooling stream, 1 − T<sub>c</sub> ln(T<sub>p</sub>/T<sub>r</sub>)/ΔT, is ${f(p.streamLimit*100,1)}% with positive temperature drop.`,
      `P<sub>gross</sub> = η<sub>conv</sub>P<sub>th</sub> = ${f(p.conversion*100,1)}% × ${f(p.thermal)} ≈ ${f(p.gross,3)} MW${input.efficiency/100 > p.streamLimit ? ' (efficiency limited by the reversible stream bound)' : ''} · Heat rejected = P<sub>th</sub> − P<sub>gross</sub> = ${f(p.rejected)} MW thermal`,
      loads,
      `P<sub>net</sub> = P<sub>gross</sub> − P<sub>par</sub> = ${f(p.net,3)} MW · Grid export = min(P<sub>r,net</sub>, max(0,P<sub>net</sub>)) = ${f(p.exported,3)} MW`,
      `E<sub>export</sub> = ΣP<sub>export,i</sub>Δt<sub>i</sub> = ${f(p.exported,3)} × ${input.hours} ≈ ${f(p.interval,3)} MWh for the selected interval`,
      `P<sub>y</sub> = P<sub>export</sub>(1 − d)<sup>y−1</sup> · E<sub>y</sub> = P<sub>y</sub> × a × 8760 h · Year-one capacity factor = E<sub>1</sub>/(P<sub>r,net</sub> × 8760 h) = ${f(p.cf*100,1)}%`,
    ].map(eq).join('');
    get('geo-extra-0').innerHTML = [
      `T(z) = T<sub>0</sub> + G<sub>T</sub>z = ${f(p.depthTemp,1)} °C · |q| = k<sub>T</sub>G<sub>T</sub> = ${f(p.heatFlux,1)} mW/m² (Fourier conduction is toward lower temperature)`,
      `|q<sub>D</sub>| = (k<sub>perm</sub>/μ)|dp/dx| = ${p.darcy.toExponential(2)} m/s (Darcy flux)`,
      `E<sub>heat</sub> ≈ V[(1−φ)ρ<sub>r</sub>c<sub>r</sub> + φρ<sub>f</sub>c<sub>p</sub>](T<sub>res</sub> − T<sub>0</sub>) = ${f(p.stored,1)} GWh thermal in place`,
    ].map(eq).join('');
    get('geo-extra-1').innerHTML = eq(`For direct use, up to ${f(p.thermal)} MW thermal is recovered from the brine at these conditions. Delivery depends on the required temperature. In the separate heat-pump example, Q̇<sub>delivered</sub> = COP<sub>h</sub>P<sub>electric</sub> = ${f(input.cop*input.heatPumpPower)} MW thermal.`);
    for (const key of ['parasitic','pressure','pumpEfficiency','auxiliaries']) {
      get(key).disabled = (key==='parasitic') !== (input.loadMode==='fraction');
    }
    chart(get('geo-chart'),p.annual);
    scene?.update(input,p,animate);
  }
  for (const [key,,,,,unit] of fields) {
    const slider=get(key);
    const update=()=>{
      input[key]=Number(slider.value);
      const text=`${slider.value} ${unit}`.trim();
      get(key+'-value').textContent=text;
      slider.setAttribute('aria-valuetext',text);
    };
    update();
    slider.oninput=()=>{ update(); render(); };
  }
  get('loadMode').onchange=event=>{ input.loadMode=event.target.value; render(); };
  const toggle=get('geo-play');
  if(toggle){
    const label=()=>{ updateMotionControl(toggle, animate); };
    label();
    toggle.onclick=()=>{ animate=!animate; label(); render(); };
  }
  render();
  return { destroy(){ scene?.destroy(); root.replaceChildren(); } };
}

const host=document.getElementById('geothermal-lab-root');
if(host) mountGeothermalLab(host);
