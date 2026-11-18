import { updateMotionControl } from '../motion-control.js';
import shared from '../hydro-lab/styles.css';
import layout from '../solar-lab/styles.css';
import styles from './styles.css';
import { operatingPoint, powerModel, annualEnergy, circularMean, YEAR_H } from './model.js';
import { mountWindScene } from './scene.js';

const groups = [
  ['Wind and turbine', [
    ['speed', 'Wind speed at input height', 0, 35, .1, 8, 'm/s'],
    ['direction', 'Wind from (clockwise from north)', 0, 360, 1, 12, '°'],
    ['radius', 'Blade radius R', 15, 100, 1, 58, 'm'],
    ['hub', 'Hub height z', 35, 180, 1, 90, 'm'],
  ]],
  ['Air density and height correction', [
    ['pressure', 'Air pressure p', 75, 104, .1, 101.3, 'kPa'],
    ['temperature', 'Air temperature T', -20, 45, 1, 15, '°C'],
    ['reference', 'Measurement height z<sub>r</sub>', 10, 120, 1, 10, 'm'],
    ['alpha', 'Shear exponent α', 0, .45, .01, .14, ''],
    ['roughness', 'Roughness length z<sub>0</sub>', .01, 1, .01, .05, 'm'],
    ['displacement', 'Displacement height d', 0, 8, .1, 0, 'm'],
  ]],
  ['Rotor, drivetrain and power curve', [
    ['cp', 'Power coefficient C<sub>p</sub>', .05, .59, .01, .42, ''],
    ['efficiency', 'Drivetrain efficiency η<sub>drive</sub>', 60, 100, 1, 95, '%'],
    ['tsr', 'Target tip-speed ratio below rated', 3, 10, .1, 7, ''],
    ['ratedMW', 'Rated electrical power P<sub>r</sub>', .1, 12, .1, 3, 'MW'],
    ['cutIn', 'Cut-in speed v<sub>in</sub>', 1, 6, .1, 3, 'm/s'],
    ['ratedSpeed', 'Rated speed v<sub>r</sub> (course curve)', 7, 18, .1, 12, 'm/s'],
    ['cutOut', 'Cut-out speed v<sub>out</sub>', 19, 35, .1, 25, 'm/s'],
    ['hours', 'Time at current wind speed Δt', .25, 24, .25, 2, 'h'],
  ]],
  ['Annual energy and losses', [
    ['shape', 'Weibull shape k', 1, 4, .1, 2, ''],
    ['scale', 'Weibull scale c at hub height', 3, 18, .1, 8.5, 'm/s'],
    ['availability', 'Availability', 50, 100, 1, 97, '%'],
    ['wake', 'Wake loss', 0, 35, 1, 5, '%'],
    ['loss', 'Electrical and other site losses', 0, 35, 1, 3, '%'],
  ]],
  ['Wind measurement examples', [
    ['sigma', 'Wind-speed standard deviation σ<sub>v</sub>', 0, 10, .1, 1.2, 'm/s'],
    ['bearingA', 'First direction observation', 0, 360, 1, 350, '°'],
    ['bearingB', 'Second direction observation', 0, 360, 1, 10, '°'],
  ]],
];
const fields = groups.flatMap(([, sliders]) => sliders);
const defaults = Object.fromEntries(fields.map(([key,,,,,initial]) => [key, initial]));
const fixed = (value, digits = 2) => value === null ? 'undefined' : value.toFixed(digits);

function sliders(items) {
  return items.map(([key,label,min,max,step,initial]) => `<div class="control">
    <div class="control-line"><label for="${key}">${label}</label><output id="${key}-value" aria-hidden="true"></output></div>
    <input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${initial}">
  </div>`).join('');
}

function drawCurve(svg, input, point) {
  const powerAt = powerModel(input);
  // Maximum hub-height speed across the wind and height-correction controls is below 270 m/s.
  const maxSpeed = 270;
  const x = speed => 53 + 460 * speed / maxSpeed;
  const y = power => 160 - 132 * power / 12;
  const cutInRight = input.outputMode === 'aero' ? powerAt(input.cutIn + 1e-7) : 0;
  const points = [[input.cutIn, cutInRight]];
  for (let i = 1; i <= 120; i++) {
    const speed = input.cutIn + (input.cutOut - input.cutIn) * i / 120;
    points.push([speed, powerAt(Math.min(speed, input.cutOut - 1e-7))]);
  }
  const cutoff = x(input.cutOut);
  const leftLimit = y(points.at(-1)[1]);
  const ticks = [[input.cutIn, 'Cut-in'], [input.cutOut, 'Cut-out']];
  if (input.outputMode === 'curve') ticks.splice(1, 0, [input.ratedSpeed, 'Rated']);
  svg.innerHTML = `<path class="axis" fill="none" d="M53 16V160H513"/>
    <text x="4" y="32">12 MW</text>
    <text x="16" y="164">0 MW</text><text x="53" y="190">0</text><text x="513" y="190" text-anchor="end">270</text><text x="514" y="180" text-anchor="end">Hub-height wind (m/s)</text>
    ${ticks.map(([speed,label]) => `<path stroke="#91a6a5" stroke-dasharray="4 4" d="M${x(speed)} 16V160"/>
      <text x="${x(speed)}" y="176" text-anchor="middle">${label}</text>`).join('')}
    <path stroke="#087660" stroke-width="3" d="M${x(0)} ${y(0)}H${x(input.cutIn)}"/>
    <polyline fill="none" stroke="#087660" stroke-width="3" points="${points.map(([speed,power]) => `${x(speed)},${y(power)}`).join(' ')}"/>
    ${cutInRight > 0 ? `<circle cx="${x(input.cutIn)}" cy="${y(cutInRight)}" r="4" fill="#fff" stroke="#087660" stroke-width="1.5"/>
      <circle cx="${x(input.cutIn)}" cy="${y(0)}" r="4" fill="#087660"/>` : ''}
    <path stroke="#087660" stroke-width="3" d="M${cutoff} ${y(0)}H${x(maxSpeed)}"/>
    <circle cx="${cutoff}" cy="${leftLimit}" r="4" fill="#fff" stroke="#087660" stroke-width="1.5"/>
    <circle cx="${cutoff}" cy="${y(0)}" r="4" fill="#087660"/>
    <circle fill="#c77616" stroke="#fff" stroke-width="2" cx="${x(point.speed)}" cy="${y(point.gridMW)}" r="6"/>`;
}

export function mountWindLab(container) {
  const root = container.shadowRoot ?? container.attachShadow({ mode: 'open' });
  const stylesheet = document.createElement('style');
  stylesheet.textContent = shared + layout + styles;
  const content = document.createElement('div');
  content.innerHTML = `<article class="hydro-lab solar-lab wind-lab" aria-labelledby="wind-title">
    <header class="lab-head"><span class="lab-kicker">Interactive engineering diagram</span>
      <h2 id="wind-title">From moving air to grid electricity</h2>
      <p>Change the wind and follow energy through the rotor, generator, transformer and transmission line.</p></header>
    <div class="solar-control-panel">
      <fieldset><legend>Wind and turbine</legend><div class="solar-controls">${sliders(groups[0][1])}</div></fieldset>
      <div class="wind-selects"><label>Wind speed input
        <select id="heightMode"><option value="hub">At hub height</option><option value="power">At measurement height · power law</option>
        <option value="log">At measurement height · neutral log law</option></select></label>
        <label>Electrical output model
        <select id="outputMode"><option value="curve">Course power curve</option><option value="aero">Fixed Cₚ with equipment limits</option></select></label></div>
      ${groups.slice(1,4).map(([name,items]) => `<details><summary>${name}</summary><div class="solar-controls">${sliders(items)}</div></details>`).join('')}
    </div>
    <div class="solar-stage">
      <figure class="engineering-card">
        <div id="wind-scene" aria-label="Isometric wind turbine cutaway, step-up transformer and transmission pylon">
          <div class="plant-tooltip" id="wind-tooltip" hidden></div>
          <button class="motion-toggle" id="wind-play" type="button"></button>
        </div>
        <figcaption>Drag to orbit. Hover or tap a part to identify it. Blue streaks show wind direction. Gold pulses trace the electrical route to the grid. Geometry and pulse speed are schematic.</figcaption>
      </figure>
      <div class="wind-output-stack">
        <section class="metrics-card" aria-label="Wind turbine output">
          <div class="power-card"><span class="eyebrow">Electrical power to grid</span><strong id="grid"></strong><span class="status" id="region"></span></div>
          <div class="metric-list" id="wind-metrics"></div>
        </section>
        <section class="chart-card"><h3>Electrical power curve</h3><p>The orange dot marks the current hub-height wind speed. At cut-out, the filled point marks zero output.</p>
          <svg id="wind-chart" viewBox="0 0 535 192" role="img" aria-label="Electrical power versus hub-height wind speed"></svg></section>
      </div>
    </div>
    <div class="solar-analysis">
      <details open><summary>Live equations</summary><div id="wind-equations"></div></details>
      <details><summary>${groups[4][0]}</summary><div class="solar-controls">${sliders(groups[4][1])}</div><div id="wind-measurements"></div></details>
      <p class="assumptions">Air density, blade radius, and drivetrain efficiency change electrical output in the fixed-Cₚ model. The course curve uses the specified electrical rating and speed thresholds. Annual energy uses a separate hub-height Weibull distribution and applies availability and site losses once. The rotor stops at cut-out in this teaching model.</p>
    </div>
  </article>`;
  root.replaceChildren(stylesheet, content);
  const get = id => root.getElementById(id);
  const input = { ...defaults, heightMode: 'hub', outputMode: 'curve' };
  let scene;
  let animate = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  try { scene = mountWindScene(get('wind-scene'), get('wind-tooltip')); }
  catch {
    get('wind-scene').textContent = 'The 3D view requires WebGL. The equations, controls and power curve are available below.';
  }
  const eq = value => `<div class="equation">${value}</div>`;
  function render() {
    const p = operatingPoint(input);
    const year = annualEnergy(input);
    const mean = circularMean(input.bearingA, input.bearingB);
    get('grid').textContent = fixed(p.gridMW, 3) + ' MW';
    const badge = get('region');
    badge.textContent = p.region;
    badge.className = `status ${p.gridMW ? 'open' : 'clipped'}`;
    get('wind-metrics').innerHTML = [
      ['Hub-height wind', `${fixed(p.speed, 1)} m/s`], ['Rotor speed', `${fixed(p.rpm, 1)} rpm`],
      [`Energy over ${input.hours} h`, `${fixed(input.hours * p.gridMW, 3)} MWh`],
      ['Net annual energy (Weibull)', `${fixed(year.net, 0)} MWh`],
      ['Net annual capacity factor', `${fixed(year.capacityFactor * 100, 1)}%`],
    ].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join('');
    const heightEquation = input.heightMode === 'power' ?
      `v = v<sub>ref</sub>(z/z<sub>r</sub>)<sup>α</sup> = ${input.speed} × (${input.hub}/${input.reference})<sup>${input.alpha}</sup>` :
      input.heightMode === 'log' ? `u<sub>*</sub> = κv<sub>ref</sub>/ln[(z<sub>r</sub>−d)/z<sub>0</sub>] = ${fixed(p.frictionVelocity,3)} m/s · v = (u<sub>*</sub>/κ) ln[(z−d)/z<sub>0</sub>], with κ = 0.4` :
      `v = entered hub-height wind speed`;
    const outputEquation = input.outputMode === 'aero' ?
      'P<sub>grid</sub> = min(P<sub>e</sub>, P<sub>r</sub>) during operation, zero at/below cut-in and at/above cut-out' :
      'P<sub>grid</sub> = P<sub>r</sub>(v³−v<sub>in</sub>³)/(v<sub>r</sub>³−v<sub>in</sub>³) on the ramp, P<sub>r</sub> on the plateau, zero at/below cut-in and at/above cut-out';
    get('wind-equations').innerHTML = [
      `${heightEquation} → hub wind ${fixed(p.speed, 2)} m/s`,
      `ρ = p/(R<sub>d</sub>T) = ${fixed(input.pressure*1000,0)}/[287.05 × ${fixed(input.temperature+273.15)}] ≈ ${fixed(p.density,3)} kg/m³`,
      `A = πR² = ${fixed(p.area,1)} m² · ṁ = ρAv = ${fixed(p.massFlow,0)} kg/s · P<sub>wind</sub> = ½ρAv³ = ${fixed(p.windMW,3)} MW`,
      `Fixed-coefficient estimate: P<sub>rotor</sub> = C<sub>p</sub>P<sub>wind</sub> = ${fixed(p.rotorMW,3)} MW · C<sub>p,max</sub> = 16/27 · P<sub>e</sub> = η<sub>drive</sub>P<sub>rotor</sub> = ${fixed(p.aeroMW,3)} MW before equipment limits.`,
      `${outputEquation}. Current electrical output: ${fixed(p.gridMW,3)} MW.`,
      `Ω = 2πn/60 = ${fixed(p.omega,3)} rad/s · λ = ΩR/v = ${fixed(p.tsr)} · τ = P<sub>shaft</sub>/Ω = ${p.torque === null ? 'undefined when stopped' : fixed(p.torque/1000,1) + ' kN·m'} (shaft power estimated from grid output and drivetrain efficiency)`,
      `E = Σ P<sub>i</sub>Δt<sub>i</sub> = ${fixed(p.gridMW,3)} × ${input.hours} ≈ ${fixed(input.hours*p.gridMW,3)} MWh for the current interval`,
      `Weibull F(v) = 1 − exp[−(v/c)<sup>k</sup>], f(v) = (k/c)(v/c)<sup>k−1</sup> exp[−(v/c)<sup>k</sup>]. Bin probability = F(b) − F(a). Mean power ≈ ΣP(v<sub>mid</sub>)p<sub>i</sub> = ${fixed(year.meanMW,3)} MW.`,
      `AEP<sub>gross</sub> = ${YEAR_H} × mean power = ${fixed(year.gross,0)} MWh · AEP<sub>net</sub> = AEP<sub>gross</sub> × availability × (1 − wake loss) × (1 − site loss) = ${fixed(year.net,0)} MWh`,
      `CF<sub>net</sub> = AEP<sub>net</sub>/(P<sub>r</sub> × ${YEAR_H} h) = ${fixed(year.capacityFactor*100,1)}%`,
    ].map(eq).join('');
    get('wind-measurements').innerHTML = [
      `TI = σ<sub>v</sub>/v̄ = ${p.speed > 0 ? fixed(input.sigma/p.speed*100,1) + '%' : 'undefined at zero mean speed'} using hub-height input as the example mean.`,
      `Circular mean = atan2(mean sin θ, mean cos θ) = ${mean.degrees === null ? 'undefined for opposing directions' : fixed(mean.degrees,1) + '°'} · resultant length = ${fixed(mean.length,3)}.`,
    ].map(eq).join('');
    drawCurve(get('wind-chart'), input, p);
    scene?.update(input, p, animate);
  }
  for (const [key,,,,,,unit] of fields) {
    const slider = get(key);
    const update = () => {
      input[key] = Number(slider.value);
      const text = `${slider.value} ${unit}`.trim();
      get(key+'-value').textContent = text;
      slider.setAttribute('aria-valuetext', text);
    };
    update();
    slider.oninput = () => { update(); render(); };
  }
  for (const key of ['heightMode','outputMode']) get(key).onchange = event => { input[key] = event.target.value; render(); };
  const toggle = get('wind-play');
  if (toggle) {
    const label = () => { updateMotionControl(toggle, animate); };
    label();
    toggle.onclick = () => { animate = !animate; label(); render(); };
  }
  render();
  return { destroy() { scene?.destroy(); root.replaceChildren(); } };
}

const host = document.getElementById('wind-lab-root');
if (host) mountWindLab(host);
