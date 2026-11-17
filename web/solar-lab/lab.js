import { updateMotionControl } from '../motion-control.js';
import sharedStyles from '../hydro-lab/styles.css';
import styles from './styles.css';
import { RAD, STC, solarPower, solarDay, batteryDay, lifetime } from './model.js';
import { mountScene } from './scene.js';

const groups = [
  ['Sun and panel', [
    ['hour', 'Local solar time', 0, 24, .1, 10, 'h'],
    ['tilt', 'Panel tilt β', 0, 90, 1, 30, '°'],
    ['azimuth', 'Panel azimuth γ<sub>p</sub>', 0, 360, 1, 180, '°'],
    ['latitude', 'Latitude', -66, 66, 1, 35, '°'],
    ['declination', 'Solar declination', -23.4, 23.4, .1, 0, '°'],
  ]],
  ['Irradiance and temperature', [
    ['dni', 'Beam irradiance scale', 0, 1100, 10, 850, 'W/m²'],
    ['dhi', 'Diffuse irradiance scale', 0, 300, 5, 110, 'W/m²'],
    ['albedo', 'Ground reflectance ρ<sub>g</sub>', 0, 1, .01, .2, ''],
    ['ambient', 'Air temperature T<sub>a</sub>', -20, 55, 1, 25, '°C'],
    ['noct', 'Nominal cell temperature', 25, 65, 1, 45, '°C'],
    ['coefficient', 'Power temperature coefficient γ<sub>P</sub>', -.7, 0, .01, -.4, '%/°C'],
  ]],
  ['Module and inverter', [
    ['area', 'Module area A', .5, 4, .1, 2, 'm²'],
    ['efficiency', 'Module efficiency at STC', 5, 30, .5, 21, '%'],
    ['inverterEfficiency', 'Inverter efficiency η<sub>inv</sub>', 50, 100, 1, 97, '%'],
    ['rating', 'Inverter AC rating P<sub>AC,r</sub>', 50, 1200, 10, 350, 'W'],
    ['soiling', 'Soiling loss', 0, 35, 1, 0, '%'],
    ['wiring', 'Wiring loss', 0, 20, 1, 0, '%'],
    ['interval', 'Interval at current conditions Δt', .25, 6, .25, .5, 'h'],
  ]],
  ['I–V and fill factor example', [
    ['voc', 'Open-circuit voltage V<sub>oc</sub>', 15, 70, 1, 48, 'V'],
    ['isc', 'Short-circuit current I<sub>sc</sub>', 1, 20, .1, 11.5, 'A'],
    ['vmp', 'Voltage at maximum power / V<sub>oc</sub>', 45, 95, 1, 85, '%'],
    ['imp', 'Current at maximum power / I<sub>sc</sub>', 45, 99, 1, 90, '%'],
  ]],
  ['Solar thermal and storage example', [
    ['flow', 'Fluid mass flow', 0, .5, .01, .04, 'kg/s'],
    ['cp', 'Fluid specific heat', 1000, 4500, 10, 4180, 'J/(kg·K)'],
    ['rise', 'Collector outlet temperature rise', 0, 20, .1, 3, 'K'],
    ['mass', 'Tank mass', 25, 500, 5, 200, 'kg'],
    ['tankRise', 'Tank temperature rise', 0, 80, 1, 40, 'K'],
  ]],
  ['Battery and self-consumption example', [
    ['load', 'Constant electrical load', 0, 600, 10, 100, 'W'],
    ['storage', 'Usable battery capacity', 0, 3000, 50, 500, 'Wh'],
    ['batteryPower', 'Charge and discharge power limit', 0, 600, 10, 200, 'W'],
    ['batteryEfficiency', 'Efficiency per direction', 50, 100, 1, 90, '%'],
  ]],
  ['Degradation and lifetime cost example', [
    ['firstYear', 'First-year AC energy (independent estimate)', 50, 1500, 10, 500, 'kWh'],
    ['degradation', 'Annual degradation', 0, 5, .1, .5, '%'],
    ['years', 'Operating life', 1, 40, 1, 25, 'years'],
    ['capital', 'Installed cost', 0, 5000, 25, 500, '€'],
    ['maintenance', 'Annual operating cost', 0, 250, 5, 10, '€'],
    ['discount', 'Real discount rate', 0, 15, .1, 3, '%'],
  ]],
  ['Availability and performance checks', [
    ['available', 'Equipment available time', 0, 24, .25, 22, 'h'],
    ['calibration', 'Training observed / expected energy', .5, 1.5, .01, .9, ''],
    ['observed', 'Observed energy for this day', 0, 30, .01, 2, 'kWh'],
    ['threshold', 'Underperformance threshold', 50, 100, 1, 80, '%'],
  ]],
];
const definitions = groups.flatMap(([, fields]) => fields);
const dailyKeys = new Set(groups.slice(0, 3).flatMap(([, fields]) => fields.map(([key]) => key)));
export const defaults = Object.fromEntries(definitions.map(([key,,,,,value]) => [key, value]));
const fixed = (value, digits = 2) => value.toFixed(digits);

function controls(fields) {
  return fields.map(([key, label, min, max, step, value, unit]) =>
    `<div class="control"><div class="control-line"><label for="${key}">${label}</label>
    <output id="${key}-value" for="${key}" aria-hidden="true"></output></div>
    <input type="range" id="${key}" min="${min}" max="${max}" step="${step}" value="${value}"></div>`).join('');
}

function chart(svg, daily, hour, rating) {
  const width = 540, left = 53, right = 520, top = 16, bottom = 162;
  const max = Math.max(100, rating, ...daily.samples.map(sample => sample.dc));
  const x = time => left + (right - left) * time / 24;
  const y = power => bottom - (bottom - top) * power / max;
  const trace = key => daily.samples.map((sample, i) => `${x((i + .5) / 12)},${y(sample[key])}`).join(' ');
  const ticks = [0, 6, 12, 18, 24].map(h => `<text x="${x(h)}" y="185" text-anchor="middle">${h}:00</text>`).join('');
  svg.setAttribute('viewBox', `0 0 ${width} 199`);
  svg.innerHTML = `<path class="axis" fill="none" d="M${left} ${top}V${bottom}H${right}"/>
    <text x="6" y="22">${fixed(max, 0)} W</text><text x="15" y="165">0 W</text>
    <polyline fill="none" stroke="#6688a0" stroke-width="2" stroke-dasharray="5 4" points="${trace('dc')}"/>
    <polyline fill="none" stroke="#087660" stroke-width="3" points="${trace('ac')}"/>
    <path stroke="#9a5f16" stroke-width="2" d="M${x(hour)} ${top}V${bottom}"/>${ticks}`;
}

export function mountSolarLab(container) {
  const root = container.shadowRoot ?? container.attachShadow({ mode: 'open' });
  const stylesheet = document.createElement('style');
  stylesheet.textContent = sharedStyles + styles;
  const body = document.createElement('div');
  body.innerHTML = `<article class="hydro-lab solar-lab" aria-labelledby="solar-title">
    <header class="lab-head"><span class="lab-kicker">Interactive engineering diagram</span>
      <h2 id="solar-title">From sunlight to electrical energy</h2>
      <p>Move through a solar day, aim the panel, and see how much electricity one module delivers.</p>
    </header>
    <div class="solar-control-panel">
      <fieldset><legend>Sun and panel</legend><div class="solar-controls">${controls(groups[0][1])}</div></fieldset>
      ${groups.slice(1, 3).map(([name, fields]) => `<details><summary>${name}</summary><div class="solar-controls">${controls(fields)}</div></details>`).join('')}
    </div>
    <div class="solar-stage">
      <figure class="engineering-card"><div id="solar-scene" aria-label="Isometric solar module, sun path, inverter and AC output"></div>
        <button type="button" id="solar-play" class="motion-toggle"></button>
        <figcaption>Drag to orbit. The dashed arc shows the daylight path. Gold rays mark direct light on the panel and gold pulses trace electrical output through the inverter.</figcaption>
      </figure>
      <section class="metrics-card" aria-label="Module output">
        <div class="power-card"><span class="eyebrow">Estimated AC energy in one day</span><strong id="daily"></strong>
          <span id="status" class="status"></span></div>
        <div class="metric-list" id="metrics"></div>
      </section>
    </div>
    <div class="solar-analysis">
      <section class="chart-card"><h3>Power over the solar day</h3><p>Solid green is delivered AC, dashed blue is module DC, and the vertical marker indicates the selected time.</p>
        <svg id="solar-chart" role="img" aria-label="Daily DC and AC power in watts"></svg></section>
      <details open><summary>Live PV equations</summary><div id="equations"></div></details>
      ${groups.slice(3).map(([name, fields], i) => `<details><summary>${name}</summary>
        <div class="solar-controls">${controls(fields)}</div><div id="extension-${i}"></div></details>`).join('')}
      <p class="assumptions">Synthetic day at the selected latitude and solar declination. Azimuth is clockwise from north (east 90°, south 180°). Irradiance follows the sun above the horizon. Daily integration uses five-minute intervals and fixed weather and equipment settings. The selected interval assumes steady power at the current slider position.</p>
    </div>
  </article>`;
  root.replaceChildren(stylesheet, body);
  const get = key => root.getElementById(key);
  const input = { ...defaults };
  let scene;
  try { scene = mountScene(get('solar-scene')); }
  catch { get('solar-scene').textContent = '3D view unavailable. The sliders, equations and power chart still work.'; }
  let daily = solarDay(input);
  let timer;
  const eq = line => `<div class="equation">${line}</div>`;
  function render() {
    const p = solarPower(input);
    const hour = Math.round(input.hour * 60);
    get('hour-value').textContent = `${String(Math.floor(hour / 60)).padStart(2, '0')}:${String(hour % 60).padStart(2, '0')}`;
    get('hour').setAttribute('aria-valuetext', get('hour-value').textContent + ' local solar time');
    get('daily').textContent = `${fixed(daily.energy, 3)} kWh`;
    const status = get('status');
    status.className = `status ${p.clip > 0 ? 'clipped' : 'open'}`;
    status.textContent = p.elevation <= 0 ? 'Sun below horizon' : p.clip > 0 ? 'Inverter clipping' : 'Within inverter rating';
    get('metrics').innerHTML = [
      ['AC power now', `${fixed(p.ac, 1)} W`],
      ['Module DC power now', `${fixed(p.dc, 1)} W`],
      ['Energy at current conditions', `${fixed(p.ac * input.interval / 1000, 3)} kWh`],
      ['On-panel irradiance', `${fixed(p.poa, 1)} W/m²`],
      ['Sun elevation / cell temperature', `${fixed(p.elevation, 1)}° / ${fixed(p.cell, 1)} °C`],
      ['Daily clipping loss', `${fixed(daily.clipping, 3)} kWh`],
    ].map(([label,value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join('');
    const az = (Math.atan2(p.sun[0], p.sun[2]) / RAD + 360) % 360;
    get('equations').innerHTML = [
      `θ<sub>z</sub> = ${fixed(90 - p.elevation, 1)}° · γ<sub>s</sub> = ${fixed(az, 1)}° · cos θ<sub>i</sub> = cos θ<sub>z</sub> cos β + sin θ<sub>z</sub> sin β cos(γ<sub>s</sub> − γ<sub>p</sub>) = ${fixed(p.incidence, 3)}`,
      `GHI = DNI × max(0, cos θ<sub>z</sub>) + DHI = ${fixed(p.dni, 1)} × ${fixed(Math.max(0, p.sun[1]), 3)} + ${fixed(p.dhi, 1)} = ${fixed(p.ghi, 1)} W/m²`,
      `G<sub>POA</sub> = DNI × max(cos θ<sub>i</sub>, 0) + DHI × (1 + cos β)/2 + ρ<sub>g</sub> × GHI × (1 − cos β)/2 = ${fixed(p.beam, 1)} + ${fixed(p.sky, 1)} + ${fixed(p.ground, 1)} ≈ ${fixed(p.poa, 1)} W/m²`,
      `P<sub>STC</sub> = G<sub>STC</sub> × A × η<sub>STC</sub> = ${STC} × ${input.area} × ${input.efficiency / 100} = ${fixed(p.rated, 1)} W · P<sub>DC,constant η</sub> = G<sub>POA</sub> × A × η<sub>STC</sub> = ${fixed(p.areaPower, 1)} W`,
      `T<sub>c</sub> = T<sub>a</sub> + (G<sub>POA</sub>/800)(NOCT − 20) = ${fixed(p.cell, 1)} °C`,
      `P<sub>DC</sub> = P<sub>STC</sub> × (G<sub>POA</sub>/G<sub>STC</sub>) × max[0, 1 + γ<sub>P</sub>(T<sub>c</sub> − 25)] = ${fixed(p.areaPower, 1)} × ${fixed(p.tempFactor, 3)} ≈ ${fixed(p.dc, 1)} W · η<sub>PV</sub> = P<sub>DC</sub>/(G<sub>POA</sub>A) = ${p.poa > 0 ? fixed(100 * p.dc / (p.poa * input.area), 1) + '%' : 'undefined in darkness'}`,
      `Retained fraction = ∏(1 − ℓ<sub>j</sub>) = ${fixed(p.retained, 3)} · P<sub>AC,pot</sub> = η<sub>inv</sub> × P<sub>DC</sub> × retained fraction = ${fixed(p.potential, 1)} W · inverter conversion loss = ${fixed(p.conversionLoss, 1)} W`,
      `P<sub>AC</sub> = min(P<sub>AC,pot</sub>, P<sub>AC,r</sub>) = ${fixed(p.ac, 1)} W · P<sub>clip</sub> = max(P<sub>AC,pot</sub> − P<sub>AC,r</sub>, 0) = ${fixed(p.clip, 1)} W · DC/AC rating ratio = ${fixed(p.rated / input.rating)}`,
      `E<sub>interval</sub> = P<sub>AC</sub> × Δt / 1000 = ${fixed(p.ac, 1)} × ${input.interval} / 1000 ≈ ${fixed(p.ac * input.interval / 1000, 3)} kWh`,
      `E<sub>AC,day</sub> = Σ P<sub>AC,i</sub> × Δt<sub>i</sub> / 1000 = ${fixed(daily.energy, 3)} kWh · H<sub>POA</sub> = Σ G<sub>POA,i</sub> × Δt<sub>i</sub> / 1000 = ${fixed(daily.irradiation, 3)} kWh/m²`,
      `Y<sub>f</sub> = E<sub>AC</sub>/P<sub>STC</sub> = ${fixed(daily.yieldFinal)} kWh/kWp · Y<sub>r</sub> = H<sub>POA</sub>/G<sub>STC</sub> = ${fixed(daily.yieldRef)} h · PR = Y<sub>f</sub>/Y<sub>r</sub> = ${daily.pr === null ? 'undefined' : fixed(daily.pr, 3)} · AC capacity factor = E<sub>AC</sub>/(P<sub>AC,r</sub> × 24 h) = ${fixed(daily.cf * 100, 1)}% (ratings in kW)`,
    ].map(eq).join('');
    const vmp = input.voc * input.vmp / 100, imp = input.isc * input.imp / 100;
    get('extension-0').innerHTML = eq(`Independent datasheet example. P<sub>mp</sub> = V<sub>mp</sub>I<sub>mp</sub> = ${fixed(vmp)} V × ${fixed(imp)} A ≈ ${fixed(vmp * imp)} W · FF = P<sub>mp</sub>/(V<sub>oc</sub>I<sub>sc</sub>) = ${fixed(vmp * imp / (input.voc * input.isc), 3)}`);
    const heat = input.flow * input.cp * input.rise;
    const thermalEfficiency = p.poa > 0 ? heat / (p.poa * input.area) : null;
    get('extension-1').innerHTML =
      eq(`Q̇<sub>u</sub> = ṁc<sub>p</sub>(T<sub>out</sub> − T<sub>in</sub>) = ${fixed(heat, 1)} W of heat · η<sub>th</sub> = Q̇<sub>u</sub>/(G<sub>POA</sub>A) = ${thermalEfficiency === null ? 'undefined without sunlight' : fixed(thermalEfficiency * 100, 1) + '%'}`) +
      eq(`Q<sub>stored</sub> ≈ mc<sub>p</sub>ΔT = ${fixed(input.mass * input.cp * input.tankRise / 3600000, 3)} kWh of heat`) +
      (heat > p.poa * input.area ? '<p>These thermal inputs require more heat than the incident sunlight supplies.</p>' : '');
    const battery = batteryDay(daily.samples, input);
    get('extension-2').innerHTML =
      eq('Starts empty. PV serves the load first, then charges the battery. ΔSOC = (η × P<sub>charge</sub> − P<sub>discharge</sub>/η) × Δt.') +
      eq(`Load served: ${fixed(battery.used, 3)} kWh · Unmet load: ${fixed(battery.unmet, 3)} kWh · Surplus PV: ${fixed(battery.surplus, 3)} kWh · Final stored energy: ${fixed(battery.stored, 3)} kWh`);
    const economics = lifetime(input);
    get('extension-3').innerHTML =
      eq(`E<sub>y</sub> = E<sub>1</sub>(1 − d)<sup>y−1</sup> · Final year: ${fixed(economics.lastYear, 1)} kWh · Lifetime: ${fixed(economics.energy, 1)} kWh`) +
      eq(`LCOE = [C<sub>0</sub> + Σ C<sub>y</sub>/(1 + r)<sup>y</sup>] / [Σ E<sub>y</sub>/(1 + r)<sup>y</sup>] = €${fixed(economics.lcoe, 3)}/kWh`);
    const prediction = daily.energy * input.calibration;
    const relative = prediction > 0 ? input.observed / prediction : null;
    get('extension-4').innerHTML =
      eq(`Time availability = available hours / 24 h = ${fixed(input.available / 24 * 100, 1)}%. This standalone indicator does not change PV generation.`) +
      eq(`Calibrated prediction = modeled daily AC energy × training ratio = ${fixed(prediction, 3)} kWh. Observed / predicted = ${relative === null ? 'undefined at zero prediction' : fixed(relative * 100, 1) + '%'}.`) +
      eq(relative === null ? 'No performance flag without a positive prediction.' :
        relative < input.threshold / 100 ? 'Flagged below the selected threshold.' : 'At or above the selected threshold.');
    chart(get('solar-chart'), daily, input.hour, input.rating);
    scene?.update(input, p, Boolean(timer));
  }
  for (const [key,,,,,, unit] of definitions) {
    const slider = get(key);
    const refresh = () => {
      input[key] = Number(slider.value);
      if (key !== 'hour') {
        get(`${key}-value`).textContent = `${slider.value} ${unit}`.trim();
        slider.setAttribute('aria-valuetext', get(`${key}-value`).textContent);
      }
    };
    refresh();
    slider.addEventListener('input', () => {
      refresh();
      if (key !== 'hour' && dailyKeys.has(key)) daily = solarDay(input);
      render();
    });
  }
  updateMotionControl(get('solar-play'), false, 'solar day');
  get('solar-play').onclick = () => {
    if (timer) { clearInterval(timer); timer = undefined; }
    else timer = setInterval(() => {
      if (document.hidden || !container.isConnected) return;
      get('hour').value = String((Math.round(input.hour * 10 + 1) % 240) / 10);
      input.hour = Number(get('hour').value);
      render();
    }, 110);
    updateMotionControl(get('solar-play'), Boolean(timer), 'solar day');
    render();
  };
  render();
  return { destroy() { clearInterval(timer); scene?.destroy(); root.replaceChildren(); } };
}

const host = document.getElementById('solar-lab-root');
if (host) mountSolarLab(host);
