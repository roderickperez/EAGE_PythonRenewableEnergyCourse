import styles from './styles.css';
import { computeHydro, DEFAULT_INPUTS, HYDRO_CONSTANTS, normalizeHydroInputs } from './model.js';
import { HydroScene } from './scene.js';
import { renderConversionChart, renderPowerChart } from './charts.js';

const sliderDefinitions = [
  ['grossHeadM', 'Gross head H_gross', 5, 250, 1, 'm'],
  ['usableFlowM3s', 'Usable flow Q', 0, 30, 0.1, 'm³/s'],
  ['headLossM', 'Head loss h_loss', 0, 50, 0.5, 'm'],
  ['ratingMW', 'Electrical power cap P_r', 0.1, 25, 0.1, 'MW'],
];

function format(value, digits = 1) {
  return Number(value).toFixed(digits);
}

// Escape text before adding only the subscript markup used by our notation.
function mathematicalLabel(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/([A-Za-z])_([A-Za-z0-9]+)/g, '$1<sub>$2</sub>');
}

function sliderMarkup([key, label, min, max, step, unit], value) {
  return `<div class="control">
    <div class="control-line"><label for="${key}">${mathematicalLabel(label)}</label><output id="${key}-value" for="${key}">${format(value, step < 1 ? 1 : 0)} ${unit}</output></div>
    <input id="${key}" name="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${label.replace(/_/g, ' subscript ')}, ${unit}">
  </div>`;
}

function template(inputs, assetBase, standaloneUrl) {
  return `<article class="hydro-lab" aria-labelledby="hydro-lab-title">
    <header class="lab-head">
      <span class="lab-kicker">Interactive engineering diagram</span>
      <h2 id="hydro-lab-title">From reservoir head to electrical energy</h2>
      <p>Move the four controls, including the electrical power cap in MW. The plant, water paths, equations, and plots all use the same steady-state calculation; six-hour energy is shown in MWh.</p>
    </header>
    <section class="controls" aria-label="Hydroelectric plant inputs">
      ${sliderDefinitions.map((definition) => sliderMarkup(definition, inputs[definition[0]])).join('')}
    </section>
    <section class="stage-grid">
      <div class="plant-views">
      <figure class="engineering-card">
        <div id="plant-scene" aria-label="Interactive three-dimensional hydroelectric plant cutaway"></div>
        <figcaption>Drag to inspect · scroll or pinch to zoom · hover or tap to identify parts. Blue: water. Gold: electrical energy from generator through transformer to grid. The turbine drives the generator mechanically; there is no separate motor in generating mode. Motion and vertical scale are illustrative. The fixed intake stays submerged over the operating range; excess flow uses a low-level bypass, not the crest spillway.</figcaption>
        <div class="head-ruler" id="head-ruler"><div class="head-track"></div><strong>Head balance</strong><span id="gross-readout"></span><span id="loss-readout"></span><span id="net-readout"></span></div>
      </figure>
      </div>
      <aside class="metrics-card" aria-label="Calculated plant performance" aria-live="polite">
        <div class="power-card"><span class="eyebrow">Plant electrical power P<sub>plant</sub></span><strong id="plant-power"></strong><span class="status" id="clip-status"></span></div>
        <div class="metric-list">
          <div class="metric"><span>Uncapped electrical power</span><strong id="uncapped-power"></strong></div>
          <div class="metric"><span>Net head H<sub>net</sub></span><strong id="net-head"></strong></div>
          <div class="metric"><span>Turbine Q / spill-bypass</span><strong id="flow-split"></strong></div>
          <div class="metric"><span>Six-hour energy</span><strong id="six-hour-energy"></strong></div>
        </div>
        <div class="equations">
          <h3>Live equation substitutions</h3>
          <div class="equation" id="eq-head"></div>
          <div class="equation" id="eq-power"></div>
          <div class="equation" id="eq-cap"></div>
          <div class="equation" id="eq-energy"></div>
        </div>
        <p class="assumptions">Gross head is the headwater-to-tailwater difference in this schematic. Fixed teaching assumptions: η = ${HYDRO_CONSTANTS.efficiency}, ρ = ${HYDRO_CONSTANTS.densityKgM3} kg/m³, g = ${HYDRO_CONSTANTS.gravityMS2} m/s², Δt = ${HYDRO_CONSTANTS.durationHours} h. Excess usable flow is labelled spill/bypass because routing depends on the real plant.</p>
      </aside>
    </section>
    <section class="charts" aria-label="Hydroelectric plant charts">
      <div class="chart-card"><h3>Power versus usable flow</h3><p>Blue shows uncapped potential at the current net head. Green follows the plant output to the dashed rating limit.</p><svg id="power-chart" role="img" aria-label="Uncapped potential and capped plant output versus usable flow"></svg></div>
      <div class="chart-card"><h3>Conversion chain and energy</h3><p>Hydraulic resource, efficiency loss, rating cap, and six-hour production.</p><svg id="conversion-chart" role="img" aria-label="Power conversion and energy chart"></svg></div>
    </section>
    <footer class="lab-foot">Educational steady-state model—not a reservoir-routing, dam-safety, or design tool.${standaloneUrl ? ` <a href="${standaloneUrl}" target="_blank" rel="noopener">Open the lab full page</a>.` : ''}</footer>
  </article>`;
}

export { computeHydro } from './model.js';

export function mountHydroLab(container, initialInputs = DEFAULT_INPUTS, options = {}) {
  if (!(container instanceof Element)) throw new TypeError('mountHydroLab requires a DOM element');
  const root = container.shadowRoot ?? container.attachShadow({ mode: 'open' });
  const style = document.createElement('style');
  style.textContent = styles;
  const body = document.createElement('div');
  let inputs = normalizeHydroInputs({ ...DEFAULT_INPUTS, ...initialInputs });
  const assetBase = options.assetBase ?? new URL('.', import.meta.url).href;
  body.innerHTML = template(inputs, assetBase, options.standaloneUrl === undefined ? assetBase : options.standaloneUrl);
  root.replaceChildren(style, body);

  const get = (id) => root.getElementById(id);
  let scene;
  try { scene = new HydroScene(get('plant-scene')); }
  catch (error) { get('plant-scene').textContent = 'WebGL is unavailable. Calculations and charts remain available.'; console.warn('Hydro scene unavailable', error); }
  const sliderUnits = Object.fromEntries(sliderDefinitions.map(([key, , , , step, unit]) => [key, { unit, digits: step < 1 ? 1 : 0 }]));

  function render(nextInputs = inputs) {
    inputs = normalizeHydroInputs(nextInputs);
    const result = computeHydro(inputs);
    get('headLossM').max = String(Math.min(50, Math.max(0, inputs.grossHeadM - 0.5)));
    for (const [key] of sliderDefinitions) {
      const slider = get(key);
      slider.value = String(inputs[key]);
      const spec = sliderUnits[key];
      get(`${key}-value`).textContent = `${format(inputs[key], spec.digits)} ${spec.unit}`;
    }
    get('plant-power').textContent = `${format(result.plantMW, 3)} MW`;
    get('uncapped-power').textContent = `${format(result.uncappedMW, 2)} MW`;
    get('net-head').textContent = `${format(result.netHeadM, 1)} m`;
    get('flow-split').textContent = `${format(result.turbineFlowM3s, 1)} / ${format(result.spillFlowM3s, 1)} m³/s`;
    get('six-hour-energy').textContent = `${format(result.energy6hMWh, 2)} MWh`;
    const status = get('clip-status');
    status.className = `status ${result.clipped ? 'clipped' : 'open'}`;
    status.textContent = result.clipped ? `Clipped at P_r · uncapped ${format(result.uncappedMW, 2)} MW` : 'Within generator rating · not clipped';
    get('eq-head').textContent = `H_net = H_gross − h_loss = ${format(inputs.grossHeadM, 1)} − ${format(inputs.headLossM, 1)} = ${format(result.netHeadM, 1)} m`;
    const { efficiency, densityKgM3, gravityMS2 } = HYDRO_CONSTANTS;
    get('eq-power').textContent = `P_uncapped (MW) = ηρgQ_usableH_net ÷ 10⁶ = (${efficiency} × ${densityKgM3} × ${gravityMS2} × ${format(inputs.usableFlowM3s, 1)} × ${format(result.netHeadM, 1)}) ÷ 10⁶ ≈ ${format(result.uncappedMW, 2)} MW`;
    get('eq-cap').textContent = `P_plant = min(P_uncapped, P_r) = min(${format(result.uncappedMW, 2)}, ${format(inputs.ratingMW, 2)}) MW = ${format(result.plantMW, 2)} MW`;
    get('eq-energy').textContent = `E_6h = P_plant × Δt = ${format(result.plantMW, 2)} × 6 = ${format(result.energy6hMWh, 2)} MWh`;
    for (const id of ['clip-status', 'eq-head', 'eq-power', 'eq-cap', 'eq-energy']) {
      get(id).innerHTML = mathematicalLabel(get(id).textContent);
    }
    get('gross-readout').textContent = `Gross ${format(inputs.grossHeadM, 0)} m`;
    get('loss-readout').textContent = `Loss ${format(inputs.headLossM, 1)} m`;
    get('net-readout').textContent = `Net ${format(result.netHeadM, 1)} m`;
    get('head-ruler').style.setProperty('--loss-pct', `${Math.min(90, inputs.headLossM / inputs.grossHeadM * 100)}%`);
    renderPowerChart(get('power-chart'), result);
    renderConversionChart(get('conversion-chart'), result);
    scene?.update(result);
    return result;
  }

  for (const [key] of sliderDefinitions) {
    get(key).addEventListener('input', (event) => render({ ...inputs, [key]: event.currentTarget.value }));
  }
  const result = render(inputs);
  return {
    result,
    update(values) { return render({ ...inputs, ...values }); },
    destroy() { scene?.destroy(); root.replaceChildren(); },
  };
}

if (document.documentElement.dataset.hydroStandalone === 'true') {
  const target = document.getElementById('hydro-lab-root');
  if (target) mountHydroLab(target, DEFAULT_INPUTS, { standaloneUrl: '' });
}
