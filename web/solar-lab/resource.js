import { addPlot, mountPlotGroup, setupPlot } from '../hydro-lab/comparison-plots.js';
import { mountDiagramMotion } from '../hydro-lab/diagram-motion.js';
import { RAD, beamOnPlane, incidenceCosine, intervalIrradiation, isotropicIrradiance, solarVector } from './model.js';
import { mountResourceScene, solarColors, solarSvgNode } from './resource-scene.js';

const $ = id => document.getElementById(id);
const format = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
const n = value => format.format(Math.abs(value) < .005 ? 0 : value);
const irradiance = value => `${n(value)} W/m²`;
const angle = value => `${n(value)}°`;
const dni = ['dni', 'Direct normal DNI', 'W/m²', 0, 1100, 10, 800];
const dhi = ['dhi', 'Diffuse horizontal DHI', 'W/m²', 0, 350, 5, 100];
const zenith = ['zenith', 'Solar zenith θ<sub>z</sub>', '°', 0, 90, 1, 40];
const tilt = ['tilt', 'Module tilt β', '°', 0, 90, 1, 30];
const sunAzimuth = ['sunAzimuth', 'Solar azimuth γ<sub>s</sub>', '°', 0, 360, 1, 180];
const moduleAzimuth = ['azimuth', 'Module azimuth γ<sub>p</sub>', '°', 0, 360, 1, 180];
const intervalControls = [320, 740, 460].flatMap((value, i) => [
  [`g${i}`, `Interval ${i + 1} · irradiance`, 'W/m²', 0, 1200, 10, value],
  [`t${i}`, `Interval ${i + 1} · duration`, 'h', .25, 4, .25, [1, 1.5, 2][i]],
]);
const demos = {
  horizontal: {
    title: 'Irradiance on a horizontal surface',
    sliders: [dni, dhi, zenith],
    legend: { beam: 'Direct beam on horizontal surface', sky: 'Diffuse sky', normal: 'Vertical reference' },
    note: 'Zenith is measured from vertical. The horizontal beam component falls to zero at the horizon even when the supplied DNI is positive.',
  },
  poa: {
    title: 'What reaches the module plane',
    sliders: [dni, ['aoi', 'Incidence angle θ<sub>i</sub>', '°', 0, 135, 1, 20],
      ['sky', 'Sky diffuse on module', 'W/m²', 0, 350, 5, 90],
      ['ground', 'Ground reflected on module', 'W/m²', 0, 250, 5, 30]],
    legend: { beam: 'Direct beam on front face', sky: 'Sky diffuse on module', ground: 'Ground reflected on module', normal: 'Module normal' },
    note: 'The example holds module tilt at 45°. Sky diffuse and ground reflected are supplied directly on the module plane. The front receives no beam at or beyond 90° incidence.',
  },
  irradiation: {
    title: 'From irradiance to irradiation',
    sliders: intervalControls,
    legend: { beam: 'Incident solar radiation' },
    note: 'Each interval uses its average plane-of-array irradiance. The animation follows the intervals in proportion to their duration. The sum is incident solar energy per square metre.',
  },
  tilt: {
    title: 'Tilt, diffuse sky and ground reflection',
    sliders: [dni, dhi, zenith, tilt, ['albedo', 'Ground albedo ρ<sub>g</sub>', '', 0, 1, .01, .2], sunAzimuth],
    legend: { beam: 'Direct beam on front face', sky: 'Isotropic sky diffuse', ground: 'Ground reflected', normal: 'Module normal' },
    note: 'The module faces south. Azimuth is clockwise from north. At zero tilt the ground term vanishes and total POA irradiance equals GHI.',
  },
  incidence: {
    title: 'Solar incidence angle',
    sliders: [zenith, tilt, sunAzimuth, moduleAzimuth],
    legend: { beam: 'Sun direction', normal: 'Module normal' },
    note: 'Both azimuths are clockwise from north. The angle is measured from the front normal. Sun direction stays visible when the sun is behind the module.',
  },
};

function geometry(tilt, azimuth, zenith, sunAzimuth) {
  return { tilt, azimuth, sunDirection: solarVector(zenith, sunAzimuth), normal: solarVector(tilt, azimuth) };
}

function calculate(kind, inputs) {
  if (kind === 'irradiation') {
    const intervals = [0, 1, 2].map(i => ({ hours: inputs[`t${i}`], irradiance: inputs[`g${i}`] }));
    return { ...geometry(30, 180, 40, 180), intervals, total: intervalIrradiation(intervals) };
  }
  const demoTilt = kind === 'horizontal' ? 0 : kind === 'poa' ? 45 : inputs.tilt;
  const zen = kind === 'poa' ? Math.abs(demoTilt - inputs.aoi) : inputs.zenith;
  const heading = kind === 'poa' && inputs.aoi > demoTilt ? 0 : inputs.sunAzimuth ?? 180;
  const state = geometry(demoTilt, inputs.azimuth ?? 180, zen, heading);
  const projection = incidenceCosine(state.sunDirection, state.normal);
  const incidence = Math.acos(projection) / RAD;
  if (kind === 'incidence') return { ...state, incidence, projection, beam: beamOnPlane(800, projection), sky: 0, ground: 0 };
  if (kind === 'poa') {
    const beam = beamOnPlane(inputs.dni, projection);
    return { ...state, beam, sky: inputs.sky, ground: inputs.ground, total: beam + inputs.sky + inputs.ground };
  }
  const components = isotropicIrradiance(inputs.dni, inputs.dhi, inputs.albedo ?? 0, state.sunDirection, state.normal);
  return { ...state, ...components, incidence, total: kind === 'horizontal' ? components.ghi : components.poa };
}

function breakdown(state) {
  return `Beam <strong>${irradiance(state.beam)}</strong> · Sky <strong>${irradiance(state.sky)}</strong> · Ground <strong>${irradiance(state.ground)}</strong>`;
}

function readout(kind, input, state) {
  switch (kind) {
    case 'horizontal':
      return [`GHI = ${irradiance(state.total)}`,
        `GHI = DNI × cos θ<sub>z</sub> + DHI = ${n(input.dni)} × cos(${angle(input.zenith)}) + ${n(input.dhi)} = <strong>${irradiance(state.total)}</strong>`,
        `Horizontal beam contribution · <strong>${irradiance(state.beam)}</strong>`];
    case 'poa':
      return [`G<sub>POA</sub> = ${irradiance(state.total)}`,
        `G<sub>POA</sub> = ${n(input.dni)} × max[cos(${angle(input.aoi)}), 0] + ${n(input.sky)} + ${n(input.ground)} = <strong>${irradiance(state.total)}</strong>`,
        breakdown(state)];
    case 'tilt':
      return [`G<sub>POA</sub> = ${irradiance(state.total)}`,
        `θ<sub>i</sub> = ${angle(state.incidence)} · GHI = ${irradiance(state.ghi)} · ρ<sub>g</sub> = ${n(input.albedo)}`,
        `Sky view factor (1 + cos β)/2 = ${n((1 + state.normal[1]) / 2)} · Ground view factor (1 − cos β)/2 = ${n((1 - state.normal[1]) / 2)}`,
        breakdown(state)];
    case 'incidence': {
      const delta = input.sunAzimuth - input.azimuth;
      const front = state.projection > 1e-12;
      return [`θ<sub>i</sub> = ${angle(state.incidence)}`,
        `cos θ<sub>i</sub> = cos(${angle(input.zenith)}) × cos(${angle(input.tilt)}) + sin(${angle(input.zenith)}) × sin(${angle(input.tilt)}) × cos(${angle(delta)}) = <strong>${n(state.projection)}</strong>`,
        `Front beam factor · <strong>${n(front ? state.projection : 0)}</strong> · ${front ? 'Sun in front of module' : 'No direct beam on front face'}`];
    }
    case 'irradiation': {
      const value = `${n(state.total)} kWh/m²`;
      const terms = state.intervals.map(({ irradiance: level, hours }) => `${n(level)} × ${n(hours)}`).join(' + ');
      return [`H<sub>POA</sub> = ${value}`,
        `H<sub>POA</sub> = (${terms}) / 1000 = <strong>${value}</strong>`];
    }
  }
}

function mountIntervalPlot() {
  const plots = mountPlotGroup($('scene'), 'Irradiance by interval');
  const svg = addPlot(plots, 'Irradiance over the selected intervals', 'Each rectangle’s area is irradiance × hours, in Wh/m².');
  let intervals, bars, marker, chart, duration, active;
  return (state, phase) => {
    if (state.intervals !== intervals) {
      intervals = state.intervals;
      duration = intervals.reduce((sum, { hours }) => sum + hours, 0);
      chart = setupPlot(svg, duration, 1200, 'Elapsed time (h)', 'Irradiance (W/m²)');
      svg.setAttribute('aria-label', intervals.map(({ irradiance, hours }, i) =>
        `Interval ${i + 1}, ${irradiance} watts per square metre for ${hours} hours`).join('. '));
      let start = 0;
      bars = intervals.map(({ irradiance, hours }) => {
        const end = start + hours;
        const bar = solarSvgNode(svg, 'rect', {
          x: chart.x(start), y: chart.y(irradiance), width: chart.x(end) - chart.x(start),
          height: chart.y(0) - chart.y(irradiance), stroke: '#237f9b', 'stroke-width': 1.3,
        });
        start = end;
        return { bar, end, irradiance };
      });
      marker = chart.line(chart.x(0), chart.y(0), chart.x(0), chart.y(1200), '#94602a', '4 3');
      active = undefined;
    }
    const hour = phase * duration;
    const selected = bars.find(item => hour < item.end) ?? bars.at(-1);
    if (active !== selected) {
      active = selected;
      for (const item of bars) item.bar.setAttribute('fill', item === selected ? '#90c5d3' : '#cfe6e9');
    }
    marker.setAttribute('x1', chart.x(hour));
    marker.setAttribute('x2', chart.x(hour));
    return selected.irradiance;
  };
}

const requested = new URLSearchParams(location.search).get('demo');
const kind = Object.hasOwn(demos, requested) ? requested : 'horizontal';
const demo = demos[kind];
$('title').textContent = document.title = demo.title;
$('note').textContent = demo.note;
$('drawing').setAttribute('aria-label', `Isometric solar diagram. ${Object.values(demo.legend).join('. ')}.`);
for (const [key, name] of Object.entries(demo.legend)) {
  const item = document.createElement('span');
  item.textContent = name;
  item.style.setProperty('--swatch', solarColors[key]);
  $('legend').append(item);
}
const fields = demo.sliders.map(([key, label, unit, min, max, step, initial]) => {
  const control = document.createElement('div');
  control.className = 'control';
  control.innerHTML = `<label for="${key}"><span>${label}</span><output for="${key}"></output></label>
    <input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${initial}">`;
  $('controls').append(control);
  return { key, unit, input: control.querySelector('input'), output: control.querySelector('output') };
});

const render = mountResourceScene($('drawing'), kind !== 'irradiation');
const intervalPlot = kind === 'irradiation' ? mountIntervalPlot() : null;
const updateMotion = mountDiagramMotion(document.querySelector('main'), $('motion'), (state, phase) => {
  const scene = intervalPlot ? { ...state, beam: intervalPlot(state, phase), sky: 0, ground: 0 } : state;
  render(scene, intervalPlot ? phase * 5 : phase);
}, () => kind === 'irradiation' ? 1 / 9000 : 1 / 1650);

function update() {
  const inputs = Object.fromEntries(fields.map(({ key, unit, input, output }) => {
    const number = input.valueAsNumber;
    const text = unit === '°' ? angle(number) : unit ? `${n(number)} ${unit}` : n(number);
    output.value = text;
    input.setAttribute('aria-valuetext', text);
    return [key, number];
  }));
  const state = calculate(kind, inputs);
  const [result, ...rows] = readout(kind, inputs, state);
  $('result').innerHTML = result;
  $('equations').innerHTML = rows.map(row => `<p>${row}</p>`).join('');
  updateMotion(state);
}

for (const { input } of fields) input.addEventListener('input', update);
update();

if (window.frameElement) {
  const frame = window.frameElement;
  const card = document.querySelector('main');
  new ResizeObserver(() => {
    frame.style.setProperty('--solar-resource-height', `${Math.ceil(card.getBoundingClientRect().height)}px`);
  }).observe(card);
}
