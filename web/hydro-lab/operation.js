import { MAX_PIPE_FLOW_M3S, pipeHeadProfile, pipeLoss, turbineOperation } from './balance-model.js';
import { mountBalanceScene, palette } from './balance-scene.js';
import { HYDRO_CONSTANTS } from './model.js';
import { mountPipeLossPlots } from './pipe-loss-plots.js';

export const operationDemos = new Map([
  ['hydraulic-loss', 'Hydraulic losses in a pipe'],
  ['turbine-operation', 'Turbine families and generator limits'],
]);
const families = ['Pelton', 'Francis', 'Kaplan'];
const familyDescriptions = [
  'Pelton · A jet strikes buckets around an impulse runner.',
  'Francis · Water enters radially and exits along the shaft.',
  'Kaplan · Water passes along the shaft through a propeller runner.',
];

export function mountOperationDemo(kind) {
  const $ = id => document.getElementById(id);
  const lossMode = kind === 'hydraulic-loss';
  const format = new Intl.NumberFormat('en', { maximumFractionDigits: 3 });
  const n = value => format.format(value);
  $('title').textContent = document.title = operationDemos.get(kind);
  $('result').setAttribute('aria-label', lossMode ? 'Total head loss in metres' : 'Electrical output in megawatts');
  $('drawing').setAttribute('aria-label', lossMode
    ? 'A level, constant-diameter pipe with moving water and connected upstream and downstream open pressure taps'
    : 'A selectable turbine runner with separate generating and bypass water routes');
  $('note').textContent = lossMode
    ? 'Tap B is held at 2 m of pressure head. Tap A rises by the head needed to drive the selected flow. The columns share a scale that adjusts for large losses. The bend represents one lumped fitting loss; gravity is fixed at 9.81 m/s².'
    : 'The family changes the runner and flow path. The calculation uses the head and efficiency you choose, with ρ = 1,000 kg/m³ and g = 9.81 m/s².';

  const legend = lossMode
    ? [[palette.water, 'Tap A · Inlet'], [palette.net, 'Tap B · Outlet'], ['#8a645f', 'Fitting']]
    : [[palette.water, 'Turbine flow'], [palette.net, 'Bypass flow'], [palette.electricity, 'Electrical output']];
  for (const [color, name] of legend) {
    const item = document.createElement('span');
    item.style.setProperty('--swatch', color);
    item.textContent = name;
    $('legend').append(item);
  }

  let family;
  if (!lossMode) {
    const label = document.createElement('label');
    label.htmlFor = 'family';
    label.textContent = 'Turbine family';
    family = document.createElement('select');
    family.id = 'family';
    families.forEach(name => family.add(new Option(name, name)));
    family.value = 'Francis';
    const group = document.createElement('div');
    group.className = 'family-choice';
    group.append(label, family);
    $('controls').before(group);
  }

  const sliders = lossMode ? [
    ['flow', 'Volume flow Q', 'm³/s', 0, MAX_PIPE_FLOW_M3S, .1, 4],
    ['length', 'Pipe length L', 'm', 50, 1000, 10, 250],
    ['diameter', 'Pipe diameter D', 'm', .8, 4, .1, 1.5],
    ['factor', 'Darcy friction factor', '', .01, .05, .001, .02],
    ['fittings', 'Sum of fittings ΣK', '', 0, 10, .1, 2],
  ] : [
    ['flow', 'Available flow Q', 'm³/s', 0, 30, .1, 10],
    ['head', 'Net head', 'm', 0, 220, 1, 28],
    ['efficiency', 'Overall efficiency η', '%', 0, 100, 1, 85],
    ['rating', 'Generator rating', 'MW', .1, 25, .1, 2],
  ];
  const fields = Object.fromEntries(sliders.map(([key, label, unit, min, max, step, value]) => {
    const control = document.createElement('div');
    control.className = 'control';
    control.innerHTML = `<label for="${key}"><span>${label}</span><output for="${key}"></output></label>
      <input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${unit ? `${label} (${unit})` : label}">`;
    $('controls').append(control);
    return [key, { input: control.querySelector('input'), output: control.querySelector('output'), unit }];
  }));
  const equations = Array.from({ length: lossMode ? 4 : 3 }, () => document.createElement('div'));
  $('equations').append(...equations);
  const { densityKgM3, gravityMS2 } = HYDRO_CONSTANTS;
  const renderPlots = lossMode ? mountPipeLossPlots($('note'), gravityMS2) : null;
  let draw;
  try {
    draw = mountBalanceScene($('scene'), kind, $('motion'));
  } catch {
    $('fallback').hidden = false;
  }

  function update() {
    const values = Object.fromEntries(Object.entries(fields).map(([key, { input, output, unit }]) => {
      const value = input.valueAsNumber;
      output.value = unit ? `${n(value)} ${unit}` : n(value);
      return [key, value];
    }));
    if (lossMode) {
      const { flow, length, diameter, factor, fittings } = values;
      const loss = pipeLoss(flow, length, diameter, factor, fittings, gravityMS2);
      $('result').innerHTML = `h<sub>loss</sub> = ${loss.total.toFixed(3)} m`;
      equations[0].innerHTML = `A<sub>p</sub> = πD²/4 ≈ ${n(loss.area)} m² · v = Q/A<sub>p</sub> ≈ ${n(loss.velocity)} m/s`;
      equations[1].innerHTML = `h<sub>f</sub> = f<sub>D</sub>(L/D)v²/(2g) ≈ ${loss.friction.toFixed(3)} m`;
      equations[2].innerHTML = `h<sub>m</sub> = ΣK · v²/(2g) ≈ ${loss.fittings.toFixed(3)} m · Total = h<sub>f</sub> + h<sub>m</sub>`;
      const head = pipeHeadProfile(loss, length);
      equations[3].innerHTML = `Tap A: <strong>${head[0][1].toFixed(3)} m</strong> · Tap B: <strong>${head.at(-1)[1].toFixed(3)} m</strong> · A − B = h<sub>loss</sub>`;
      draw?.([length, loss.velocity, diameter, head[0][1], head.at(-1)[1]]);
      renderPlots(values, loss);
    } else {
      const { flow, head, efficiency, rating } = values;
      const result = turbineOperation(flow, head, efficiency / 100, rating, densityKgM3, gravityMS2);
      $('result').innerHTML = `P<sub>plant</sub> = ${result.outputMW.toFixed(3)} MW`;
      equations[0].textContent = familyDescriptions[family.selectedIndex];
      equations[1].innerHTML = result.ratedFlow === null
        ? 'Rated flow is undefined at zero head or efficiency. The teaching model routes all available water to bypass.'
        : `Q<sub>r</sub> = P<sub>r</sub> × 10⁶/(ηρgH<sub>net</sub>) ≈ ${n(result.ratedFlow)} m³/s`;
      equations[2].innerHTML = `Q<sub>turbine</sub> = ${n(result.turbineFlow)} m³/s · Q<sub>bypass</sub> = ${n(result.bypassFlow)} m³/s`;
      draw?.([family.selectedIndex, result.turbineFlow, result.bypassFlow, result.outputMW]);
    }
  }
  for (const { input } of Object.values(fields)) input.addEventListener('input', update);
  family?.addEventListener('change', update);
  update();
}
