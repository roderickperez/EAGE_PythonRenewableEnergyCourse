import { flowBalance, MAX_FLOW_M3S, MAX_INTERVAL_MINUTES, netHead } from './balance-model.js';
import { mountBalanceScene, palette } from './balance-scene.js';

const isNetHead = new URLSearchParams(location.search).get('demo') !== 'flow-volume';
const kind = isNetHead ? 'net-head' : 'flow-volume';
const $ = id => document.getElementById(id);
const format = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
const number = value => format.format(value);
const title = isNetHead ? 'From gross head to net head' : 'From flow rate to water volume';
$('title').textContent = document.title = title;
$('result').setAttribute('aria-label', isNetHead ? 'Net head' : 'Accumulated water volume');
$('scene').setAttribute('aria-label', isNetHead
  ? 'Upstream and downstream water surfaces and a stacked gauge showing usable head and hydraulic loss'
  : 'Water flowing from an outlet into a tank whose water level represents the accumulated volume');
$('note').textContent = isNetHead
  ? 'Gross head is the difference between the water surfaces in this example. The downstream surface stays fixed. The orange segment represents dissipated head.'
  : 'The tank shows the total for the selected interval. Time changes the amount collected, while density changes mass flow at the same volume flow.';

const legend = isNetHead
  ? [[palette.water, 'Water surfaces'], [palette.net, 'Net head'], [palette.loss, 'Hydraulic loss']]
  : [[palette.water, 'Water collected'], ['#42a6bf', 'Water entering the tank']];
for (const [color, name] of legend) {
  const item = document.createElement('span');
  item.style.setProperty('--swatch', color);
  item.textContent = name;
  $('legend').append(item);
}

const definitions = isNetHead ? [
  ['gross', 'Gross head', 'H<sub>gross</sub>', 'm', 100, 1, 30],
  ['loss', 'Hydraulic loss', 'h<sub>loss</sub>', 'm', 30, 1, 2],
] : [
  ['flow', 'Volume flow', 'Q', 'm³/s', MAX_FLOW_M3S, .1, 3],
  ['minutes', 'Elapsed time', 'Δt', 'min', MAX_INTERVAL_MINUTES, 5, 120],
  ['density', 'Water density', 'ρ', 'kg/m³', 1050, 1, 1000, 950],
];
const fields = Object.fromEntries(definitions.map(([key, label, symbol, unit, max, step, value, min = 0]) => {
  const control = document.createElement('div');
  control.className = 'control';
  control.innerHTML = `<label for="${key}"><span>${label} <i>${symbol}</i></span><output for="${key}"></output></label>
    <input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${label} (${unit})">`;
  $('controls').append(control);
  return [key, { input: control.querySelector('input'), output: control.querySelector('output'), unit }];
}));

const equations = isNetHead ? [document.createElement('div')] : Array.from({ length: 3 }, () => document.createElement('div'));
$('equations').append(...equations);
let draw;
try {
  draw = mountBalanceScene($('scene'), kind);
} catch {
  $('fallback').hidden = false;
}

function update() {
  if (isNetHead) fields.loss.input.max = fields.gross.input.value;
  const values = Object.fromEntries(Object.entries(fields).map(([key, { input, output, unit }]) => {
    const value = input.valueAsNumber;
    output.value = `${number(value)} ${unit}`;
    return [key, value];
  }));
  if (isNetHead) {
    const { gross, loss } = values;
    const net = netHead(gross, loss);
    $('result').innerHTML = `H<sub>net</sub> = ${number(net)} m`;
    equations[0].innerHTML = `H<sub>net</sub> = H<sub>gross</sub> − h<sub>loss</sub> = ${number(gross)} m − ${number(loss)} m = <strong>${number(net)} m</strong>`;
    draw?.([gross, loss]);
  } else {
    const { flow, minutes, density } = values;
    const { seconds, volume, massFlow } = flowBalance(flow, minutes, density);
    $('result').value = `V = ${number(volume)} m³`;
    equations[0].textContent = `Δt = ${number(minutes)} min × 60 s/min = ${number(seconds)} s`;
    equations[1].textContent = `V = Q × Δt = ${number(flow)} m³/s × ${number(seconds)} s = ${number(volume)} m³`;
    equations[2].textContent = `ṁ = ρQ = ${number(density)} kg/m³ × ${number(flow)} m³/s = ${number(massFlow)} kg/s`;
    draw?.([volume, flow]);
  }
}
for (const { input } of Object.values(fields)) input.addEventListener('input', update);
update();

if (window.frameElement) {
  const frame = window.frameElement;
  const card = document.querySelector('main');
  new ResizeObserver(() => {
    frame.style.setProperty('--hydraulic-head-height', `${Math.ceil(card.getBoundingClientRect().height)}px`);
  }).observe(card);
}
