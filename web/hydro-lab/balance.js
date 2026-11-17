import { flowBalance, MAX_FLOW_M3S, MAX_INTERVAL_MINUTES, netHead, powerBalance } from './balance-model.js';
import { mountBalanceScene, palette } from './balance-scene.js';
import { mountOperationDemo, operationDemos } from './operation.js';
import { mountPumpedStorage } from './pumped-storage.js';

const requested = new URLSearchParams(location.search).get('demo');
if (requested === 'pumped-storage') mountPumpedStorage();
else if (operationDemos.has(requested)) mountOperationDemo(requested);
else mountBalanceDemo(requested);

function mountBalanceDemo(requested) {
  const titles = new Map([
    ['net-head', 'From gross head to net head'],
    ['flow-volume', 'From flow rate to water volume'],
    ['hydraulic-power', 'From flow and head to hydraulic power'],
    ['electrical-power', 'From hydraulic power to electricity'],
  ]);
  const kind = titles.has(requested) ? requested : 'net-head';
  const isNetHead = kind === 'net-head';
  const isFlow = kind === 'flow-volume';
  const isElectrical = kind === 'electrical-power';
  const $ = id => document.getElementById(id);
  const format = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
  const number = value => format.format(value);
  const title = titles.get(kind);
  $('title').textContent = document.title = title;
  $('result').setAttribute('aria-label', isNetHead ? 'Net head' : isFlow ? 'Accumulated water volume' : isElectrical ? 'Electrical power' : 'Hydraulic power');
  $('drawing').setAttribute('aria-label', {
    'net-head': 'Upstream and downstream water surfaces and a stacked gauge showing usable head and hydraulic loss',
    'flow-volume': 'Water flowing from an outlet into a tank whose water level represents the accumulated volume',
    'hydraulic-power': 'Water flowing down a penstock as the vertical difference represents net head',
    'electrical-power': 'Water drives a turbine connected by a shaft to a generator, which sends electrical power along a cable',
  }[kind]);
  $('note').textContent = {
    'net-head': 'Gross head is the difference between the water surfaces in this example. The downstream surface stays fixed. The orange segment represents dissipated head.',
    'flow-volume': 'The tank shows the total for the selected interval. Time changes the amount collected, while density changes mass flow at the same volume flow.',
    'hydraulic-power': 'The diagram represents the usable head at the turbine and flow through a fixed pipe. The power calculation uses net head with hydraulic losses already deducted.',
    'electrical-power': 'The turbine delivers shaft power to the generator. This example reports electrical power at the generator, before transformer losses, station loads or rating limits.',
  }[kind];

  const legend = isNetHead
    ? [[palette.water, 'Water surfaces'], [palette.net, 'Net head'], [palette.loss, 'Hydraulic loss']]
    : isFlow
      ? [[palette.water, 'Water collected'], ['#42a6bf', 'Water entering the tank']]
      : [[palette.water, 'Water flow'], ...(isElectrical
        ? [[palette.shaft, 'Turbine, shaft and generator'], [palette.electricity, 'Electrical output']]
        : [])];
  for (const [color, name] of legend) {
    const item = document.createElement('span');
    item.style.setProperty('--swatch', color);
    item.textContent = name;
    $('legend').append(item);
  }

  const definitions = isNetHead ? [
    ['gross', 'Gross head', 'H<sub>gross</sub>', 'm', 100, 1, 30],
    ['loss', 'Hydraulic loss', 'h<sub>loss</sub>', 'm', 30, 1, 2],
  ] : isFlow ? [
    ['flow', 'Volume flow', 'Q', 'm³/s', MAX_FLOW_M3S, .1, 3],
    ['minutes', 'Elapsed time', 'Δt', 'min', MAX_INTERVAL_MINUTES, 5, 120],
    ['density', 'Water density', 'ρ', 'kg/m³', 1050, 1, 1000, 950],
  ] : [
    ['head', 'Net head', 'H<sub>net</sub>', 'm', 100, 1, 28],
    ['flow', 'Turbine flow', 'Q', 'm³/s', 30, .1, 10],
    ['density', 'Water density', 'ρ', 'kg/m³', 1050, 1, 1000, 950],
    ['gravity', 'Gravity', 'g', 'm/s²', 9.85, .01, 9.81, 9.75],
    ...(isElectrical ? [
      ['turbine', 'Turbine efficiency', 'η<sub>t</sub>', '%', 100, 1, 85],
      ['generator', 'Generator efficiency', 'η<sub>g</sub>', '%', 100, 1, 95],
    ] : []),
  ];
  const fields = Object.fromEntries(definitions.map(([key, label, symbol, unit, max, step, value, min = 0]) => {
    const control = document.createElement('div');
    control.className = 'control';
    control.innerHTML = `<label for="${key}"><span>${label} <i>${symbol}</i></span><output for="${key}"></output></label>
      <input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${label} (${unit})">`;
    $('controls').append(control);
    return [key, { input: control.querySelector('input'), output: control.querySelector('output'), unit }];
  }));

  const equations = Array.from({ length: isNetHead || kind === 'hydraulic-power' ? 1 : 3 }, () => document.createElement('div'));
  $('equations').append(...equations);
  let draw;
  try {
    draw = mountBalanceScene($('scene'), kind, $('motion'));
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
    } else if (isFlow) {
      const { flow, minutes, density } = values;
      const { seconds, volume, massFlow } = flowBalance(flow, minutes, density);
      $('result').value = `V = ${number(volume)} m³`;
      equations[0].textContent = `Δt = ${number(minutes)} min × 60 s/min = ${number(seconds)} s`;
      equations[1].textContent = `V = Q × Δt = ${number(flow)} m³/s × ${number(seconds)} s = ${number(volume)} m³`;
      equations[2].textContent = `ṁ = ρQ = ${number(density)} kg/m³ × ${number(flow)} m³/s = ${number(massFlow)} kg/s`;
      draw?.([volume, flow]);
    } else {
      const { head, flow, density, gravity, turbine = 100, generator = 100 } = values;
      const { hydraulicW, shaftW, electricalW } = powerBalance(flow, head, density, gravity, turbine / 100, generator / 100);
      const megawatts = watts => `${(watts / 1e6).toFixed(3)} MW`;
      $('result').innerHTML = isElectrical ? `P<sub>e</sub> = ${megawatts(electricalW)}` : `P<sub>hyd</sub> = ${megawatts(hydraulicW)}`;
      equations[0].innerHTML = `P<sub>hyd</sub> = ρgQH<sub>net</sub> = ${number(density)} × ${number(gravity)} × ${number(flow)} × ${number(head)} = ${number(hydraulicW)} W`;
      if (isElectrical) {
        equations[1].innerHTML = `P<sub>shaft</sub> = η<sub>t</sub>P<sub>hyd</sub> = ${number(turbine / 100)} × ${number(hydraulicW)} W = ${number(shaftW)} W`;
        equations[2].innerHTML = `P<sub>e</sub> = η<sub>g</sub>P<sub>shaft</sub> = ${number(generator / 100)} × ${number(shaftW)} W ≈ ${number(electricalW)} W`;
      }
      draw?.([head, flow, shaftW, electricalW]);
    }
  }
  for (const { input } of Object.values(fields)) input.addEventListener('input', update);
  update();
}

if (window.frameElement) {
  const frame = window.frameElement;
  const card = document.querySelector('main');
  new ResizeObserver(() => {
    frame.style.setProperty('--hydraulic-head-height', `${Math.ceil(card.getBoundingClientRect().height)}px`);
  }).observe(card);
}
