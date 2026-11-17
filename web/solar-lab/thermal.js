import { addPlot, mountPlotGroup, setupPlot } from '../hydro-lab/comparison-plots.js';
import { mountDiagramMotion, mountDiagramRenderer } from '../hydro-lab/diagram-motion.js';
import { mountControls, mountLegend, showInput } from './resource-controls.js';
import { THERMAL_ABSORPTION, THERMAL_AMBIENT_C, THERMAL_IRRADIANCE_MAX, thermalCollector } from './thermal-model.js';
import { mountThermalFallback, mountThermalThree, thermalColors } from './thermal-scene.js';

const $ = id => document.getElementById(id);
const number = new Intl.NumberFormat('en', { maximumFractionDigits: 3 });
const n = value => number.format(value);
const kw = watts => `${n(watts / 1000)} kW`;
const definitions = [
  ['irradiance', 'Collector irradiance G', 0, THERMAL_IRRADIANCE_MAX, 10, 800, 'W/m²'],
  ['area', 'Collector area A', 1, 12, .5, 4, 'm²'],
  ['flowDensity', 'Mass flow per m²', 0, .05, .01, .02, 'kg/(s·m²)'],
  ['heatCapacity', 'Fluid specific heat c<sub>p</sub>', 3800, 4500, 10, 4180, 'J/(kg·K)'],
  ['inlet', 'Inlet temperature T<sub>in</sub>', THERMAL_AMBIENT_C, 65, 1, 35, '°C'],
  ['lossCoefficient', 'Heat-loss coefficient U', 4, 10, .5, 6, 'W/(m²·K)'],
];

export function mountSolarThermal() {
  $('title').textContent = document.title = 'Useful heat from a solar collector';
  $('note').textContent = 'Ambient air is fixed at 20 °C and the collector absorbs 78% of incident power before heat loss. Total mass flow equals flow per unit area × collector area. The diagram shows the absorber with its cover removed, and the flow animation is schematic.';
  const label = 'Isometric flat-plate thermal collector. Gold pulses show sunlight. Fluid enters through the cool branch, crosses the absorber and exits warmer.';
  const host = $('scene');
  const svg = $('drawing');
  svg.setAttribute('aria-label', label);
  mountLegend([[thermalColors.sun, 'Incident sunlight'], [thermalColors.inlet, 'Cooler inlet'],
    [thermalColors.outlet, 'Warmer outlet']]);
  const plotGroup = mountPlotGroup(host, 'Solar collector heat balance');
  const plot = addPlot(plotGroup, 'Useful heat versus irradiance',
    'The dashed line is solar power absorbed by the collector. The solid line is useful fluid heating. The dot follows the irradiance slider.');
  const render = mountDiagramRenderer(host, () => {
    const draw = mountThermalThree(host, label);
    svg.setAttribute('hidden', '');
    return draw;
  }, () => {
    svg.removeAttribute('hidden');
    return mountThermalFallback(svg);
  });
  const animate = mountDiagramMotion(document.querySelector('main'), $('motion'), render, ({ inputs }) =>
    inputs.irradiance > 0 ? .00013 + .004 * inputs.flowDensity : 0);

  function update() {
    const inputs = Object.fromEntries(definitions.map(([id, , , , , , unit]) => {
      const input = $(id);
      const value = input.valueAsNumber;
      showInput(input, `${n(value)} ${unit}`);
      return [id, value];
    }));
    inputs.flow = inputs.area * inputs.flowDensity;
    const result = thermalCollector(inputs);
    $('result').innerHTML = `Q̇<sub>u</sub> = ${kw(result.useful)} of heat`;
    const heat = result.running
      ? `Q̇<sub>u</sub> = ṁ c<sub>p</sub>(T<sub>out</sub> − T<sub>in</sub>) ≈ ${n(inputs.flow)} × ${n(inputs.heatCapacity)} × (${n(result.outlet)} − ${n(inputs.inlet)}) = <strong>${kw(result.useful)}</strong>`
      : `Pump off · ${inputs.flow === 0 ? 'selected flow is zero' : 'solar input cannot provide useful heating at this inlet temperature'}. Outlet temperature is unavailable without circulation.`;
    const loss = result.running ? ` · Ambient heat loss ${kw(result.thermalLoss)}` : '';
    const efficiency = result.efficiency === null ? 'undefined at zero irradiance' : `${n(result.efficiency * 100)}%`;
    $('equations').innerHTML = `<p>${heat}</p>
      <p>Incident solar power ${kw(result.incident)} · Absorbed solar power ${kw(result.absorbed)}${loss}</p>
      <p>Thermal efficiency η<sub>th</sub> = Q̇<sub>u</sub> / (GA) = <strong>${efficiency}</strong></p>`;

    const evaluate = irradiance => thermalCollector({ ...inputs, irradiance });
    const end = evaluate(THERMAL_IRRADIANCE_MAX);
    const top = Math.max(1, Math.ceil(end.absorbed / 1000));
    const axes = setupPlot(plot, THERMAL_IRRADIANCE_MAX, top, 'Collector irradiance (W/m²)', 'Thermal power (kW)');
    const threshold = inputs.lossCoefficient * (inputs.inlet - THERMAL_AMBIENT_C) / THERMAL_ABSORPTION;
    const useful = [[0, 0], ...(threshold > 0 ? [[threshold, 0]] : []), [THERMAL_IRRADIANCE_MAX, end.useful / 1000]];
    axes.series([[0, 0], [THERMAL_IRRADIANCE_MAX, end.absorbed / 1000]], thermalColors.sun, '6 4');
    axes.series(useful, '#237f9b');
    axes.line(axes.x(inputs.irradiance), axes.y(0), axes.x(inputs.irradiance), axes.y(result.useful / 1000), thermalColors.outlet, '4 3');
    axes.dot(inputs.irradiance, result.useful / 1000, thermalColors.outlet, 5.5);
    plot.setAttribute('aria-label', `At ${inputs.irradiance} watts per square metre, the collector absorbs ${kw(result.absorbed)} and delivers ${kw(result.useful)} of heat.`);
    animate({ inputs, result });
  }

  mountControls(definitions, update);
  update();
}
