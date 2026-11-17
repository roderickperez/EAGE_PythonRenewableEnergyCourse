import { addPlot, mountPlotGroup, setupPlot } from '../hydro-lab/comparison-plots.js';
import { STC, estimateCellTemperature, inverterOutput, retainedFraction, temperatureMultiplier } from './model.js';
import { mountControls, mountLegend, showInput } from './resource-controls.js';

const $ = id => document.getElementById(id);
const blue = '#237f9b', teal = '#398470', gray = '#647c78', orange = '#a9581c';
const number = new Intl.NumberFormat('en', { maximumFractionDigits: 3 });
const n = value => number.format(value);
const withUnit = (value, unit) => `${n(value)} ${unit}`;
const poa = ['poa', 'Plane-of-array irradiance G<sub>POA</sub>', 0, 1400, 10, 800, 'W/m²'];
const nameplate = ['nameplate', 'DC nameplate P<sub>STC</sub>', 1, 25, .1, 10, 'kWp'];
const sample = (min, max, evaluate) => Array.from({ length: 41 }, (_, i) => {
  const x = min + (max - min) * i / 40;
  return [x, evaluate(x)];
});
const activePoint = (axes, x, y, baseline = 0) => {
  axes.line(axes.x(x), axes.y(baseline), axes.x(x), axes.y(y), orange, '4 3');
  axes.dot(x, y, orange, 5.7);
};

const demos = {
  'dc-temperature': {
    title: 'DC power at the selected cell temperature',
    plot: 'DC power versus cell temperature',
    caption: 'The dashed line shows irradiance-scaled DC power at 25 °C. The point follows your selected cell temperature.',
    note: 'The DC nameplate is defined at 1,000 W/m² and 25 °C cell temperature. The coefficient slider is expressed in percent per degree, so the equation uses its value divided by 100.',
    legend: [[blue, 'Temperature-corrected DC power'], [gray, 'DC power at 25 °C (dashed)'], [orange, 'Selected cell temperature']],
    controls: [poa, ['cell', 'Cell temperature T<sub>c</sub>', -20, 100, 1, 50, '°C'],
      nameplate, ['coefficient', 'Power-temperature coefficient γ<sub>P</sub>', -.8, 0, .01, -.4, '%/°C']],
    draw(svg, { poa, cell, nameplate, coefficient }) {
      const scaled = nameplate * poa / STC;
      const atTemperature = temp => scaled * temperatureMultiplier(temp, coefficient / 100);
      const result = atTemperature(cell);
      const bounds = [-20, 100];
      const axes = setupPlot(svg, bounds[1], Math.max(1, atTemperature(bounds[0]) * 1.15), 'Cell temperature (°C)', 'DC power (kW)', bounds[0]);
      axes.line(axes.x(bounds[0]), axes.y(scaled), axes.x(bounds[1]), axes.y(scaled), gray, '5 4');
      axes.series(sample(...bounds, atTemperature), blue);
      activePoint(axes, cell, result);
      svg.setAttribute('aria-label', `Modeled DC power ${withUnit(result, 'kilowatts')} at ${withUnit(cell, 'degrees Celsius')} cell temperature.`);
      return [`P<sub>DC</sub> = ${withUnit(result, 'kW')}`,
        `Irradiance scaling · ${n(nameplate)} × (${n(poa)} / ${n(STC)}) = ${withUnit(scaled, 'kW')} at 25 °C`,
        `Temperature multiplier · 1 + (${(coefficient / 100).toFixed(4)}) × (${n(cell)} − 25) = <strong>${n(temperatureMultiplier(cell, coefficient / 100))}</strong>`];
    },
  },
  'cell-temperature': {
    title: 'Estimating cell temperature from NOCT',
    plot: 'Cell temperature versus irradiance',
    caption: 'The line starts at ambient temperature. Higher NOCT makes the line steeper for the same ambient conditions.',
    note: 'This is the lesson’s reduced NOCT equation. It uses fixed coefficients and selected irradiance, ambient temperature and NOCT.',
    legend: [[blue, 'Estimated cell temperature'], [gray, 'Ambient temperature (dashed)'], [orange, 'Selected irradiance']],
    controls: [poa, ['ambient', 'Ambient temperature T<sub>a</sub>', -20, 50, 1, 25, '°C'],
      ['noct', 'Nominal operating cell temperature', 35, 60, 1, 45, '°C']],
    draw(svg, { poa, ambient, noct }) {
      const atIrradiance = level => estimateCellTemperature(level, ambient, noct);
      const result = atIrradiance(poa);
      const axes = setupPlot(svg, 1400, 140, 'Plane-of-array irradiance (W/m²)', 'Cell temperature (°C)', 0, -20);
      axes.line(axes.x(0), axes.y(ambient), axes.x(1400), axes.y(ambient), gray, '5 4');
      axes.series(sample(0, 1400, atIrradiance), blue);
      activePoint(axes, poa, result, -20);
      svg.setAttribute('aria-label', `Estimated cell temperature ${withUnit(result, 'degrees Celsius')} at irradiance ${withUnit(poa, 'watts per square metre')}.`);
      return [`T<sub>c</sub> = ${withUnit(result, '°C')}`,
        `T<sub>c</sub> = ${n(ambient)} + (${n(poa)} / 800) × (${n(noct)} − 20) = <strong>${withUnit(result, '°C')}</strong>`,
        `Cell temperature rise above ambient · ${withUnit(result - ambient, '°C')}`];
    },
  },
  'inverter-clipping': {
    title: 'From DC power to delivered AC power',
    plot: 'AC output versus DC power before losses',
    caption: 'The gap between potential and delivered AC at the selected point is clipped power. The dashed horizontal line marks the inverter rating.',
    note: 'This example applies soiling and wiring as sequential losses to modeled DC power before the inverter. The DC nameplate is only used to calculate the DC/AC ratio.',
    legend: [[teal, 'Delivered AC'], [blue, 'Potential AC (dashed)'], [gray, 'Inverter AC rating (dashed)'], [orange, 'Selected operating point']],
    controls: [
      ['dc', 'DC power before losses', 0, 30, .1, 12, 'kW'],
      ['rating', 'Inverter AC rating', .5, 25, .1, 8, 'kW'],
      ['efficiency', 'Inverter efficiency', 75, 100, .1, 97, '%'],
      nameplate,
      ['soiling', 'Soiling loss', 0, 30, 1, 5, '%'],
      ['wiring', 'Wiring loss', 0, 20, 1, 2, '%'],
    ],
    draw(svg, { dc, rating, efficiency, nameplate, soiling, wiring }) {
      const retained = retainedFraction(soiling / 100, wiring / 100);
      const atPower = power => inverterOutput(power * retained, efficiency / 100, rating);
      const { potential, ac, clip, conversionLoss } = atPower(dc);
      const dcAtInverter = dc * retained;
      const clippedAt = rating / (retained * efficiency / 100);
      const top = Math.max(rating, atPower(30).potential) * 1.12;
      const axes = setupPlot(svg, 30, top, 'DC power before losses (kW)', 'AC power (kW)');
      axes.line(axes.x(0), axes.y(rating), axes.x(30), axes.y(rating), gray, '2 4');
      axes.series([[0, 0], [30, atPower(30).potential]], blue, '6 4');
      const points = [[0, 0], ...(clippedAt < 30 ? [[clippedAt, rating]] : []), [30, atPower(30).ac]];
      axes.series(points, teal);
      if (clip > 0) {
        axes.line(axes.x(dc), axes.y(ac), axes.x(dc), axes.y(potential), blue, '3 3');
        axes.dot(dc, potential, blue, 4.5);
      }
      activePoint(axes, dc, ac);
      svg.setAttribute('aria-label', `Delivered AC ${withUnit(ac, 'kilowatts')}. Potential AC ${withUnit(potential, 'kilowatts')}. Clipping ${withUnit(clip, 'kilowatts')}.`);
      return [`P<sub>AC</sub> = ${withUnit(ac, 'kW')}`,
        `Retained fraction · (1 − ${n(soiling / 100)}) × (1 − ${n(wiring / 100)}) = ${n(retained)} · DC at inverter <strong>${withUnit(dcAtInverter, 'kW')}</strong>`,
        `Potential AC = ${n(efficiency / 100)} × ${n(dcAtInverter)} ≈ <strong>${withUnit(potential, 'kW')}</strong> · Delivered AC = min(${n(potential)}, ${n(rating)})`,
        `Upstream loss ${withUnit(dc - dcAtInverter, 'kW')} · Conversion loss ${withUnit(conversionLoss, 'kW')} · Clipping <strong>${withUnit(clip, 'kW')}</strong> · DC/AC ratio ${n(nameplate / rating)}`];
    },
  },
};

export function mountPvEquation(kind) {
  const demo = demos[kind];
  $('title').textContent = document.title = demo.title;
  $('scene').remove();
  const container = mountPlotGroup(document.querySelector('header'), demo.plot);
  const svg = addPlot(container, demo.plot, demo.caption);
  $('note').textContent = demo.note;
  mountLegend(demo.legend);

  function update() {
    const inputs = Object.fromEntries(demo.controls.map(([id, , , , , , unit]) => {
      const input = $(id);
      const value = input.valueAsNumber;
      showInput(input, withUnit(value, unit));
      return [id, value];
    }));
    const [result, ...equations] = demo.draw(svg, inputs);
    $('result').innerHTML = result;
    $('equations').innerHTML = equations.map(line => `<p>${line}</p>`).join('');
  }
  mountControls(demo.controls, update);
  update();
}
