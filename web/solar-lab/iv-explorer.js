import { addPlot, mountPlotGroup, setupPlot } from '../hydro-lab/comparison-plots.js';
import { moduleCurve } from './iv-model.js';
import { solarSvgNode } from './resource-scene.js';
import { mountControls, mountLegend, showInput } from './resource-controls.js';

const $ = id => document.getElementById(id);
const fixed = (value, unit) => `${value.toFixed(2)} ${unit}`;
const irradianceControls = [
  ['irradiance', 'Effective irradiance', 0, 1200, 10, 1000],
  ['temperature', 'Cell temperature', -10, 85, 1, 25],
];

function readConditions() {
  showInput($('irradiance'), `${$('irradiance').value} W/m²`);
  showInput($('temperature'), `${$('temperature').value} °C`);
  return moduleCurve($('irradiance').valueAsNumber, $('temperature').valueAsNumber);
}

export function mountIvExplorer() {
  $('title').textContent = document.title = 'Current, voltage and maximum power';
  $('scene').remove();
  const plots = mountPlotGroup(document.querySelector('header'), 'Photovoltaic module curves');
  plots.classList.add('iv-plots');
  const iv = addPlot(plots, 'Current–voltage (I–V)', 'Short circuit at 0 V. Open circuit at 0 A.');
  const pv = addPlot(plots, 'Power–voltage (P–V)', 'Power is zero at both endpoints. The peak marks maximum DC power.');
  mountLegend([['#237f9b', 'Selected conditions'], ['#647c78', 'STC reference (dashed)'], ['#a9581c', 'Maximum power point']]);
  const reference = moduleCurve(1000, 25);
  $('note').textContent = 'Canadian Solar CS5P-220M, De Soto single-diode model. Effective irradiance is the light available to the cells. Cell temperature is set independently of irradiance. Axes stay fixed so operating points can be compared.';

  function update() {
    const { peak, isc, voc, samples, fillFactor } = readConditions();
    $('result').innerHTML = peak ? `P<sub>mp</sub> = ${fixed(peak.power, 'W')}` : 'No generating point';
    $('equations').innerHTML = peak
      ? `<p>V<sub>mp</sub> = <strong>${fixed(peak.voltage, 'V')}</strong> · I<sub>mp</sub> = <strong>${fixed(peak.current, 'A')}</strong> · P<sub>mp</sub> = V<sub>mp</sub> × I<sub>mp</sub></p>
        <p>I<sub>sc</sub> = ${fixed(isc, 'A')} · V<sub>oc</sub> = ${fixed(voc, 'V')} · Fill factor = ${fillFactor.toFixed(3)}</p>`
      : '<p>At zero irradiance, available DC output is zero.</p>';
    for (const [svg, column, maximum, yLabel, property, unit] of [
      [iv, 1, 8, 'Module current (A)', 'current', 'A'],
      [pv, 2, 360, 'Module DC power (W)', 'power', 'W'],
    ]) {
      const axes = setupPlot(svg, 72, maximum, 'Module voltage (V)', yLabel);
      const curve = list => list.map(sample => [sample[0], sample[column]]);
      if (peak) {
        axes.series(curve(samples), '#237f9b', '', 3.2);
        axes.series(curve(reference.samples), '#647c78', '5 4', 1.5);
        const x = axes.x(peak.voltage);
        axes.line(x, axes.y(0), x, axes.y(peak[property]), '#a9581c', '4 3');
        axes.dot(peak.voltage, peak[property], '#a9581c', 6);
        svg.setAttribute('aria-label', `${yLabel} versus voltage. Maximum power point at ${fixed(peak.voltage, 'V')} and ${fixed(peak[property], unit)}.`);
      } else {
        axes.series(curve(reference.samples), '#647c78', '5 4', 1.5);
        axes.series([[0, 0], [72, 0]], '#237f9b');
        svg.setAttribute('aria-label', `${yLabel} versus voltage. Zero output at the selected irradiance.`);
      }
    }
  }
  mountControls(irradianceControls, update);
  update();
}

export function mountIvEquation(kind) {
  const isFillFactor = kind === 'fill-factor';
  $('title').textContent = document.title = isFillFactor ? 'Fill factor as a ratio of areas' : 'DC power at an operating point';
  $('scene').remove();
  const plots = mountPlotGroup(document.querySelector('header'), 'Photovoltaic current–voltage plot');
  const svg = addPlot(plots, 'Module current–voltage curve', isFillFactor
    ? 'The solid rectangle ends at the maximum power point. The dashed rectangle uses open-circuit voltage and short-circuit current.'
    : 'The rectangle’s area is voltage × current. Slide to either end of the curve to get zero power.');
  const reference = moduleCurve(1000, 25);
  mountLegend([
    ['#237f9b', 'I–V curve'],
    ['#a9581c', isFillFactor ? 'Maximum power rectangle' : 'Selected operating point'],
    ...(isFillFactor ? [['#647c78', 'Open-circuit × short-circuit rectangle (dashed)']] : []),
  ]);
  $('note').textContent = isFillFactor
    ? 'The area ratio is dimensionless. Both rectangles use the same module under the selected conditions.'
    : 'Irradiance is fixed at 1,000 W/m² and cell temperature at 25 °C. The slider spans 0 V to open circuit.';

  function update() {
    const selected = isFillFactor ? readConditions() : reference;
    const axes = setupPlot(svg, 72, 8, 'Module voltage (V)', 'Module current (A)');
    function rectangle(voltage, current, color, dashed = false) {
      if (voltage <= 0 || current <= 0) return;
      solarSvgNode(svg, 'rect', {
        x: axes.x(0), y: axes.y(current), width: axes.x(voltage) - axes.x(0),
        height: axes.y(0) - axes.y(current), fill: color, 'fill-opacity': dashed ? .08 : .20,
        stroke: color, 'stroke-width': 1.4, 'stroke-dasharray': dashed ? '5 4' : 'none',
      });
    }
    if (isFillFactor && !selected.peak) {
      $('result').textContent = 'Fill factor undefined';
      $('equations').innerHTML = '<p>At zero irradiance, the model gives zero available power. The reference area is also zero.</p>';
      axes.series([[0, 0], [72, 0]], '#237f9b');
      svg.setAttribute('aria-label', 'I–V curve at zero irradiance. Both rectangle areas are zero.');
      return;
    }

    let point;
    if (isFillFactor) {
      const { voc, isc, peak, fillFactor } = selected;
      point = peak;
      rectangle(voc, isc, '#647c78', true);
      $('result').textContent = `FF = ${fillFactor.toFixed(3)}`;
      $('equations').innerHTML = `<p>V<sub>mp</sub>I<sub>mp</sub> ≈ ${fixed(peak.power, 'W')} · V<sub>oc</sub>I<sub>sc</sub> ≈ ${fixed(voc * isc, 'W')}</p>
        <p>FF = (V<sub>mp</sub>I<sub>mp</sub>) / (V<sub>oc</sub>I<sub>sc</sub>) ≈ <strong>${fillFactor.toFixed(3)}</strong></p>`;
      svg.setAttribute('aria-label', `Maximum point area ${fixed(peak.power, 'watts')}. Open circuit times short circuit area ${fixed(voc * isc, 'watts')}. Fill factor ${fillFactor.toFixed(3)}.`);
    } else {
      const input = $('operating-voltage');
      point = selected.atVoltage(selected.voc * input.valueAsNumber / 100);
      showInput(input, fixed(point.voltage, 'V'));
      $('result').innerHTML = `P<sub>DC</sub> = ${fixed(point.power, 'W')}`;
      $('equations').innerHTML = `<p>P<sub>DC</sub> = VI ≈ ${fixed(point.voltage, 'V')} × ${fixed(point.current, 'A')} ≈ <strong>${fixed(point.power, 'W')}</strong></p>`;
      svg.setAttribute('aria-label', `Operating point at ${fixed(point.voltage, 'volts')} and ${fixed(point.current, 'amperes')}. DC power ${fixed(point.power, 'watts')}.`);
    }
    rectangle(point.voltage, point.current, '#a9581c');
    axes.series(selected.samples.map(([volts, amps]) => [volts, amps]), '#237f9b');
    axes.dot(point.voltage, point.current, '#a9581c', 6);
  }

  mountControls(isFillFactor ? irradianceControls : [['operating-voltage', 'Operating voltage', 0, 100, 1, 72]], update);
  update();
}
