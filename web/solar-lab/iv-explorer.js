import { addPlot, mountPlotGroup, setupPlot } from '../hydro-lab/comparison-plots.js';
import { moduleCurve } from './iv-model.js';

export function mountIvExplorer() {
  const $ = id => document.getElementById(id);
  $('title').textContent = document.title = 'Current, voltage and maximum power';
  $('scene').remove();
  const plots = mountPlotGroup(document.querySelector('header'), 'Photovoltaic module curves');
  plots.classList.add('iv-plots');
  const iv = addPlot(plots, 'Current–voltage (I–V)', 'Short circuit at 0 V. Open circuit at 0 A.');
  const pv = addPlot(plots, 'Power–voltage (P–V)', 'Power is zero at both endpoints. The peak marks maximum DC power.');
  const legend = [['#237f9b', 'Selected conditions'], ['#647c78', 'STC reference (dashed)'], ['#a9581c', 'Maximum power point']];
  for (const [color, label] of legend) {
    const item = document.createElement('span');
    item.style.setProperty('--swatch', color);
    item.textContent = label;
    $('legend').append(item);
  }
  $('controls').innerHTML = `
    <div class="control">
      <label for="irradiance"><span>Effective irradiance</span><output for="irradiance"></output></label>
      <input id="irradiance" type="range" min="0" max="1200" step="10" value="1000">
    </div>
    <div class="control">
      <label for="temperature"><span>Cell temperature</span><output for="temperature"></output></label>
      <input id="temperature" type="range" min="-10" max="85" step="1" value="25">
    </div>`;
  const irradiance = $('irradiance');
  const temperature = $('temperature');
  const reference = moduleCurve(1000, 25);
  $('note').textContent = 'Canadian Solar CS5P-220M, De Soto single-diode model. Effective irradiance is the light available to the cells. Cell temperature is set independently of irradiance. Axes stay fixed so operating points can be compared.';
  const fixed = (value, unit) => `${value.toFixed(2)} ${unit}`;

  function update() {
    const { peak, isc, voc, samples } = moduleCurve(irradiance.valueAsNumber, temperature.valueAsNumber);
    for (const [input, unit] of [[irradiance, 'W/m²'], [temperature, '°C']]) {
      const text = `${input.value} ${unit}`;
      input.previousElementSibling.querySelector('output').value = text;
      input.setAttribute('aria-valuetext', text);
    }
    $('result').innerHTML = peak ? `P<sub>mp</sub> = ${fixed(peak.power, 'W')}` : 'No generating point';
    $('equations').innerHTML = peak
      ? `<p>V<sub>mp</sub> = <strong>${fixed(peak.voltage, 'V')}</strong> · I<sub>mp</sub> = <strong>${fixed(peak.current, 'A')}</strong> · P<sub>mp</sub> = V<sub>mp</sub> × I<sub>mp</sub></p>
        <p>I<sub>sc</sub> = ${fixed(isc, 'A')} · V<sub>oc</sub> = ${fixed(voc, 'V')} · Fill factor = ${(peak.power / (isc * voc)).toFixed(3)}</p>`
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
  irradiance.addEventListener('input', update);
  temperature.addEventListener('input', update);
  update();
}
