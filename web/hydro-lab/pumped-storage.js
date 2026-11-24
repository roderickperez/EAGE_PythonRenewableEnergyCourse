import { MAX_CYCLE_VOLUME_M3, pumpedCycle, pumpedStorage } from './balance-model.js';
import { mountBalanceRenderer, palette } from './balance-scene.js';
import { mountDiagramMotion } from './diagram-motion.js';
import { HYDRO_CONSTANTS } from './model.js';

export function mountPumpedStorage() {
  const $ = id => document.getElementById(id);
  const n = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
  const value = number => n.format(number);
  const energy = number => `${value(number)} MWh`;
  const { densityKgM3, gravityMS2 } = HYDRO_CONSTANTS;
  $('title').textContent = document.title = 'Pumped storage energy cycle';
  $('result').setAttribute('aria-label', 'Theoretical round-trip efficiency');
  $('drawing').setAttribute('aria-label', 'Upper and lower reservoirs linked by a reversible pump-turbine. Blue particles follow the water, and gold pulses show the direction of energy transfer between the machine and the grid.');
  $('note').textContent = 'Playback compresses time. The energy balance uses constant head, the same water volume in each direction, ρ = 1,000 kg/m³ and g = 9.81 m/s². Water levels show the transferred volume schematically. The displayed theoretical efficiency is the product of the stage efficiencies; when pumping input is zero, an observed output-to-input ratio is undefined.';

  const phaseLine = document.createElement('p');
  phaseLine.className = 'storage-phase';
  const phaseName = document.createElement('strong');
  const upperVolume = document.createElement('span');
  phaseLine.append(phaseName, upperVolume);
  $('scene').after(phaseLine);
  for (const [color, label] of [[palette.water, 'Water transfer'], [palette.shaft, 'Reversible pump-turbine and motor-generator'], [palette.electricity, 'Electrical energy transfer']]) {
    const item = document.createElement('span');
    item.style.setProperty('--swatch', color);
    item.textContent = label;
    $('legend').append(item);
  }
  const equations = Array.from({ length: 4 }, () => document.createElement('div'));
  $('equations').append(...equations);
  const sliders = [
    ['volume', 'Cycle water volume <i>V</i>', 'm³', 0, MAX_CYCLE_VOLUME_M3, 5000, 120000],
    ['head', 'Constant head <i>H</i>', 'm', 0, 250, 1, 80],
    ['pump', 'Combined motor, drive and pump efficiency <i>η<sub>p</sub></i>', '%', 10, 100, 1, 85],
    ['turbine', 'Turbine efficiency <i>η<sub>t</sub></i>', '%', 10, 100, 1, 90],
    ['generator', 'Generator efficiency <i>η<sub>g</sub></i>', '%', 10, 100, 1, 95],
  ].map(([id, label, unit, min, max, step, initial]) => {
    const control = document.createElement('div');
    control.className = 'control';
    control.innerHTML = `<label for="${id}"><span>${label}</span><output for="${id}"></output></label>
      <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${initial}">`;
    $('controls').append(control);
    return { input: control.querySelector('input'), output: control.querySelector('output'), unit };
  });

  let render;
  try {
    render = mountBalanceRenderer($('scene'), 'pumped-storage');
  } catch {
    $('fallback').hidden = false;
  }
  const active = ([volume, head]) => volume > 0 && head > 0;
  const draw = mountDiagramMotion($('scene'), $('motion'), (inputs, phase) => {
    const running = active(inputs);
    const { pumping, upper } = pumpedCycle(running ? phase : 0);
    const label = running ? (pumping ? 'Pumping · grid to upper reservoir' : 'Generating · upper reservoir to grid') : 'No active cycle';
    const amount = ` · ${value(Math.round(upper * inputs[0]))} m³ transferred to the upper reservoir`;
    if (phaseName.textContent !== label) phaseName.textContent = label;
    if (upperVolume.textContent !== amount) upperVolume.textContent = amount;
    render?.(inputs, phase);
  }, inputs => active(inputs) ? 1 / 16000 : 0);

  function update() {
    const [volume, head, pump, turbine, generator] = sliders.map(({ input, output, unit }) => {
      const number = input.valueAsNumber;
      output.value = `${value(number)} ${unit}`;
      input.setAttribute('aria-valuetext', output.value);
      return number;
    });
    const { stored, input, recovered, roundTrip } = pumpedStorage(volume, head, pump / 100, turbine / 100, generator / 100, densityKgM3, gravityMS2);
    $('result').innerHTML = `Theoretical η<sub>RT</sub> = ${value(roundTrip * 100)}%`;
    equations[0].innerHTML = `Stored at the top · E<sub>pot</sub> = ρgVH = <strong>${energy(stored)}</strong>`;
    equations[1].innerHTML = `Pumping input · E<sub>pump</sub> = E<sub>pot</sub> / ${value(pump / 100)} = <strong>${energy(input)}</strong>`;
    equations[2].innerHTML = `Recovered electricity · E<sub>gen</sub> = ${value(turbine / 100)} × ${value(generator / 100)} × E<sub>pot</sub> = <strong>${energy(recovered)}</strong>`;
    equations[3].innerHTML = `Full-cycle loss · E<sub>pump</sub> − E<sub>gen</sub> = <strong>${energy(input - recovered)}</strong>`;
    draw([volume, head]);
  }
  for (const { input } of sliders) input.addEventListener('input', update);
  update();
}
