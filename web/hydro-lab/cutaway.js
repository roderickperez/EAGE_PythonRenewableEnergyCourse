import artwork from './plant-cutaway.svg';

// Component target, followed by the label position in illustration coordinates.
const callouts = [
  ['Reservoir', 61, 151, 20, 79],
  ['Trash rack', 128, 219, 20, 182],
  ['Control gate', 161, 163, 193, 158],
  ['Intake', 142, 240, 22, 257],
  ['Penstock', 277, 302, 168, 365],
  ['Dam', 383, 182, 301, 134],
  ['Spillway / bypass', 565, 192, 523, 105],
  ['Powerhouse', 741, 392, 730, 305],
  ['Turbine', 638, 473, 683, 503],
  ['Generator', 544, 382, 446, 386],
  ['Draft tube', 563, 544, 439, 554],
  ['Tailrace / outflow', 838, 553, 810, 510],
  ['Transformer', 502, 314, 371, 279],
  ['Transmission', 827, 118, 762, 68],
];

// Invisible hit regions follow the visible components rather than label positions.
const regions = [
  'M0 110H118V290H0Z', 'M110 200h37v81h-37Z', 'M151 132h22v137h-22Z',
  'M173 215h58v48h-58Z', 'M228 247 391 410 495 438v37l-134-29L209 267Z',
  'M175 120 361 104 418 217 506 377 398 398 260 204Z',
  'M415 78 472 76 879 386 854 408 737 368Z',
  'M650 310 864 358v70l-116 51-100-37Z',
  'M516 451H660V505H516Z', 'M524 322H646V409H524Z',
  'M545 509h80v26h137v62H645l-100-34Z',
  'M817 434H1000V610H817Z', 'M483 278h51v66h-51Z',
  'M778 90H960V243H778Z',
];

export function cutawayMarkup() {
  const targets = callouts.map(([name, x, y], index) => {
    const width = name.length * 10 + 24;
    const left = Math.max(8, Math.min(992 - width, x - width / 2));
    const top = Math.max(8, y - 44);
    return `<g class="component-target" tabindex="0" role="button" aria-label="${name}" aria-pressed="false">
      <path class="component-hit" d="${regions[index]}"/>
      <g class="component-tooltip" aria-hidden="true" transform="translate(${left} ${top})">
        <rect width="${width}" height="31" rx="5"/>
        <text x="12" y="21">${name}</text>
      </g>
    </g>`;
  }).join('');
  return artwork.replace('role="img"', 'role="group"').replace('</svg>', `${targets}</svg>`);
}

export function mountCutaway(root) {
  const diagram = root.querySelector('.plant-cutaway');
  const targets = [...diagram.querySelectorAll('.component-target')];
  function select(target) {
    const selected = target?.getAttribute('aria-pressed') !== 'true';
    targets.forEach((item) => item.setAttribute('aria-pressed', String(item === target && selected)));
  }
  diagram.addEventListener('click', (event) => select(event.target.closest('.component-target')));
  diagram.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') select(null);
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      select(event.target.closest('.component-target'));
    }
  });
  const lake = diagram.querySelector('#cutaway-lake');
  const spill = diagram.querySelector('#cutaway-spill');
  const spillMotion = diagram.querySelector('#cutaway-spill-motion');
  const generationMotion = diagram.querySelector('#cutaway-generation-motion');
  const arrows = diagram.querySelector('#cutaway-direction');
  return (result) => {
    const fraction = (result.inputs.grossHeadM - 5) / 245;
    lake.setAttribute('transform', `translate(0 ${157 - fraction * 81})`);
    spill.style.opacity = result.spillFlowM3s > 0 ? String(Math.min(1, 0.35 + result.spillFlowM3s / 15)) : '0';
    spillMotion.style.opacity = result.spillFlowM3s > 0 ? '1' : '0';
    generationMotion.style.opacity = result.turbineFlowM3s > 0 ? '1' : '0';
    arrows.style.opacity = generationMotion.style.opacity;
    diagram.style.setProperty('--turbine-speed', `${4 / (1 + Math.sqrt(result.turbineFlowM3s))}s`);
    diagram.style.setProperty('--spill-speed', `${3 / (1 + Math.sqrt(result.spillFlowM3s))}s`);
    diagram.style.setProperty('--grid-active', result.plantMW > 0 ? '#f5dc8c' : '#354e49');
  };
}
