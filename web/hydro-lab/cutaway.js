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

export function cutawayMarkup() {
  const svgLabels = callouts.map(([name, x, y, lx, ly], index) => {
    const start = lx < x ? Math.min(x - 8, lx + name.length * 10 + 5) : lx - 6;
    // Offset the clustered intake markers so they stay individually legible.
    const markerX = index === 3 ? x + 23 : x;
    const markerY = index === 3 ? y + 15 : y;
    return `<g class="cutaway-label">
      <path d="M${start} ${ly - 6}L${x} ${y}"/><circle cx="${x}" cy="${y}" r="3"/>
      <text x="${lx}" y="${ly}">${name}</text>
    </g><g class="cutaway-marker" transform="translate(${markerX} ${markerY})">
      <circle r="15"/><text y="1">${index + 1}</text></g>`;
  }).join('');
  const legend = callouts.map(([name], index) => `<li><span>${index + 1}</span>${name}</li>`).join('');
  return `${artwork.replace('</svg>', `<g aria-hidden="true">${svgLabels}</g></svg>`)}
    <ol class="cutaway-key" aria-label="Cutaway components">${legend}</ol>`;
}

export function mountCutaway(root) {
  const diagram = root.querySelector('.plant-cutaway');
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
