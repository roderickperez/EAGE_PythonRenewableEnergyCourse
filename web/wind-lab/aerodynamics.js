import { addPlot, mountPlotGroup } from '../hydro-lab/comparison-plots.js';
import { mountDiagramMotion } from '../hydro-lab/diagram-motion.js';
import { mountControls, mountLegend, showInput } from '../solar-lab/resource-controls.js';
import { solarSvgNode as node } from '../solar-lab/resource-scene.js';
import { energyDemos } from './energy-demos.js';
import { physicalDemos } from './physical-demos.js';
import { n, windColors } from './demo-plots.js';
import { mountWindScene } from './scene.js';

const $ = id => document.getElementById(id);
const key = new URLSearchParams(location.search).get('demo');
const hasScene = Object.hasOwn(physicalDemos, key);
const demo = hasScene ? physicalDemos[key] : Object.hasOwn(energyDemos, key) ? energyDemos[key] : energyDemos['annual-energy'];
$('title').textContent = document.title = demo.title;
$('note').textContent = demo.note;
mountLegend(demo.legend);
const plotGroup = mountPlotGroup(hasScene ? $('scene') : document.querySelector('header'), demo.plot);
const plot = addPlot(plotGroup, demo.plot, demo.caption);
const updateScene = hasScene ? mountSceneCard() : () => {};
if (!hasScene) $('scene').remove();

function update() {
  const inputs = Object.fromEntries(demo.controls.map(([id, , , , , , unit]) => {
    const input = $(id);
    const value = input.valueAsNumber;
    const text = unit === '°' ? `${n(value)}°` : [n(value), unit].filter(Boolean).join(' ');
    showInput(input, text);
    return [id, value];
  }));
  const result = demo.draw(plot, inputs);
  $('result').innerHTML = result.headline;
  $('equations').innerHTML = result.rows.map(row => `<p>${row}</p>`).join('');
  plot.setAttribute('aria-label', `${demo.plot}. ${$('result').textContent}`);
  updateScene(result);
}
mountControls(demo.controls, update);
update();

if (window.frameElement) {
  const frame = window.frameElement;
  const main = document.querySelector('main');
  new ResizeObserver(() => {
    frame.style.setProperty('--wind-equation-height', `${Math.ceil(main.getBoundingClientRect().height)}px`);
  }).observe(main);
}

function mountSceneCard() {
  const host = $('scene');
  const svg = $('drawing');
  const button = $('motion');
  host.style.height = 'clamp(215px, 33vw, 300px)';
  const label = 'Isometric wind turbine and swept rotor disk. Blue streaks represent wind. Gold pulses indicate electrical output in conversion demos.';
  svg.setAttribute('aria-label', label);
  const tooltip = document.createElement('div');
  tooltip.hidden = true;
  tooltip.style.cssText = 'position:absolute;z-index:2;pointer-events:none;background:#fffffff0;color:#31524f;padding:4px 6px;border-radius:5px;font-size:.74rem';
  host.append(tooltip);

  let scene, fallback, current, phase = 0, lastState, lastSpeed, lastPlaying;
  function useFallback() {
    scene?.destroy();
    scene = null;
    tooltip.hidden = true;
    svg.removeAttribute('hidden');
    fallback = mountWindFallback(svg);
  }
  try {
    scene = mountWindScene(host, tooltip, { showSweep: true });
    const canvas = host.querySelector('canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', label);
    canvas.dataset.orbit = 'true';
    canvas.style.cursor = 'grab';
    svg.setAttribute('hidden', '');
    canvas.addEventListener('webglcontextlost', () => {
      useFallback();
      if (current) setState(current);
    }, { once: true });
  } catch {
    useFallback();
  }

  function draw(state, nextPhase) {
    current = state;
    phase = nextPhase;
    const speed = state.atPhase ? state.atPhase(phase) : state.speed;
    const playing = button.title === 'Stop animation';
    const radius = state.radius ?? 40;
    const hub = state.hub ?? 90;
    const omega = state.stopped ? 0 : Math.min(3.2, 7 * speed / radius);
    if (scene) {
      if (state !== lastState || speed !== lastSpeed || playing !== lastPlaying) {
        scene.update({ hub, radius, direction: 0 }, { speed, omega, gridMW: state.electrical ?? 0,
          region: state.stopped ? 'Stopped' : 'Illustrative operating point' }, playing);
        lastState = state;
        lastSpeed = speed;
        lastPlaying = playing;
      }
    } else {
      fallback({ speed, radius, hub, stopped: state.stopped, electrical: state.electrical ?? 0 }, phase);
    }
  }
  const setState = mountDiagramMotion(document.querySelector('main'), button, draw, state =>
    state.atPhase ? 1 / 7000 : fallback && state.speed > 0 ? .00008 + Math.min(.00036, state.speed * .00002) : 0);
  button.addEventListener('click', () => { if (current) draw(current, phase); });
  return setState;
}

function mountWindFallback(svg) {
  const add = (tag, attributes) => node(svg, tag, attributes);
  add('polygon', { points: '122,180 221,153 444,181 344,221', fill: '#a0b397' });
  const tower = add('line', { x1: 257, x2: 257, y2: 184, stroke: '#788a93', 'stroke-width': 8 });
  const sweep = add('circle', { fill: '#d0e7ed', stroke: '#66a8ba', 'stroke-width': 1.4 });
  const blades = add('g', {});
  for (let i = 0; i < 3; i++) {
    node(blades, 'path', { d: 'M0,3 C-8,-8 -6,-34 1,-48 C5,-30 9,-7 3,3 Z',
      transform: `rotate(${i * 120})`, fill: '#e3e9eb', stroke: '#788a93', 'stroke-width': 1.25 });
  }
  node(blades, 'circle', { r: 6, fill: '#e3e9eb', stroke: '#788a93', 'stroke-width': 1.5 });
  add('polyline', { points: '260,183 309,184 367,158', fill: 'none', stroke: '#334752', 'stroke-width': 2.5 });
  add('rect', { x: 306, y: 173, width: 15, height: 14, fill: '#334752' });
  const air = Array.from({ length: 7 }, () => add('line', { stroke: windColors.wind, 'stroke-width': 2, opacity: .60 }));
  const pulses = Array.from({ length: 4 }, () => add('circle', { r: 3, fill: '#e4aa37' }));
  let previousPhase = 0;
  let angle = 0;
  return ({ speed, radius, hub, stopped, electrical }, phase) => {
    const y = 91 - (hub - 90) * .23;
    const scale = radius / 57;
    tower.setAttribute('y1', y);
    sweep.setAttribute('cx', 257);
    sweep.setAttribute('cy', y);
    sweep.setAttribute('r', 48 * scale);
    const delta = (phase - previousPhase + 1) % 1;
    previousPhase = phase;
    if (!stopped && speed > 0) angle = (angle + delta * 720) % 360;
    blades.setAttribute('transform', `translate(257 ${y}) rotate(${angle}) scale(${scale})`);
    air.forEach((line, i) => {
      const x = 145 + 213 * ((phase + i / 7) % 1);
      const yy = y + (i % 5 - 2) * 14;
      line.style.display = speed > 0 ? '' : 'none';
      for (const [attribute, value] of Object.entries({ x1: x, y1: yy, x2: x + 11, y2: yy })) line.setAttribute(attribute, value);
    });
    pulses.forEach((pulse, i) => {
      const progress = (phase + i / 4) % 1;
      pulse.style.display = electrical > 0 ? '' : 'none';
      pulse.setAttribute('cx', 260 + 107 * progress);
      pulse.setAttribute('cy', 184 - 26 * Math.max(0, (progress - .47) / .53));
    });
  };
}
