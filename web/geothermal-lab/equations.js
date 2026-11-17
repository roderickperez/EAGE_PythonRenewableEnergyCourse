import { demos } from './equation-demos.js';
import { defaults, geothermal } from './model.js';
import { mountGeothermalScene } from './scene.js';
import { mountControls, showInput } from '../solar-lab/resource-controls.js';
import { updateMotionControl } from '../motion-control.js';

const $ = id => document.getElementById(id);
const requested = new URLSearchParams(location.search).get('demo');
const name = Object.hasOwn(demos, requested) ? requested : 'heat';
const demo = demos[name];
$('title').textContent = document.title = demo.title;
$('note').textContent = demo.note + ' The 3D plant is a conceptual illustration, not a scaled model of the equation.';
$('legend').textContent = 'Drag the scene to rotate. Hot fluid is orange; cooled fluid is blue.';
$('drawing').setAttribute('hidden', '');
const host = $('scene');
const tip = document.createElement('div');
tip.hidden = true; tip.style.cssText = 'position:absolute;z-index:2;background:#fff;padding:4px;border-radius:4px;font-size:12px;pointer-events:none';
host.append(tip);
const gauge = document.createElement('div');
gauge.style.cssText = 'margin:8px 0;height:12px;border-radius:6px;overflow:hidden;background:#dcebe7';
gauge.setAttribute('role','img');
const fill = document.createElement('div'); fill.style.cssText = 'height:100%;width:0;background:#237f9b;transition:width .2s'; gauge.append(fill);
$('equations').after(gauge);
const ranges = { darcy:.00009, gradient:500, heat:150, carnot:.5, conversion:30, pump:10, heatpump:30, storage:20000, decline:20, energy:175200 };
const message = document.createElement('p'); message.style.cssText = 'padding:14px;position:absolute;inset:0;pointer-events:none'; host.append(message);
let scene, isVisible = false, motion = !matchMedia('(prefers-reduced-motion: reduce)').matches, latest;
function update() {
  const input = Object.fromEntries(demo.controls.map(([id,,,,,,unit]) => {
    const el = $(id); showInput(el, `${el.value} ${unit}`.trim()); return [id, el.valueAsNumber];
  }));
  const result = demo.calculate(input);
  $('result').textContent = result.headline;
  $('equations').innerHTML = `<p>${result.equation}</p>`;
  fill.style.width = `${Math.min(100, Math.max(0, 100 * result.metric / ranges[name]))}%`;
  gauge.setAttribute('aria-label', `Relative result: ${result.headline}; bar uses a fixed scale for this example`);
  const sceneInput = { ...defaults, ...result.scene };
  latest = [sceneInput, geothermal(sceneInput), motion];
  scene?.update(...latest);
}
mountControls(demo.controls, update);
const button = $('motion');
function label() { updateMotionControl(button,motion,'animation'); }
button.addEventListener('click', () => { motion = !motion; label(); update(); });
label(); update();
// Create WebGL only while the card is near the viewport; a lesson can contain many cards.
const observer = new IntersectionObserver(([entry]) => {
  isVisible = entry.isIntersecting;
  if (isVisible && !scene) {
    try { scene = mountGeothermalScene(host,tip); message.textContent = ''; scene.update(...latest); }
    catch { message.textContent = '3D view unavailable; use the sliders and calculations below.'; }
  } else if (!isVisible && scene) { scene.destroy(); scene = undefined; }
}, { rootMargin: '150px' });
observer.observe(host);
window.addEventListener('pagehide', () => { observer.disconnect(); scene?.destroy(); });
if (window.frameElement) {
  new ResizeObserver(() => window.frameElement?.style.setProperty('--geothermal-equation-height', `${Math.ceil(document.querySelector('main').getBoundingClientRect().height)}px`)).observe(document.querySelector('main'));
}
