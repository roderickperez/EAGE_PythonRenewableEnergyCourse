import { demos } from './equation-demos.js';
import { renderEquationVisual } from './equation-visuals.js';
import { mountControls, showInput } from '../solar-lab/resource-controls.js';
import { updateMotionControl } from '../motion-control.js';

const $ = id => document.getElementById(id);
const requested = new URLSearchParams(location.search).get('demo');
const name = Object.hasOwn(demos, requested) ? requested : 'heat';
const demo = demos[name];
const drawing = $('drawing');
const button = $('motion');
let motion = !matchMedia('(prefers-reduced-motion: reduce)').matches;

$('title').textContent = document.title = demo.title;
$('note').textContent = demo.note;
button.hidden = name !== 'darcy';

function update() {
  const input = Object.fromEntries(demo.controls.map(([id,,,,,,unit]) => {
    const el = $(id);
    showInput(el, `${el.value} ${unit}`.trim());
    return [id, el.valueAsNumber];
  }));
  const result = demo.calculate(input);
  $('result').textContent = result.headline;
  $('equations').innerHTML = `<p>${result.equation}</p>`;
  renderEquationVisual(name, drawing, input, result.values, motion);
}

button.addEventListener('click', () => {
  motion = !motion;
  updateMotionControl(button, motion, 'flow animation');
  if (motion) drawing.unpauseAnimations();
  else drawing.pauseAnimations();
});
updateMotionControl(button, motion, 'flow animation');
mountControls(demo.controls, update);
update();

const frame = window.frameElement;
if (frame) {
  const main = document.querySelector('main');
  new ResizeObserver(() => {
    frame.style.height = Math.ceil(main.getBoundingClientRect().height + 4) + 'px';
  }).observe(main);
}
