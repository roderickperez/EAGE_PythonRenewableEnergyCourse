import { demos } from './equation-demos.js';
import { renderDarcyPressure, renderEquipmentPlot, renderEquationVisual } from './equation-visuals.js';
import { mountControls, showInput } from '../solar-lab/resource-controls.js';
import { updateMotionControl } from '../motion-control.js';

const $ = id => document.getElementById(id);
const requested = new URLSearchParams(location.search).get('demo');
const name = Object.hasOwn(demos, requested) ? requested : 'heat';
const demo = demos[name];
const drawing = $('drawing');
const button = $('motion');
const plot = $('equation-plot');
const host = $('model');
const modelKey = $('model-key');
const sceneKeys = {
  darcy: [['#247fae','Inlet'],['#bd522e','Producer'],['#579db0','Pore paths']],
  heat: [['#bd522e','Produced brine'],['#247fae','Cooled return'],['#279084','Separate heat recovery circuit']],
  carnot: [['#bd522e','Hot reservoir'],['#247fae','Cold reservoir'],['#267762','Work out']],
  pump: [['#247fae','Suction inlet'],['#23718c','Discharge outlet'],['#718f96','Motor and impeller']],
  storage: [['#baa183','Rock grains'],['#3296b3','Water-filled pores']],
};
const supportsScene = Object.hasOwn(sceneKeys,name);
let motion = true, scene, inputs, values, sceneObserver;

$('title').textContent = document.title = demo.title;
$('note').textContent = demo.note;
button.hidden = !supportsScene;
plot.closest('figure').hidden = !supportsScene;
if (supportsScene) document.querySelector('.explorer-grid').classList.add('has-scene');
if (name==='darcy') {
  $('plot-note').hidden = false;
  $('plot-note').textContent = 'Illustrative 1 m interval. Outlet pressure is the reference.';
}

function describeScene(input,result) {
  const descriptions = {
    darcy: `Path width shows relative permeability. Dot speed shows relative flux.`,
    heat: `Brine enters at ${input.reservoir} °C and returns at ${input.reinjection} °C. Heat passes through the exchanger wall into the separate circuit.`,
    carnot: `The ${(input.reservoir+273.15).toFixed(2)} K reservoir supplies heat. The engine delivers work and rejects heat to the ${(input.sink+273.15).toFixed(2)} K reservoir.`,
    pump: `The motor turns the impeller. It adds ${input.pressure} MPa at ${result.volumeFlow?.toFixed(4)} m³/s.`,
    storage: `${input.porosity}% of the bulk volume is pore space. The blue pockets show where water can occupy space between grains.`,
  };
  const description = descriptions[name];
  host.setAttribute('aria-label',description);
  modelKey.innerHTML = '<ul>' + sceneKeys[name].map(([color,label])=>
    `<li><i style="background:${color}" aria-hidden="true"></i>${label}</li>`).join('') +
    `</ul><p>${description} Drag to orbit.</p>`;
}

function update() {
  const input = Object.fromEntries(demo.controls.map(([id,,,,,,unit]) => {
    const el = $(id);
    showInput(el, `${el.value} ${unit}`.trim());
    return [id, el.valueAsNumber];
  }));
  const result = demo.calculate(input);
  $('result').textContent = result.headline;
  $('equations').innerHTML = `<p>${result.equation}</p>`;
  inputs = input;
  values = result.values;
  if (scene) {
    scene.update(input, values);
    describeScene(input, values);
  } else renderEquationVisual(name, drawing, input, values, motion);
  if (name==='darcy') renderDarcyPressure(plot,input.pressureGradient);
  else if (supportsScene) renderEquipmentPlot(name,plot,input,values);
}

button.addEventListener('click', () => {
  motion = !motion;
  updateMotionControl(button, motion);
  scene?.setPlaying(motion);
  if (motion) drawing.unpauseAnimations();
  else drawing.pauseAnimations();
});
updateMotionControl(button, motion);
mountControls(demo.controls, update);
update();

if (supportsScene) {
  sceneObserver = new IntersectionObserver(async ([entry]) => {
    if (!entry.isIntersecting) return;
    sceneObserver.disconnect();
    try {
      const { mountEquationScene } = await import('./equation-scenes.js?v=1');
      scene = mountEquationScene(host, name);
      scene.update(inputs, values);
      scene.setPlaying(motion);
      describeScene(inputs,values);
      drawing.setAttribute('hidden','');
      host.hidden = modelKey.hidden = false;
    } catch (error) {
      scene?.destroy();
      scene = undefined;
      if (name==='storage') button.hidden = true;
      console.warn('Could not open the 3D view. Using the equation diagram.',error);
    }
  });
  sceneObserver.observe(drawing);
  window.addEventListener('pagehide',event=>{
    if (event.persisted) return;
    sceneObserver.disconnect();
    scene?.destroy();
  });
}

const frame = window.frameElement;
if (frame) {
  const main = document.querySelector('main');
  new ResizeObserver(() => {
    frame.style.height = Math.ceil(main.getBoundingClientRect().height + 4) + 'px';
  }).observe(main);
}
