import { fitFrameToContent } from './fit-frame.js';
import { DiversionScene, ReservoirWorkedScene } from './worked-scenes.js';
import { diversionCase, diversionSiteHead, reservoirCase } from './worked-model.js';
import { energyPlot } from './worked-energy.js';
import styles from './worked-examples.css';

const diversionSliders = [
  ['width', 'Channel width', 'm', 6, 18, .1, 12],
  ['depth', 'Water depth', 'm', 1, 5, .1, 4.5],
  ['slope', 'Channel slope', '%', .002, .05, .001, .01],
  ['flow', 'Mean river flow', 'm³/s', 10, 100, 1, 60],
  ['precip', 'Annual precipitation', 'mm', 200, 2000, 10, 600],
  ['runoff', 'Runoff coefficient', '', .1, 1, .01, .4],
];
const reservoirSliders = [
  ['area', 'Catchment area', 'km²', 50, 400, 5, 200],
  ['rain', 'Annual precipitation', 'mm', 400, 2000, 10, 1200],
  ['coefficient', 'Runoff coefficient', '', .1, 1, .01, .6],
  ['meanhead', 'Mean net head', 'm', 40, 100, 1, 80],
  ['designhead', 'Design net head', 'm', 80, 150, 1, 100],
  ['eco', 'Environmental release', 'm³/s', 0, 3, .1, 1],
];

const fmt = (number, digits = 2) => number.toFixed(digits);
const root = document.querySelector('#root');
root.innerHTML = `<style>${styles}</style><div class="wrap grid"></div>`;
const grid = root.querySelector('.grid');

function makeExample({ title, description, sliders, calculate, Scene, initialAxis, details, results, plotNote, showTotal }) {
  const section = document.createElement('section');
  section.className = 'example';
  section.innerHTML = `<h2>${title}</h2><p class="note">${description}</p>
    <div class="example-body">
      <div class="visualization">
        <div class="scene" role="group" aria-label="Interactive 3D view of the ${title.toLowerCase()}"></div>
        <p class="scene-details note"></p>
      </div>
      <div class="controls">${sliders.map(([id, label, unit, min, max, step, value]) => `
        <label for="${id}"><span>${label}<output for="${id}"></output></span>
        <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${label}${unit ? ` (${unit})` : ''}"></label>`).join('')}</div>
      <div class="plot">
        <div class="plot-title"><h3></h3><output class="energy"></output></div>
        ${showTotal ? '<div class="plot-key"><span>━ Main plant</span><span>╌ Main + ecological</span></div>' : ''}
        <canvas role="img"></canvas>
        <p class="note"></p>
      </div>
    </div>
    <div class="results"></div>`;
  grid.append(section);
  const sceneHost = section.querySelector('.scene');
  let scene;
  try {
    scene = new Scene(sceneHost, { reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches });
  } catch {
    sceneHost.innerHTML = '<p class="note">The 3D view needs WebGL. The energy plot and calculations are still available.</p>';
  }
  const canvas = section.querySelector('.plot canvas');
  const controls = new Map(sliders.map(definition => [definition[0], definition]));
  const inputs = Object.fromEntries(sliders.map(([id, , , , , , value]) => [id, value]));
  const drawPlot = new ResizeObserver(() => renderPlot());
  let axis = initialAxis;

  function renderPlot() {
    const activeSlider = controls.get(axis);
    const selected = calculate(inputs).energy;
    section.querySelector('.plot h3').textContent = `Annual energy vs. ${activeSlider[1].toLowerCase()}`;
    section.querySelector('.energy').value = `${fmt(selected)} GWh/year`;
    section.querySelector('.plot p').textContent = plotNote(axis);
    const scenario = value => calculate({ ...inputs, [axis]: value });
    energyPlot(canvas, activeSlider, inputs[axis], value => scenario(value).energy, selected,
      showTotal ? value => scenario(value).totalEnergy : undefined);
  }

  function render() {
    const data = calculate(inputs);
    for (const [id, , unit] of sliders) {
      section.querySelector(`output[for="${id}"]`).value = unit ? `${inputs[id]} ${unit}` : String(inputs[id]);
    }
    scene?.updateWorked(data);
    section.querySelector('.scene-details').textContent = details(data);
    section.querySelector('.results').innerHTML = results(data);
    renderPlot();
  }

  const onSlider = event => {
    axis = event.target.id;
    inputs[axis] = event.target.valueAsNumber;
    render();
  };
  section.addEventListener('input', onSlider);
  section.addEventListener('focusin', event => {
    if (controls.has(event.target.id) && axis !== event.target.id) {
      axis = event.target.id;
      renderPlot();
    }
  });
  drawPlot.observe(canvas);
  render();
}

makeExample({
  title: 'Diversion plant',
  description: 'Size a concrete channel and two penstocks for a 50 MW plant.',
  sliders: diversionSliders,
  Scene: DiversionScene,
  calculate: diversionCase,
  initialAxis: 'flow',
  details: ({ operatingFlow, riverFlow, area }) => `Channel: ${fmt(operatingFlow)} m³/s at ${fmt(operatingFlow / area)} m/s · Bypass river: ${fmt(riverFlow - operatingFlow)} m³/s. Drag to rotate the valley.`,
  results: data => `Catchment area: <b>${fmt(data.catchment, 0)} km²</b><br>
    Channel area: ${fmt(data.area)} m² · Hydraulic radius: ${fmt(data.radius)} m<br>
    Manning design velocity: ${fmt(data.velocity)} m/s · Channel capacity: <b>${fmt(data.channelFlow)} m³/s</b><br>
    Channel losses: ${fmt(data.channelLoss)} m · Penstock losses: ${fmt(data.pipeLoss)} m · Total: <b>${fmt(data.channelLoss + data.pipeLoss)} m</b><br>
    For 50 MW electrical: net head ${fmt(data.netHead)} m · Gross head ${fmt(data.grossHead)} m · Tailwater elevation ${fmt(200 - data.grossHead)} m<br>
    <small>Two equal pipes at 5 m/s at design flow. Manning n = 0.014 (channel), 0.012 (pipe). Intake and transition losses: 1 m and 0.5 m. Intake elevation: 200 m.</small>`,
  plotNote: axis => {
    const assumption = `Uses constant mean conditions, a fixed ${fmt(diversionSiteHead)} m gross head, the selected design losses, and a 50 MW plant limit.`;
    return axis === 'precip' || axis === 'runoff'
      ? `This slider changes the inferred catchment. Mean river flow stays fixed, so energy stays flat. ${assumption}`
      : assumption;
  },
});

makeExample({
  title: 'Reservoir plant',
  description: 'Allocate mean inflow to the main turbine, environmental release and spill.',
  sliders: reservoirSliders,
  Scene: ReservoirWorkedScene,
  showTotal: true,
  calculate: reservoirCase,
  initialAxis: 'eco',
  details: ({ mean, ecological, spill, usable }) => `Inflow: ${fmt(mean)} m³/s · Main turbine: ${fmt(usable)} · Environmental release: ${fmt(ecological)} · Spill: ${fmt(spill)} m³/s.`,
  results: data => `Annual runoff: <b>${fmt(data.annual / 1e6)} hm³</b> · Mean inflow: ${fmt(data.mean)} m³/s<br>
    Design flow: ${fmt(data.designFlow)} m³/s · Main usable mean flow: ${fmt(data.usable)} m³/s<br>
    Design shaft / generator / transformer: <b>${fmt(data.designShaft)} / ${fmt(data.designGenerator)} / ${fmt(data.designTransformer)} kW</b><br>
    Equivalent full-load hours: ${fmt(data.hours, 0)} h/year<br>
    Main generator / transformer / grid energy: ${fmt(data.generatorEnergy)} / ${fmt(data.transformerEnergy)} / <b>${fmt(data.gridEnergy)} GWh</b><br>
    Environmental flow served / shortfall: ${fmt(data.ecological)} / ${fmt(data.shortfall)} m³/s<br>
    Ecological group at design head: ${fmt(data.ecologicalPower)} kW · At mean head: ${fmt(data.ecologicalEnergy)} GWh/year at the transformer<br>
    <small>10% inflow spilled. Efficiencies: 91%, 95%, 99%. Line loss: 0.3%/km over 5 km. Environmental release is limited to water available after spill.</small>`,
  plotNote: axis => axis === 'designhead'
    ? 'Design head changes rated power. The annual energy estimate uses mean head, so energy stays flat.'
    : 'Both lines show grid energy. With the same head and efficiency, environmental release shifts generation between the turbines without changing their combined energy.',
});

fitFrameToContent(grid);
