import { turbineOperation } from './balance-model.js';
import { mountBalanceRenderer } from './balance-scene.js';
import { mountDiagramMotion } from './diagram-motion.js';
import { addPlot, mountPlotGroup, setupPlot } from './comparison-plots.js';

export const MAX_TURBINE_FLOW = 30;
export const MAX_TURBINE_HEAD = 220;
export const turbineFamilies = [
  { name: 'Pelton', efficiency: 82, color: '#965a35', dash: '', description: 'A jet strikes buckets on the runner.' },
  { name: 'Francis', efficiency: 90, color: '#247b96', dash: '9 5', description: 'Water enters radially and exits along the shaft.' },
  { name: 'Kaplan', efficiency: 87, color: '#357865', dash: '2 5', description: 'Water passes through a propeller runner.' },
];

export function mountTurbineComparison(scene, motion, anchor, density, gravity) {
  scene.classList.add('turbine-comparison');
  scene.querySelector('#drawing').remove();
  scene.querySelector('#fallback').remove();
  motion.classList.add('comparison-control');
  scene.previousElementSibling.append(motion);
  const cards = turbineFamilies.map(({ name, color, description }) => {
    const card = document.createElement('article');
    card.className = 'turbine-card';
    card.style.setProperty('--family', color);
    card.innerHTML = `<h2>${name}</h2><output aria-label="${name} plant electrical power"></output>
      <div class="turbine-viewport" role="img" aria-label="${name} runner and water flow"></div>
      <p>${description}</p><p class="turbine-flow"></p><p class="turbine-rating"></p>`;
    scene.append(card);
    const viewport = card.querySelector('.turbine-viewport');
    let draw;
    try {
      draw = mountBalanceRenderer(viewport, 'turbine-operation');
    } catch {
      viewport.textContent = 'Diagram unavailable';
    }
    return { draw, output: card.querySelector('output'), flow: card.querySelector('.turbine-flow'), rating: card.querySelector('.turbine-rating') };
  });
  if (cards.some(card => card.draw)) motion.hidden = false;
  const animate = mountDiagramMotion(scene, motion, (results, phase) => {
    results.forEach(({ turbineFlow, bypassFlow, outputMW }, index) => {
      cards[index].draw?.([index, turbineFlow, bypassFlow, outputMW], phase);
    });
  }, results => Math.max(...results.map(result => Math.max(result.turbineFlow, result.bypassFlow))) * .000045);

  const plots = mountPlotGroup(anchor, 'Turbine power comparison');
  const flowPlot = addPlot(plots, 'Plant power versus available flow', 'Net head stays at the slider setting. The dots show the selected flow.');
  const headPlot = addPlot(plots, 'Plant power versus net head', 'Available flow stays at the slider setting. Flat sections show the shared generator limit.');
  const legend = document.createElement('div');
  legend.className = 'legend';
  legend.setAttribute('aria-label', 'Turbine plot legend');
  turbineFamilies.forEach(({ name, color, dash }) => {
    const item = document.createElement('span');
    item.style.setProperty('--swatch', color);
    item.textContent = `${name} (${dash === '' ? 'solid' : dash === '9 5' ? 'dashed' : 'dotted'})`;
    legend.append(item);
  });
  plots.before(legend);

  return ({ flow, head, rating, efficiencies }) => {
    const evaluate = (q, h, eta) => turbineOperation(q, h, eta, rating, density, gravity);
    const results = efficiencies.map(eta => evaluate(flow, head, eta));
    results.forEach(({ turbineFlow, bypassFlow, outputMW, ratedFlow }, index) => {
      cards[index].output.innerHTML = `P<sub>plant</sub> = ${outputMW.toFixed(3)} MW`;
      cards[index].flow.textContent = `Through runner ${turbineFlow.toFixed(2)} · Bypass ${bypassFlow.toFixed(2)} m³/s`;
      cards[index].rating.textContent = ratedFlow === null ? 'Rated flow unavailable at zero head or efficiency' : `Flow at generator rating: ${ratedFlow.toFixed(2)} m³/s`;
    });
    animate(results);

    function plot(svg, xLabel, max, selected, at, threshold) {
      const top = Math.max(.1, ...efficiencies.map(eta => at(max, eta))) * 1.12;
      const axes = setupPlot(svg, max, top, xLabel, 'Plant electrical power (MW)');
      efficiencies.forEach((eta, index) => {
        const { color, dash } = turbineFamilies[index];
        const cutoff = threshold(eta);
        const points = cutoff > 0 && cutoff < max ? [0, cutoff, max] : [0, max];
        axes.series(points.map(x => [x, at(x, eta)]), color, dash, 4 - index * .8);
      });
      results.forEach((point, index) => axes.dot(selected, point.outputMW, turbineFamilies[index].color, 7 - index * 1.8));
      svg.setAttribute('aria-label', `${xLabel}. ${results.map((result, index) =>
        `${turbineFamilies[index].name} ${result.outputMW.toFixed(3)} megawatts`).join(', ')} at the selected setting.`);
    }
    plot(flowPlot, 'Available flow (m³/s)', MAX_TURBINE_FLOW, flow,
      (q, eta) => evaluate(q, head, eta).outputMW, eta => evaluate(flow, head, eta).ratedFlow);
    plot(headPlot, 'Net head (m)', MAX_TURBINE_HEAD, head,
      (h, eta) => evaluate(flow, h, eta).outputMW,
      eta => eta > 0 && flow > 0 ? rating * 1e6 / (eta * density * gravity * flow) : null);
  };
}
