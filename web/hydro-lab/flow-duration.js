import { flowDuration } from './balance-model.js';
import { addPlot, mountPlotGroup, setupPlot } from './comparison-plots.js';

const dailyFlows = [21, 19, 22, 28, 38, 57, 68, 60, 48, 38, 32, 27, 23, 20, 17, 15, 14];
const flowMax = 80;
const highlight = '#a9581c';

export function mountFlowDuration() {
  const $ = id => document.getElementById(id);
  $('title').textContent = document.title = 'From hydrograph to flow-duration curve';
  $('result').hidden = true;
  for (const id of ['scene', 'legend', 'equations']) $(id).remove();
  const ranked = flowDuration(dailyFlows);
  const observationsByDay = new Map(ranked.map(observation => [observation.day, observation]));
  const plots = mountPlotGroup(document.querySelector('header'), 'Linked discharge plots');
  const chronological = addPlot(plots, 'Hydrograph', 'Daily discharge in time order.');
  const duration = addPlot(plots, 'Flow-duration curve', 'The same observations ranked from highest to lowest flow.');

  function plot(svg, points, xMin, xMax, xLabel, color) {
    const chart = setupPlot(svg, xMax, flowMax, xLabel, 'Discharge (m³/s)', xMin);
    chart.series(points, color);
    const horizontal = chart.line(chart.x(xMin), chart.y(0), chart.x(xMax), chart.y(0), highlight, '4 3');
    const vertical = chart.line(chart.x(xMin), chart.y(0), chart.x(xMin), chart.y(0), highlight, '4 3');
    const marker = chart.dot(xMin, 0, highlight, 6);
    return (xValue, flow, label) => {
      const x = chart.x(xValue);
      const y = chart.y(flow);
      horizontal.setAttribute('y1', y);
      horizontal.setAttribute('y2', y);
      vertical.setAttribute('x1', x);
      vertical.setAttribute('x2', x);
      vertical.setAttribute('y2', y);
      marker.setAttribute('cx', x);
      marker.setAttribute('cy', y);
      svg.setAttribute('aria-label', label);
    };
  }

  const showDay = plot(chronological, dailyFlows.map((flow, index) => [index + 1, flow]), 1, dailyFlows.length, 'Day', '#237f9b');
  const showRank = plot(duration, ranked.map(({ exceedancePercent, flow }) => [exceedancePercent, flow]), 0, 100, 'Exceedance plotting position (%)', '#398470');

  const control = document.createElement('div');
  control.className = 'control flow-duration-control';
  control.innerHTML = `<label for="flow-day">Select an observation by day</label>
    <input type="range" id="flow-day" min="1" max="${dailyFlows.length}" value="7" step="1">
    <p class="flow-duration-summary" id="flow-selection"></p>`;
  plots.after(control);
  $('controls').hidden = true;
  $('note').textContent = 'Synthetic daily flows. The duration curve uses 100 × r/(n + 1), where r is the descending rank and n is the sample count. The earlier day is ranked first when flows tie, so each observation can be tracked separately.';
  const input = $('flow-day');

  function update() {
    const { day, flow, rank, exceedancePercent } = observationsByDay.get(input.valueAsNumber);
    const position = `${exceedancePercent.toFixed(1)}%`;
    const selected = `Day ${day} · ${flow} m³/s`;
    const rankedPosition = `Rank ${rank} of ${ranked.length} · ${position} exceedance plotting position`;
    $('flow-selection').textContent = `${selected} · ${rankedPosition}`;
    input.setAttribute('aria-valuetext', `${selected}. ${rankedPosition}.`);
    showDay(day, flow, `Hydrograph. Selected ${selected}.`);
    showRank(exceedancePercent, flow, `Flow-duration curve. Selected ${selected}. ${rankedPosition}.`);
  }
  input.addEventListener('input', update);
  update();
}
