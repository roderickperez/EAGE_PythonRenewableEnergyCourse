import { FITTING_POSITION, MAX_PIPE_FLOW_M3S, pipeHeadProfile, pipeLoss } from './balance-model.js';

import { addPlot, mountPlotGroup, setupPlot } from './comparison-plots.js';

const colors = { pipe: '#237f9b', friction: '#ba722e', fittings: '#8a645f', head: '#398470' };

export function mountPipeLossPlots(anchor, gravity) {
  const container = mountPlotGroup(anchor, 'Hydraulic loss plots');
  const profile = addPlot(container, 'Pressure head along the pipe', 'The slope shows friction along the straight run. The step shows the loss assigned to the fitting.');
  const sweep = addPlot(container, 'Head loss versus flow', 'Blue = total. Dashed orange = friction. Dotted brown = fittings. The dot marks the slider setting.');

  return ({ flow, length, diameter, factor, fittings }, loss) => {
    const head = pipeHeadProfile(loss, length);
    const p = setupPlot(profile, length, head[0][1] * 1.16, 'Distance along pipe (m)', 'Pressure head (m)');
    p.line(p.x(length * FITTING_POSITION), 14, p.x(length * FITTING_POSITION), 149, '#9eaaa5', '3 4');
    p.series(head.slice(0, 2), colors.head);
    p.series(head.slice(1, 3), colors.fittings);
    p.series(head.slice(2), colors.head);
    p.dot(...head[0], colors.pipe);
    p.dot(...head.at(-1), colors.head);
    profile.setAttribute('aria-label', `Pressure head drops from ${head[0][1].toFixed(3)} to ${head.at(-1)[1].toFixed(3)} metres`);

    const samples = Array.from({ length: 33 }, (_, index) => {
      const q = MAX_PIPE_FLOW_M3S * index / 32;
      return [q, pipeLoss(q, length, diameter, factor, fittings, gravity)];
    });
    const q = setupPlot(sweep, MAX_PIPE_FLOW_M3S, Math.max(1, samples.at(-1)[1].total * 1.08), 'Volume flow (m³/s)', 'Head loss (m)');
    for (const [term, color, dash] of [['total', colors.pipe, ''], ['friction', colors.friction, '6 4'], ['fittings', colors.fittings, '2 4']]) {
      q.series(samples.map(([rate, item]) => [rate, item[term]]), color, dash);
    }
    q.dot(flow, loss.total, colors.pipe);
    sweep.setAttribute('aria-label', `Loss versus flow. Selected flow ${flow} cubic metres per second, head loss ${loss.total.toFixed(3)} metres`);
  };
}
