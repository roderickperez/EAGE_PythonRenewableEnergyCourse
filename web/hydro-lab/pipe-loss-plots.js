import { FITTING_POSITION, MAX_PIPE_FLOW_M3S, pipeHeadProfile, pipeLoss } from './balance-model.js';

const ns = 'http://www.w3.org/2000/svg';
const colors = { pipe: '#237f9b', friction: '#ba722e', fittings: '#8a645f', head: '#398470' };

function svgNode(tag, attrs, content) {
  const el = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  if (content !== undefined) el.textContent = content;
  return el;
}

export function mountPipeLossPlots(anchor, gravity) {
  const container = document.createElement('section');
  container.className = 'pipe-loss-plots';
  container.setAttribute('aria-label', 'Hydraulic loss plots');
  function card(title, caption) {
    const figure = document.createElement('figure');
    const heading = document.createElement('h2');
    heading.textContent = title;
    const svg = svgNode('svg', { viewBox: '0 0 410 194', role: 'img', 'aria-label': title });
    const description = document.createElement('figcaption');
    description.textContent = caption;
    figure.append(heading, svg, description);
    container.append(figure);
    return svg;
  }
  const profile = card('Pressure head along the pipe', 'The slope shows friction along the straight run. The step shows the loss assigned to the fitting.');
  const sweep = card('Head loss versus flow', 'Blue = total. Dashed orange = friction. Dotted brown = fittings. The dot marks the slider setting.');
  anchor.after(container);

  function setup(svg, xMax, yMax, xLabel, yLabel) {
    svg.replaceChildren();
    const plot = { left: 59, right: 393, top: 14, bottom: 149 };
    const x = value => plot.left + value / xMax * (plot.right - plot.left);
    const y = value => plot.bottom - value / yMax * (plot.bottom - plot.top);
    const tick = value => Number(value.toPrecision(3)).toLocaleString('en');
    function line(x1, y1, x2, y2, stroke, dash = '') {
      svg.append(svgNode('line', { x1, y1, x2, y2, stroke, 'stroke-dasharray': dash }));
    }
    function text(value, xx, yy, anchor = 'middle') {
      svg.append(svgNode('text', { x: xx, y: yy, 'text-anchor': anchor, fill: '#365951', 'font-size': 11 }, value));
    }
    for (let i = 0; i <= 4; i++) {
      const xx = x(xMax * i / 4);
      const yy = y(yMax * i / 4);
      line(plot.left, yy, plot.right, yy, '#d8e8e1');
      text(tick(xMax * i / 4), xx, plot.bottom + 15);
      text(tick(yMax * i / 4), plot.left - 7, yy + 4, 'end');
    }
    line(plot.left, plot.top, plot.left, plot.bottom, '#829d93');
    line(plot.left, plot.bottom, plot.right, plot.bottom, '#829d93');
    text(xLabel, (plot.left + plot.right) / 2, 185);
    svg.append(svgNode('text', { transform: 'translate(14 81) rotate(-90)', 'text-anchor': 'middle', fill: '#365951', 'font-size': 11 }, yLabel));
    function series(points, stroke, dash = '') {
      svg.append(svgNode('polyline', {
        points: points.map(([xx, yy]) => `${x(xx)},${y(yy)}`).join(' '),
        fill: 'none', stroke, 'stroke-width': 2.6, 'stroke-dasharray': dash, 'stroke-linejoin': 'round',
      }));
    }
    function dot(xx, yy, color) {
      svg.append(svgNode('circle', { cx: x(xx), cy: y(yy), r: 4.5, fill: color, stroke: 'white', 'stroke-width': 1.2 }));
    }
    return { x, y, line, series, dot };
  }

  return ({ flow, length, diameter, factor, fittings }, loss) => {
    const head = pipeHeadProfile(loss, length);
    const p = setup(profile, length, head[0][1] * 1.16, 'Distance along pipe (m)', 'Pressure head (m)');
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
    const q = setup(sweep, MAX_PIPE_FLOW_M3S, Math.max(1, samples.at(-1)[1].total * 1.08), 'Volume flow (m³/s)', 'Head loss (m)');
    for (const [term, color, dash] of [['total', colors.pipe, ''], ['friction', colors.friction, '6 4'], ['fittings', colors.fittings, '2 4']]) {
      q.series(samples.map(([rate, item]) => [rate, item[term]]), color, dash);
    }
    q.dot(flow, loss.total, colors.pipe);
    sweep.setAttribute('aria-label', `Loss versus flow. Selected flow ${flow} cubic metres per second, head loss ${loss.total.toFixed(3)} metres`);
  };
}
