const ns = 'http://www.w3.org/2000/svg';

function svgNode(tag, attrs, content) {
  const el = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  if (content !== undefined) el.textContent = content;
  return el;
}

export function mountPlotGroup(anchor, label) {
  const container = document.createElement('section');
  container.className = 'comparison-plots';
  container.setAttribute('aria-label', label);
  anchor.after(container);
  return container;
}

export function addPlot(container, title, caption) {
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

export function setupPlot(svg, xMax, yMax, xLabel, yLabel) {
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
  function series(points, stroke, dash = '', width = 2.6) {
    svg.append(svgNode('polyline', {
      points: points.map(([xx, yy]) => `${x(xx)},${y(yy)}`).join(' '),
      fill: 'none', stroke, 'stroke-width': width, 'stroke-dasharray': dash, 'stroke-linejoin': 'round',
    }));
  }
  function dot(xx, yy, color, radius = 4.5) {
    svg.append(svgNode('circle', { cx: x(xx), cy: y(yy), r: radius, fill: color, stroke: 'white', 'stroke-width': 1.2 }));
  }
  return { x, y, line, series, dot };
}
