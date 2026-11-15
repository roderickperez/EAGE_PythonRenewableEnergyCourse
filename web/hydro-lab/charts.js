const NS = 'http://www.w3.org/2000/svg';

function node(name, attributes = {}, text = '') {
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text) element.textContent = text;
  return element;
}

function clear(svg) {
  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 270');
}

function horizontalAxis(svg, x1, x2, y, label) {
  svg.append(node('line', { x1, y1: y, x2, y2: y, class: 'axis' }));
  svg.append(node('text', { x: x2, y: y + 27, class: 'axis-label', 'text-anchor': 'end' }, label));
}

export function renderPowerChart(svg, result) {
  clear(svg);
  const plot = { left: 58, right: 575, top: 22, bottom: 224 };
  const qMax = 30;
  const slope = result.netHeadM * 0.85 * 1000 * 9.81 / 1e6;
  const yMax = Math.max(1, result.inputs.ratingMW * 1.18, slope * qMax * 1.05);
  const x = (q) => plot.left + q / qMax * (plot.right - plot.left);
  const y = (mw) => plot.bottom - mw / yMax * (plot.bottom - plot.top);
  for (let i = 0; i <= 4; i += 1) {
    const value = yMax * i / 4;
    const yy = y(value);
    svg.append(node('line', { x1: plot.left, y1: yy, x2: plot.right, y2: yy, class: 'grid' }));
    svg.append(node('text', { x: plot.left - 9, y: yy + 4, class: 'tick', 'text-anchor': 'end' }, value.toFixed(value < 10 ? 1 : 0)));
  }
  for (let i = 0; i <= 6; i += 1) {
    const value = qMax * i / 6;
    svg.append(node('text', { x: x(value), y: plot.bottom + 19, class: 'tick', 'text-anchor': 'middle' }, value.toFixed(0)));
  }
  horizontalAxis(svg, plot.left, plot.right, plot.bottom, 'Usable flow Q (m³/s)');
  svg.append(node('text', { x: 16, y: plot.top, class: 'axis-label', transform: `rotate(-90 16 ${plot.top})`, 'text-anchor': 'end' }, 'Power (MW)'));
  svg.append(node('path', { d: `M ${x(0)} ${y(0)} L ${x(qMax)} ${y(slope * qMax)}`, class: 'curve uncapped' }));
  svg.append(node('line', { x1: plot.left, y1: y(result.inputs.ratingMW), x2: plot.right, y2: y(result.inputs.ratingMW), class: 'curve cap' }));
  svg.append(node('circle', { cx: x(result.inputs.usableFlowM3s), cy: y(result.plantMW), r: 7, class: 'live-point' }));
  svg.append(node('text', { x: x(result.inputs.usableFlowM3s) + 10, y: y(result.plantMW) - 11, class: 'live-label' }, `${result.plantMW.toFixed(2)} MW`));
  svg.append(node('text', { x: plot.left + 8, y: y(result.inputs.ratingMW) - 7, class: 'legend cap-text' }, `Rating cap ${result.inputs.ratingMW.toFixed(1)} MW`));
  svg.append(node('text', { x: plot.right - 6, y: Math.max(plot.top + 13, y(slope * qMax) + 17), class: 'legend', 'text-anchor': 'end' }, 'Uncapped ηρgQHₙₑₜ'));
}

export function renderConversionChart(svg, result) {
  clear(svg);
  const values = [
    ['Hydraulic resource', result.hydraulicMW, 'hydraulic'],
    ['After efficiency', result.uncappedMW, 'uncapped'],
    ['Plant export', result.plantMW, 'plant'],
  ];
  const maximum = Math.max(1, ...values.map(([, value]) => value));
  const left = 160;
  const width = 390;
  values.forEach(([label, value, className], index) => {
    const y = 40 + index * 57;
    svg.append(node('text', { x: left - 12, y: y + 20, class: 'bar-label', 'text-anchor': 'end' }, label));
    svg.append(node('rect', { x: left, y, width, height: 28, rx: 7, class: 'bar-track' }));
    svg.append(node('rect', { x: left, y, width: Math.max(2, value / maximum * width), height: 28, rx: 7, class: `bar ${className}` }));
    svg.append(node('text', { x: Math.min(560, left + value / maximum * width + 8), y: y + 20, class: 'bar-value' }, `${value.toFixed(2)} MW`));
  });
  svg.append(node('rect', { x: 28, y: 218, width: 544, height: 40, rx: 10, class: 'energy-band' }));
  svg.append(node('text', { x: 45, y: 244, class: 'energy-label' }, `Six-hour energy: ${result.energy6hMWh.toFixed(2)} MWh = ${result.plantMW.toFixed(2)} MW × 6 h`));
}
