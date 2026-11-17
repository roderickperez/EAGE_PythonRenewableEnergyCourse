import { MAX_FLOW_M3S, MAX_INTERVAL_MINUTES } from './balance-model.js';

const colors = { water: '#288daa', net: '#398470', loss: '#c17932', pipe: '#49666f', electricity: '#e5a124' };

export function mountCanvasDiagram(host, kind) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas is unavailable');
  (host.querySelector('#drawing') || host).append(canvas);
  let values, phase;

  function polygon(points, color) {
    ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }
  function line(points, color, width = 2) {
    ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = width;
    ctx.stroke();
  }
  function dot(x, y, radius, color) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }
  function tank(x, floor, width, wallHeight, fill) {
    const top = floor - wallHeight;
    const rear = 30;
    polygon([[x, floor], [x + width, floor], [x + width + rear, floor - 17], [x + rear, floor - 17]], '#bdd0c7');
    polygon([[x + rear, floor - 17], [x + width + rear, floor - 17], [x + width + rear, top - 17], [x + rear, top - 17]], '#9db2a8');
    polygon([[x + width, floor], [x + width + rear, floor - 17], [x + width + rear, top - 17], [x + width, top]], '#809b90');
    if (fill <= 0) return;
    const height = floor - fill;
    polygon([[x, floor], [x + width, floor], [x + width, height], [x, height]], '#227e99');
    polygon([[x, height], [x + width, height], [x + width + rear, height - 17], [x + rear, height - 17]], '#58b7cd');
  }
  function particles(count, points, color, radius, limit = Infinity) {
    for (let i = 0; i < count; i++) {
      const progress = ((phase + i / count) % 1) * (points.length - 1);
      const index = Math.floor(progress);
      const [x, y] = points[index];
      const [nextX, nextY] = points[index + 1];
      const t = progress - index;
      const particleY = y + (nextY - y) * t;
      if (particleY <= limit) dot(x + (nextX - x) * t, particleY, radius, color);
    }
  }
  function render() {
    if (!values) return;
    const width = canvas.width, height = canvas.height;
    if (!width || !height) return;
    const scale = Math.min(width / 490, height / 214);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.setTransform(scale, 0, 0, scale, (width - 490 * scale) / 2, (height - 214 * scale) / 2);
    const [first, flow, third, fourth] = values;

    if (kind === 'head') {
      const datum = 181;
      const y = datum - first * 1.8;
      const tap = 248;
      polygon([[98, datum], [385, datum], [418, datum - 16], [130, datum - 16]], '#9cb7ae');
      line([[112, y], [366, y]], colors.pipe, 17);
      line([[112, y + 2], [366, y + 2]], colors.water, 8);
      line([[tap, y], [tap, y - 64]], '#afcfd1', 11);
      if (third > 0) line([[tap, y], [tap, y - third * 1.8]], colors.water, 6);
      if (first > 0) line([[98, datum], [98, y], [112, y]], '#24584e');
      ctx.font = 'bold 14px system-ui';
      ctx.fillStyle = '#24584e';
      ctx.fillText('Datum · 0 m', 119, datum + 17);
      if (first > 0) ctx.fillText('z', 80, (datum + y) / 2 + 4);
      if (flow > 0) particles(12, [[115, y], [364, y]], '#e6fbff', 3.6);
    } else if (kind === 'flow-volume') {
      const fill = first / (MAX_FLOW_M3S * MAX_INTERVAL_MINUTES * 60) * 94;
      tank(257, 180, 105, 106, fill);
      const path = [[99, 62], [192, 62], [281, 62], [300, 75], [300, 175]];
      line(path.slice(0, 3), colors.pipe, 12);
      if (flow > 0) particles(17, path, '#46bed7', 4, 180 - fill);
    } else if (kind === 'net-head') {
      const gross = first * 1.18, loss = flow * 1.18, base = 153;
      tank(87, 181, 94, 150, 28 + gross);
      tank(300, 181, 94, 44, 28);
      ctx.fillStyle = colors.net;
      ctx.fillRect(228, base - gross + loss, 17, gross - loss);
      ctx.fillStyle = colors.loss;
      ctx.fillRect(228, base - gross, 17, loss);
      line([[221, base], [252, base]], colors.pipe);
    } else {
      const drop = first * 1.18;
      tank(83, 181, 82, 148, 28 + drop);
      const electric = kind === 'electrical-power';
      tank(electric ? 227 : 306, 181, 91, 45, 28);
      const outlet = electric ? 273 : 329;
      const path = [[144, 166 - drop], [outlet, 166]];
      line(path, colors.pipe, 13);
      if (flow > 0) particles(12, path, '#baf5ff', 3.6);
      if (!electric) return;
      line([[outlet, 166], [334, 166]], '#67513c', 6);
      dot(outlet, 166, 16, '#98744c');
      const angle = third > 0 ? phase * Math.PI * 4 : 0;
      for (let i = 0; i < 4; i++) {
        const a = angle + i * Math.PI / 2;
        line([[outlet, 166], [outlet + 13 * Math.cos(a), 166 + 13 * Math.sin(a)]], '#e5c391', 3);
      }
      ctx.fillStyle = '#98744c';
      ctx.fillRect(325, 146, 43, 38);
      line([[368, 165], [420, 165]], colors.pipe, 2);
      if (fourth > 0) particles(4, [[369, 165], [420, 165]], colors.electricity, 3.5);
    }
  }

  new ResizeObserver(() => {
    const ratio = Math.min(devicePixelRatio, 2);
    canvas.style.width = `${host.clientWidth}px`;
    canvas.style.height = `${host.clientHeight}px`;
    canvas.width = Math.round(host.clientWidth * ratio);
    canvas.height = Math.round(host.clientHeight * ratio);
    render();
  }).observe(host);
  return (next, position) => {
    values = next;
    phase = position;
    render();
  };
}
