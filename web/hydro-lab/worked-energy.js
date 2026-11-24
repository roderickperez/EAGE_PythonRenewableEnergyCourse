const tick = new Intl.NumberFormat('en', { maximumFractionDigits: 3 });

export function energyPlot(canvas, slider, value, estimate, selected, total) {
  const [, label, unit, min, max] = slider;
  const samples = Array.from({ length: 65 }, (_, i) => estimate(min + (max - min) * i / 64));
  const totals = total && Array.from({ length: 65 }, (_, i) => total(min + (max - min) * i / 64));
  const ceiling = Math.max(1, selected, ...samples, ...(totals || []));
  const step = 10 ** Math.floor(Math.log10(ceiling / 3));
  const upper = Math.ceil(ceiling / (3 * step)) * 3 * step;
  const { clientWidth: width, clientHeight: height } = canvas;
  if (!width || !height) return;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const left = 48, right = width - 12, top = 25, bottom = height - 40;
  const x = number => left + (right - left) * (number - min) / (max - min);
  const y = number => bottom - (bottom - top) * number / upper;
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#425c6b';
  ctx.fillText('GWh/year', 6, 13);
  for (let i = 0; i <= 3; i++) {
    const level = upper * i / 3;
    const row = y(level);
    ctx.strokeStyle = '#d8e5e9';
    ctx.beginPath(); ctx.moveTo(left, row); ctx.lineTo(right, row); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(tick.format(level), left - 6, row + 4);
  }
  ctx.textAlign = 'left'; ctx.fillText(tick.format(min), left, bottom + 16);
  ctx.textAlign = 'right'; ctx.fillText(tick.format(max), right, bottom + 16);
  ctx.textAlign = 'center'; ctx.fillText(unit ? `${label} (${unit})` : label, (left + right) / 2, height - 7);
  function drawSeries(points, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.beginPath();
    points.forEach((point, i) => {
      const px = x(min + (max - min) * i / 64);
      if (i === 0) ctx.moveTo(px, y(point));
      else ctx.lineTo(px, y(point));
    });
    ctx.stroke();
  }
  drawSeries(samples, '#328baa');
  if (totals) {
    ctx.setLineDash([5, 3]);
    drawSeries(totals, '#398470');
    ctx.setLineDash([]);
  }
  ctx.fillStyle = '#b8780c';
  ctx.beginPath(); ctx.arc(x(value), y(selected), 4.5, 0, 2 * Math.PI); ctx.fill();
  const caption = total ? ` Main and ecological combined: ${tick.format(total(value))} gigawatt-hours.` : '';
  canvas.setAttribute('aria-label', `Annual energy versus ${label}. At ${tick.format(value)} ${unit}, annual main plant energy is ${tick.format(selected)} gigawatt-hours.${caption}`);
}
