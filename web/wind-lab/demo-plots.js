import { setupPlot } from '../hydro-lab/comparison-plots.js';

export const windColors = { wind: '#237f9b', rotor: '#398470', electrical: '#a9581c', reference: '#647c78' };
const format = new Intl.NumberFormat('en', { maximumFractionDigits: 3 });
export const n = value => format.format(Math.abs(value) < .0005 ? 0 : value);
export const mw = value => `${n(value)} MW`;
export const percent = fraction => `${n(fraction * 100)}%`;
export const sample = (min, max, evaluate, count = 48) =>
  Array.from({ length: count + 1 }, (_, i) => {
    const x = min + (max - min) * i / count;
    return [x, evaluate(x)];
  });

export function axesFor(svg, xMin, xMax, yMax, xLabel, yLabel) {
  return setupPlot(svg, xMax, Math.max(.01, yMax), xLabel, yLabel, xMin);
}

export function operatingMarker(axes, x, y, color = windColors.electrical) {
  axes.line(axes.x(x), axes.y(0), axes.x(x), axes.y(y), color, '4 3');
  axes.dot(x, y, color, 5.5);
}
