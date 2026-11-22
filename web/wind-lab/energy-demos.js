import { RAD, YEAR_H, annualEnergyScenarios, circularMean, electricalPower, weibullPowerBins } from './model.js';
import { axesFor, mw, n, percent, sample, windColors as c } from './demo-plots.js';
import { referenceTurbine } from './physical-demos.js';
import { solarSvgNode as node } from '../solar-lab/resource-scene.js';

const gwh = value => `${n(value)} GWh`;

export const energyDemos = {
  'annual-energy': {
    title: 'From time intervals to annual energy',
    plot: 'Interval power and the duration-weighted mean',
    caption: 'The area under the stepped curve gives energy for the sample. The shaded height is the same mean power used in the annual estimate.',
    note: 'The annual estimate assumes the selected speed mix repeats over all 8,760 hours of a non-leap year. This is a synthetic extrapolation using the same 3 MW reference curve in both intervals.',
    legend: [[c.wind, 'Electrical power over the sample'], [c.rotor, 'Duration-weighted mean']],
    controls: [
      ['speedA', 'First interval wind speed', 0, 30, .1, 6, 'm/s'],
      ['speedB', 'Second interval wind speed', 0, 30, .1, 12, 'm/s'],
      ['share', 'First interval share', 0, 100, 1, 60, '%'],
      ['hours', 'Total sample duration', 1, 168, 1, 24, 'h'],
    ],
    draw(svg, { speedA, speedB, share, hours }) {
      const first = electricalPower(speedA, referenceTurbine);
      const second = electricalPower(speedB, referenceTurbine);
      const fraction = share / 100;
      const mean = fraction * first + (1 - fraction) * second;
      const split = hours * fraction;
      const axes = axesFor(svg, 0, 168, referenceTurbine.ratedMW, 'Elapsed sample time (h)', 'Electrical power (MW)');
      node(svg, 'rect', { x: axes.x(0), y: axes.y(mean), width: axes.x(hours) - axes.x(0),
        height: axes.y(0) - axes.y(mean), fill: c.rotor, 'fill-opacity': .13 });
      const series = fraction === 0 ? [[0, second], [hours, second]] : fraction === 1 ? [[0, first], [hours, first]]
        : [[0, first], [split, first], [split, second], [hours, second]];
      axes.series(series, c.wind);
      axes.line(axes.x(0), axes.y(mean), axes.x(hours), axes.y(mean), c.rotor, '5 4');
      return { headline: `Annual estimate = ${gwh(mean * YEAR_H / 1000)}`, rows: [
        `Sample energy = ${n(first)} MW × ${n(split)} h + ${n(second)} MW × ${n(hours - split)} h = <strong>${n(mean * hours)} MWh</strong>`,
        `Probability-weighted mean power = ${n(fraction)} × ${n(first)} + ${n(1 - fraction)} × ${n(second)} = ${mw(mean)}`,
        `Annual estimate = ${n(mean)} MW × 8,760 h = <strong>${gwh(mean * YEAR_H / 1000)}</strong>`,
      ] };
    },
  },
  weibull: {
    title: 'Integrating a Weibull wind resource',
    plot: 'Contribution of each speed bin to mean power',
    caption: 'Each bar is midpoint electrical power × probability of the bin. Add the bar heights to get mean power.',
    note: 'The bins cover 0–30 m/s and the reference turbine cuts out at 25 m/s. The omitted upper-tail probability therefore contributes zero power. Refine the bins to check convergence.',
    legend: [[c.wind, 'Probability-weighted power per bin']],
    controls: [['shape', 'Weibull shape k', 1.2, 4, .05, 2, ''],
      ['scale', 'Weibull scale c', 4, 13, .1, 8, 'm/s'],
      ['bins', 'Number of speed bins', 6, 90, 1, 30, '']],
    draw(svg, { shape, scale, bins }) {
      const result = weibullPowerBins({ shape, scale, count: bins, maximum: 30, turbine: referenceTurbine });
      const refined = weibullPowerBins({ shape, scale, count: bins * 2, maximum: 30, turbine: referenceTurbine });
      const axes = axesFor(svg, 0, 30, referenceTurbine.ratedMW, 'Wind speed bin (m/s)', 'Bin contribution (MW)');
      for (const bin of result.bins) {
        node(svg, 'rect', { x: axes.x(bin.lower) + .3, y: axes.y(bin.contribution),
          width: axes.x(bin.upper) - axes.x(bin.lower) - .6, height: axes.y(0) - axes.y(bin.contribution),
          fill: c.wind, 'fill-opacity': .65 });
      }
      const annual = mean => mean * YEAR_H / 1000;
      return { headline: `Annual energy = ${gwh(annual(result.meanMW))}`, rows: [
        `Bin width ${n(30 / bins)} m/s · Covered probability ${percent(1 - result.tail)} · Above 30 m/s ${percent(result.tail)}`,
        `Σ P(v<sub>mid</sub>) × [F(v<sub>upper</sub>) − F(v<sub>lower</sub>)] ≈ <strong>${mw(result.meanMW)}</strong>`,
        `With ${bins * 2} bins, annual energy is ${gwh(annual(refined.meanMW))}. Difference (refined − selected) ${gwh(annual(refined.meanMW - result.meanMW))}.`,
      ] };
    },
  },
  losses: {
    title: 'Annual losses and P90 uncertainty',
    plot: 'Illustrative distribution of net annual energy',
    caption: 'P90 is the 10th percentile on the energy axis. For a continuous distribution, 90% of outcomes exceed it; at zero uncertainty all outcomes equal P90.',
    note: 'Availability and loss factors are teaching assumptions. The example then applies a symmetric normal uncertainty model to net annual energy. It requires site evidence before use in project estimates.',
    legend: [[c.wind, 'Illustrative distribution'], [c.rotor, 'P50'], [c.electrical, 'P90']],
    controls: [['gross', 'Gross annual energy', 2, 30, .1, 12, 'GWh'],
      ['availability', 'Availability', 70, 100, 1, 96, '%'],
      ['wake', 'Wake energy loss', 0, 30, 1, 8, '%'],
      ['other', 'Other independent losses', 0, 20, 1, 4, '%'],
      ['uncertainty', 'Relative standard deviation', 0, 20, .5, 10, '%']],
    draw(svg, { gross, availability, wake, other, uncertainty }) {
      const { afterAvailability, afterWake, p50, p90, deviation } =
        annualEnergyScenarios(gross, availability / 100, wake / 100, other / 100, uncertainty / 100);
      const peak = deviation > 0 ? 1 / (deviation * Math.sqrt(2 * Math.PI)) : 0;
      const delta = 4 * deviation;
      // Fixed limits cover every slider value, including the smallest nonzero uncertainty.
      const axes = axesFor(svg, 0, 54, 105, 'Net annual energy (GWh)', 'Probability density (1/GWh)');
      if (deviation > 0) {
        const density = energy => peak * Math.exp(-.5 * ((energy - p50) / deviation) ** 2);
        axes.series(sample(Math.max(0, p50 - delta), p50 + delta, density, 80), c.wind);
        axes.line(axes.x(p90), axes.y(0), axes.x(p90), axes.y(density(p90)), c.electrical, '4 3');
        axes.dot(p90, density(p90), c.electrical);
      }
      axes.line(axes.x(p50), axes.y(0), axes.x(p50), axes.y(peak), c.rotor, '4 3');
      axes.dot(p50, peak, c.rotor, 5);
      return { headline: `P90 = ${gwh(p90)}`, rows: [
        `Gross ${gwh(gross)} → after availability ${gwh(afterAvailability)} → after wake loss ${gwh(afterWake)} → net ${gwh(p50)}`,
        `P50 = ${gwh(p50)} · σ = ${gwh(deviation)} · P90 = <strong>${gwh(p90)}</strong>`,
        deviation > 0 ? 'P90 = P50 − 1.282σ under the selected normal model.' : 'With zero uncertainty, P90 and P50 coincide; the point represents a single value rather than a probability density.',
      ] };
    },
  },
  direction: {
    title: 'Averaging directions on a compass',
    plot: 'Meteorological wind bearings',
    caption: 'Arrows point toward the source bearings. The mean arrow shortens as the bearings spread apart.',
    note: 'Direction is clockwise from north. The component calculation uses the direction the air moves toward, which is opposite its reported source bearing.',
    legend: [[c.wind, 'First bearing'], [c.rotor, 'Second bearing'], [c.electrical, 'Circular mean']],
    controls: [['first', 'First direction', 0, 359, 1, 350, '°'], ['second', 'Second direction', 0, 359, 1, 10, '°'],
      ['speed', 'Speed for component example', 0, 20, .1, 8, 'm/s']],
    draw(svg, { first, second, speed }) {
      svg.replaceChildren();
      const cx = 205, cy = 97, radius = 63;
      node(svg, 'circle', { cx, cy, r: radius, fill: '#edf6f2', stroke: '#829d93' });
      for (const [letter, x, y] of [['N', cx, 23], ['E', cx + 78, cy + 4], ['S', cx, 177], ['W', cx - 78, cy + 4]]) {
        const label = node(svg, 'text', { x, y, 'text-anchor': 'middle', fill: '#31524f', 'font-size': 12 });
        label.textContent = letter;
      }
      function arrow(degrees, length, color, dash = '') {
        const bearing = degrees * RAD;
        const dx = Math.sin(bearing), dy = -Math.cos(bearing);
        const x = cx + length * dx, y = cy + length * dy;
        node(svg, 'line', { x1: cx, y1: cy, x2: x, y2: y, stroke: color, 'stroke-width': 2.6, 'stroke-dasharray': dash });
        const tip = [[x, y], [x - 8 * dx - 4 * dy, y - 8 * dy + 4 * dx], [x - 8 * dx + 4 * dy, y - 8 * dy - 4 * dx]];
        node(svg, 'polygon', { points: tip.join(' '), fill: color });
      }
      arrow(first, radius * .87, c.wind);
      arrow(second, radius * .78, c.rotor, '5 3');
      const mean = circularMean(first, second);
      if (mean.degrees !== null) arrow(mean.degrees, radius * mean.length, c.electrical);
      const east = -speed * Math.sin(first * RAD), north = -speed * Math.cos(first * RAD);
      return { headline: `Circular mean = ${mean.degrees === null ? 'undefined' : n(mean.degrees % 360) + '°'}`, rows: [
        `First bearing ${first}° · Second bearing ${second}° · Resultant length <strong>${n(mean.length)}</strong>`,
        `First observation at ${n(speed)} m/s · Eastward component ${n(east)} m/s · Northward component ${n(north)} m/s`,
      ] };
    },
  },
};
