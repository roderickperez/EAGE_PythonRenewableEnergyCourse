import { RAD, incidenceCosine, solarVector } from './model.js';

const ns = 'http://www.w3.org/2000/svg';
export const solarColors = { beam: '#d98c22', sky: '#368caf', ground: '#408778', normal: '#526976' };
const center = [0, 1.26, 0];
const up = [0, 1, 0];

function offset(origin, direction, length) {
  return origin.map((value, index) => value + length * direction[index]);
}

function project([east, height, north]) {
  return [270 + 32 * (.83 * east - .60 * north), 179 - 30 * (height + .25 * east + .41 * north)];
}

export function solarSvgNode(parent, tag, attributes) {
  const element = document.createElementNS(ns, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  parent.append(element);
  return element;
}

export function mountResourceScene(svg, showAngle) {
  const add = (tag, attrs) => solarSvgNode(svg, tag, attrs);
  const polygon = fill => add('polygon', { fill });
  const points = (element, positions) => element.setAttribute('points', positions.map(project).join(' '));
  const ground = [[-3, 0, -2], [3, 0, -2], [3, 0, 2], [-3, 0, 2]];
  const lower = ground.map(point => offset(point, up, -.16));
  points(polygon('#789181'), [ground[0], ground[1], lower[1], lower[0]]);
  points(polygon('#8da49a'), [ground[0], ground[3], lower[3], lower[0]]);
  points(polygon('#b9cdbd'), ground);
  add('ellipse', { cx: 270, cy: 180, rx: 34, ry: 10, fill: '#90a99a', opacity: '.7' });
  add('line', { x1: 270, x2: 270, y1: 179, y2: project(center)[1], stroke: '#687f89', 'stroke-width': 9 });
  const under = polygon('#536f80');
  const frame = polygon('#849daa');
  const tiles = Array.from({ length: 24 }, (_, i) => polygon(['#21476c', '#2c567a', '#254f74'][i % 3]));

  const sun = add('circle', { r: 9, fill: '#f7bd52', stroke: '#fff0bd', 'stroke-width': 2.3 });
  const axis = add('polyline', { fill: 'none', stroke: solarColors.normal, 'stroke-width': 2, 'stroke-dasharray': '4 3' });
  const direction = add('polyline', { fill: 'none', stroke: solarColors.beam, 'stroke-width': 1.5, 'stroke-dasharray': '3 4' });
  const arc = add('polyline', { fill: 'none', stroke: '#94602a', 'stroke-width': 2.1 });
  const rays = ['ground', 'sky', 'beam'].map(key => ({
    key,
    set: Array.from({ length: key === 'beam' ? 3 : 2 }, () => ({
      line: add('polyline', { fill: 'none', stroke: solarColors[key], 'stroke-width': 2, 'stroke-linecap': 'round' }),
      pulse: add('circle', { r: 2.9, fill: solarColors[key], stroke: '#fff', 'stroke-width': 1 }),
    })),
  }));
  let geometry;
  let routes;

  function update({ tilt, azimuth, sunDirection, normal, ...components }, phase) {
    const state = [tilt, azimuth, ...sunDirection].join(',');
    if (state !== geometry) {
      geometry = state;
      const right = solarVector(90, azimuth + 90);
      const down = solarVector(90 + tilt, azimuth);
      const panel = (x, y) => offset(offset(center, right, x), down, y);
      const corners = (x1, y1, x2, y2) => [panel(x1, y1), panel(x2, y1), panel(x2, y2), panel(x1, y2)];
      points(frame, corners(-1.47, -.83, 1.47, .83));
      points(under, corners(-1.51, -.86, 1.51, .86).map(point => offset(point, up, -.11)));
      tiles.forEach((tile, index) => {
        const x = -1.39 + (index % 6) * .465;
        const y = -.75 + Math.floor(index / 6) * .38;
        points(tile, corners(x, y, x + .43, y + .338));
      });
      const source = offset(center, sunDirection, 3.53);
      const [x, y] = project(source);
      sun.setAttribute('cx', x);
      sun.setAttribute('cy', y);
      points(axis, [center, offset(center, normal, 2.0)]);
      points(direction, [center, source]);
      const cosine = incidenceCosine(sunDirection, normal);
      const angle = Math.acos(cosine);
      const tangent = 1 + cosine < 1e-10 ? up : sunDirection.map((value, i) =>
        (value - cosine * normal[i]) / Math.sin(angle || 1));
      const arcPoints = [];
      if (angle > .01) for (let step = 0; step <= 18; step++) {
        const t = angle * step / 18;
        const blend = normal.map((n, i) => n * Math.cos(t) + tangent[i] * Math.sin(t));
        arcPoints.push(offset(center, blend, 1.45));
      }
      points(arc, arcPoints);
      const facing = solarVector(90, azimuth);
      routes = {
        beam: [-.85, 0, .85].map(x => {
          const end = panel(x, .12);
          return [offset(end, sunDirection, 2.83), end];
        }),
        sky: [-1.0, 1.0].map(x => {
          const end = panel(x, -.16);
          return [offset(offset(end, up, 1.8), right, x * .65), end];
        }),
        ground: [-.76, .76].map(x => {
          const end = panel(x, .46);
          const start = offset(end, facing, 2.5);
          start[1] = end[1] - Math.min(end[1] - .04, Math.sin(tilt * RAD) * 1.6);
          return [start, end];
        }),
      };
      for (const { key, set } of rays) set.forEach(({ line }, i) => points(line, routes[key][i]));
    }
    const shown = showAngle || components.beam > 0 || components.sky > 0 || components.ground > 0;
    sun.style.display = shown ? '' : 'none';
    for (const element of [axis, direction, arc]) element.style.display = showAngle ? '' : 'none';
    for (const { key, set } of rays) {
      const value = components[key];
      const opacity = value > 0 ? .34 + .66 * Math.min(1, value / (key === 'beam' ? 780 : 180)) : 0;
      set.forEach(({ line, pulse }, i) => {
        const [[x1, y1], [x2, y2]] = routes[key][i].map(project);
        const t = (phase + i * .24) % 1;
        line.setAttribute('opacity', opacity);
        pulse.setAttribute('opacity', opacity);
        pulse.setAttribute('cx', x1 + (x2 - x1) * t);
        pulse.setAttribute('cy', y1 + (y2 - y1) * t);
      });
    }
  }
  return update;
}
