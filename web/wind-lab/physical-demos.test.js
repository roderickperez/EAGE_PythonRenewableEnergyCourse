import test from 'node:test';
import assert from 'node:assert/strict';
import { physicalDemos } from './physical-demos.js';
import { energyDemos } from './energy-demos.js';

function render(input) {
  globalThis.document = { createElementNS: (_, tag) => ({ tag, attrs: {}, setAttribute(key, value) { this.attrs[key] = value; } }) };
  const svg = { children: [], replaceChildren() { this.children = []; }, append(element) { this.children.push(element); } };
  physicalDemos['cube-law'].draw(svg, input);
  return {
    labels: svg.children.filter(el => el.tag === 'text').map(el => el.textContent),
    curve: svg.children.find(el => el.tag === 'polyline').attrs.points,
    markers: svg.children.filter(el => el.tag === 'circle').map(el => el.attrs),
  };
}

test('wind power plot keeps axes fixed across slider ranges', () => {
  const low = render({ speed: 0, density: 0.8, diameter: 40 });
  const middle = render({ speed: 8, density: 1.225, diameter: 80 });
  const high = render({ speed: 16, density: 1.4, diameter: 140 });
  assert.deepEqual(low.labels, middle.labels);
  assert.deepEqual(middle.labels, high.labels);
  assert.notEqual(low.curve, high.curve);
  assert.notEqual(middle.markers[0].cx, high.markers[0].cx);
  for (const marker of high.markers) {
    assert.ok(+marker.cx >= 59 && +marker.cx <= 393);
    assert.ok(+marker.cy >= 14 && +marker.cy <= 149);
  }
});


test('all Cartesian wind plots retain their scales across every control', () => {
  const demos = { ...physicalDemos, ...energyDemos };
  for (const [name, demo] of Object.entries(demos)) {
    if (name === 'direction') continue;
    const defaults = Object.fromEntries(demo.controls.map(([key, , , , , value]) => [key, value]));
    const draw = input => {
      globalThis.document = { createElementNS: (_, tag) => ({ tag, attrs: {}, setAttribute(key, value) { this.attrs[key] = value; } }) };
      const svg = { children: [], replaceChildren() { this.children = []; }, append(el) { this.children.push(el); } };
      demo.draw(svg, input);
      return svg.children.filter(el => el.tag === 'text').map(el => el.textContent);
    };
    const baseline = draw(defaults);
    for (const [key, , min, max] of demo.controls) {
      for (const value of [min, max]) {
        assert.deepEqual(draw({ ...defaults, [key]: value }), baseline, `${name}: ${key}=${value}`);
      }
    }
  }
});


test('wind plot curves and markers fit at slider-range corners', () => {
  for (const [name, demo] of Object.entries({ ...physicalDemos, ...energyDemos })) {
    if (name === 'direction') continue;
    for (let mask = 0; mask < 2 ** demo.controls.length; mask++) {
      const input = Object.fromEntries(demo.controls.map(([key, , min, max], i) => [key, mask & (1 << i) ? max : min]));
      globalThis.document = { createElementNS: (_, tag) => ({ tag, attrs: {}, setAttribute(key, value) { this.attrs[key] = value; } }) };
      const svg = { children: [], replaceChildren() { this.children = []; }, append(el) { this.children.push(el); } };
      demo.draw(svg, input);
      for (const element of svg.children) {
        const points = element.tag === 'circle' ? [[element.attrs.cx, element.attrs.cy]]
          : element.tag === 'polyline' ? element.attrs.points.split(' ').map(point => point.split(',')) : [];
        for (const [x, y] of points) {
          assert.ok(+x >= 58.99 && +x <= 393.01 && +y >= 13.99 && +y <= 149.01,
            `${name}: point (${x}, ${y}) is outside the plot at ${JSON.stringify(input)}`);
        }
      }
    }
  }
});
