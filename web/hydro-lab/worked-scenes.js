import * as T from 'three';
import { HydroScene } from './scene.js';
import { visualState } from './visual-state.js';
import { CatchmentWeather } from './worked-weather.js';

const riverSource = [
  [-11.9, 3.90, -5.7], [-10.2, 3.66, -6.0], [-8.9, 3.52, -4.9],
  [-8.0, 3.42, -3.5], [-6.55, 3.35, -2.85],
];
const bypassReach = [
  riverSource.at(-1), [-5.9, 3.24, -2.25], [-5.1, 2.62, -.6],
  [-6.0, 2.11, 1.45], [-5.2, 1.49, 3.45], [-2.8, .99, 4.3], [-.6, .64, 3.95],
  [1.4, .35, 4.9], [3.85, .04, 5.65], [6.45, -.12, 4.6], [8.4, -.2, 3.9],
];
const downstreamReach = [bypassReach.at(-1), [10.2, -.27, 4.08], [11.95, -.32, 5.1]];
const channelRoute = [
  [-6.6, 2.17, -3.14], [-5.1, 2.13, -4.25], [-2.7, 2.08, -4.45],
  [-.2, 2.03, -3.65], [2.02, 1.98, -2.37],
];
const forebay = [2.26, 1.98, -2.23];
const tailrace = [[6.6, -.27, 1.04], [7.1, -.25, 2.43], [8.4, -.2, 3.9]];

function pathFrames(points, segments = 48) {
  const curve = new T.CatmullRomCurve3(points.map(point => new T.Vector3(...point)));
  const frames = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const { x, y, z } = curve.getPointAt(t);
    const tangent = curve.getTangentAt(t);
    const lateral = new T.Vector2(-tangent.z, tangent.x).normalize();
    frames.push({ x, y, z, lateral, t });
  }
  return frames;
}

function makeStrip(frames, profile) {
  const positions = [];
  const indices = [];
  for (const frame of frames) {
    for (const [offset, height] of profile(frame)) {
      positions.push(frame.x + offset * frame.lateral.x, frame.y + height, frame.z + offset * frame.lateral.y);
    }
  }
  const stride = positions.length / (frames.length * 3);
  for (let i = 0; i < frames.length - 1; i++) {
    for (let edge = 0; edge < stride - 1; edge++) {
      const a = i * stride + edge;
      const b = a + stride;
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

class WorkedScene extends HydroScene {
  addWeather(options) {
    this.weather = new CatchmentWeather(this.scene, options);
    const [x, y, z] = options.clouds[0];
    this.weatherAnchor = new T.Vector3(x, y - 1.2, z + .8);
    this.callout = document.createElement('div');
    this.callout.className = 'scene-callout';
    this.callout.hidden = true;
    this.host.append(this.callout);
  }

  showControl(key, label, value) {
    this.calloutAnchor = this.controlAnchor(key);
    this.callout.hidden = !this.calloutAnchor;
    this.callout.textContent = label + ': ' + value;
  }

  controlAnchor(key) {
    return key === 'precip' || key === 'rain' ? this.weatherAnchor : undefined;
  }

  updateWeather(precipitation) {
    this.weather.setPrecipitation(precipitation);
    this.host.dataset.rainDrops = String(this.weather.count);
  }

  frame(time) {
    if (this.visible && this.animate) this.weather.step(Math.min((time - this.lastTime) / 1000, .05));
    super.frame(time);
    if (!this.visible || !this.calloutAnchor) return;
    const anchor = this.calloutAnchor.clone().project(this.camera);
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    const x = (anchor.x + 1) * width / 2 + 7;
    const y = (1 - anchor.y) * height / 2 + 6;
    const left = T.MathUtils.clamp(x, 4, width - this.callout.offsetWidth - 8);
    const top = T.MathUtils.clamp(y, 8, height - this.callout.offsetHeight - 8);
    this.callout.style.transform = 'translate(' + Math.round(left) + 'px,' + Math.round(top) + 'px)';
  }

  padView(scale, lift) {
    this.camera.left /= scale;
    this.camera.right /= scale;
    this.camera.top = this.camera.top / scale + lift;
    this.camera.bottom = this.camera.bottom / scale + lift;
    this.camera.updateProjectionMatrix();
  }

  surfaceCurrent(points, type, count = 16) {
    const curve = new T.CatmullRomCurve3(points.map(point => new T.Vector3(...point)));
    const foam = this.mat('#d9fbff', .2);
    const geometry = new T.SphereGeometry(.11, 8, 6);
    const particles = Array.from({ length: count }, () => {
      const particle = this.mesh(geometry, foam, [0, 0, 0]);
      particle.castShadow = false;
      return particle;
    });
    const stream = { curve, particles, type, phase: 0, offset: 0 };
    this.streams.push(stream);
    return stream;
  }

  streamMotion(stream) {
    return this.speeds?.[stream.type] ?? 0;
  }
}

export class DiversionScene extends WorkedScene {
  build() {
    this.camera.position.set(9.5, 17.5, 23);
    this.controls.target.set(-.15, 1.95, .25);
    this.controls.update();
    this.buildValley();
    this.addWeather({ clouds: [[-9.4, 6.6, -5.08, .94], [-6.95, 6.38, -5.05, .66]], dropHeight: 2.62 });

    const [x, y, z] = riverSource.at(-1);
    const weir = this.box([.28, .51, 1.7], [x + .5, y - .27, z + .43], this.concrete, 'Low diversion weir');
    weir.rotation.y = -.56;
    this.box([.43, 1.25, .43], [x + .26, y - .42, z - .58], this.cut, 'Channel intake gate');
    this.bar([x + .26, y - .25, z - .58], [x + .26, y + .54, z - .58], .035, this.steel, 'Intake gate stem');

    this.channelFrames = pathFrames(channelRoute);
    this.channelBed = this.mesh(new T.BufferGeometry(), this.concrete, [0, 0, 0], 'Concrete open-channel bed');
    this.channelWalls = this.mesh(new T.BufferGeometry(), this.cut, [0, 0, 0], 'Open channel walls');
    this.channelWater = this.mesh(new T.BufferGeometry(), this.water, [0, 0, 0], 'Water in the upper diversion channel');
    this.channelWater.castShadow = false;
    this.channelCurrent = this.surfaceCurrent(channelRoute, 'channel', 25);

    this.box([1.63, .64, 2.23], [forebay[0] + .39, 1.62, forebay[2] + .29], this.concrete, 'Forebay foundation');
    this.forebayWater = this.box([1.41, 1, 2.04], [forebay[0] + .39, 2.33, forebay[2] + .29], this.water, 'Forebay feeding both penstocks');
    this.box([.19, 1.32, 2.23], [forebay[0] + 1.14, 2.36, forebay[2] + .29], this.cut, 'Forebay wall and penstock intakes');
    for (const offset of [-.36, .36]) {
      this.pipe([[2.7, 2.04, -2.0 + offset], [3.6, 1.85, -1.6 + offset], [4.8, 1.11, -.45 + offset],
        [5.8, .53, .39 + offset], [6.52, .45, .82 + offset * .45]], .19, 'One of two steel penstocks');
    }
    this.box([3, .25, 3.2], [6.75, -.37, 1], this.concrete, 'Powerhouse slab');
    this.box([3, 2.5, .25], [6.75, 1.02, -.45], this.concrete, 'Powerhouse cutaway wall');
    this.box([3, .2, 1.05], [6.75, 2.32, -.05], this.cut, 'Powerhouse roof');
    this.cyl(.69, .22, [6.65, .31, 1.05], this.teal, 'Turbine casing');
    this.runner = new T.Group();
    this.runner.position.set(6.65, .55, 1.05);
    this.scene.add(this.runner);
    for(let blade=0;blade<8;blade++) {
      const angle = blade * Math.PI / 4;
      const part = this.box([.45, .09, .17], [.35*Math.cos(angle), 0, .35*Math.sin(angle)], this.steel, 'Turbine runner', this.runner);
      part.rotation.y = -angle;
    }
    this.cyl(.08, .9, [6.65, 1.01, 1.05], this.steel, 'Turbine–generator shaft');
    this.rotor = new T.Group();
    this.rotor.position.set(6.65, 1.56, 1.05);
    this.scene.add(this.rotor);
    this.cyl(.37, .29, [0, 0, 0], this.copper, 'Generator rotor', this.rotor);
    this.box([.17, .08, .6], [0, .16, 0], this.dark, 'Generator rotor poles', this.rotor);
    this.pipe([[6.65, .35, 1.05], [6.65, .06, 1.3], [6.64, -.24, 1.55]], .23, 'Draft tube and tailrace inlet', 'tailrace');
    this.box([.85, .65, .73], [8.25, .1, .05], this.steel, 'Transformer');
    this.energyPath([[6.65, 1.65, 1.05], [7.35, 1.35, .45], [8.25, .5, .05]], 'Electrical output');
  }

  buildValley() {
    const riverPaths = [
      { points: riverSource, width: 1.68, type: 'inflow', label: 'Upstream river', particles: 15 },
      { points: bypassReach, width: 1.73, type: 'bypass', label: 'Bypass river below the diversion weir', particles: 42 },
      { points: tailrace, width: 1.29, type: 'tailrace', label: 'Tailrace returning turbine water to the river', particles: 13 },
      { points: downstreamReach, width: 2.05, type: 'inflow', label: 'Rejoined downstream river', particles: 12 },
    ];
    const beds = riverPaths.flatMap(({ points, width, type, label, particles }) => {
      const frames = pathFrames(points);
      const surface = this.mesh(makeStrip(frames, () => [[width / 2, 0], [-width / 2, 0]]), this.water, [0, 0, 0], label);
      surface.castShadow = false;
      this.surfaceCurrent(points.map(([x, y, z]) => [x, y + .045, z]), type, particles);
      return frames.map(frame => ({ ...frame, width }));
    });
    const channel = pathFrames(channelRoute);
    const terrainHeight = (x, z) => {
      const grade = -.49 + 4.19 * T.MathUtils.smoothstep(5 - x, 0, 17);
      const slopes = .38 * Math.sin(x * .75 + z * .6) + .19 * Math.cos(z * 1.3 - x * .5);
      const nearChannel = channel.reduce((best, frame) => Math.min(best, (x - frame.x) ** 2 + (z - frame.z) ** 2), Infinity);
      let height = grade + slopes;
      if (nearChannel < 4) height = T.MathUtils.lerp(Math.min(height, 1.58), height, T.MathUtils.smoothstep(nearChannel, 1.15, 4));
      let closest;
      let bankDistance = Infinity;
      for (const river of beds) {
        const distance = Math.hypot(x - river.x, z - river.z) - river.width / 2;
        if (distance < bankDistance) {
          closest = river;
          bankDistance = distance;
        }
      }
      if (bankDistance < 1.12) {
        const bank = T.MathUtils.smoothstep(bankDistance, -.03, 1.12);
        height = T.MathUtils.lerp(closest.y - .16, height, bank);
      }
      // The broad upstream valley floor rises to the source without hiding the channel.
      return Math.max(-.72, height);
    };
    const valley = new T.PlaneGeometry(24, 15.8, 144, 96).rotateX(-Math.PI / 2);
    const positions = valley.attributes.position;
    const colors = [];
    const openMeadow = new T.Color('#779784');
    const woodland = new T.Color('#567f65');
    const bank = new T.Color('#a0937e');
    const tint = new T.Color();
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i), z = positions.getZ(i);
      const height = terrainHeight(x, z);
      positions.setY(i, height);
      const bankDistance = beds.reduce((dist, river) => Math.min(dist,
        Math.hypot(x - river.x, z - river.z) - river.width / 2), Infinity);
      tint.copy(openMeadow).lerp(woodland, .24 + .22 * Math.sin(x * .53 + z * .63));
      tint.lerp(bank, .82 * (1 - T.MathUtils.smoothstep(bankDistance, .11, .46)));
      colors.push(tint.r, tint.g, tint.b);
    }
    valley.setAttribute('color', new T.Float32BufferAttribute(colors, 3));
    valley.computeVertexNormals();
    const grass = new T.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: T.DoubleSide });
    this.mesh(valley, grass, [0, 0, 0], 'River valley and grassed banks');

    const corners = [[-12, -7.9], [12, -7.9], [12, 7.9], [-12, 7.9]];
    const base = this.mat('#94a498', .95);
    this.box([24, .42, 15.8], [0, -1.04, 0], base, 'Valley base');
    for (let i = 0; i < corners.length; i++) {
      const [x1, z1] = corners[i];
      const [x2, z2] = corners[(i + 1) % corners.length];
      const edge = Array.from({ length: 145 }, (_, step) => {
        const x = T.MathUtils.lerp(x1, x2, step / 144);
        const z = T.MathUtils.lerp(z1, z2, step / 144);
        return { x, y: terrainHeight(x, z), z, lateral: new T.Vector2() };
      });
      this.mesh(makeStrip(edge, ({ y }) => [[0, 0], [0, -.84 - y]]), base, [0, 0, 0], 'Valley cutaway');
    }

    const needles = this.mat('#3b654f', 1);
    const bark = this.mat('#685748', 1);
    for (const [x, z, scale] of [[-11, -7.1, .85], [-9.4, -7.15, .68], [-7, -6.25, .95],
      [-5.2, -6.7, .8], [-3.9, -5.9, .7], [-1.6, -6.4, .89], [.4, -5.6, .74],
      [2.3, -4.8, .7], [3.6, -5.4, .9], [-7.6, 1.65, .85], [-8.35, 3.05, .7],
      [-3.4, 6.3, .78], [-1.3, 6.75, .69], [8.5, 6.25, .77], [9.6, 6.7, .65]]) {
      const y = terrainHeight(x, z);
      this.cyl(.075 * scale, .68 * scale, [x, y + .34 * scale, z], bark);
      for (let tier = 0; tier < 3; tier++) {
        this.mesh(new T.ConeGeometry((.49 - .095 * tier) * scale, .85 * scale, 8),
          needles, [x, y + (.72 + .29 * tier) * scale, z], 'Tree');
      }
    }
  }

  buildChannel(width, depth, slope) {
    const half = .74 * width / 12;
    const depthInScene = depth * .22;
    const drop = (slope - .01) * 4;
    const frames = this.channelFrames.map(frame => ({ ...frame, y: frame.y - drop * frame.t }));
    const wetDepth = t => T.MathUtils.lerp(1.14, depthInScene, T.MathUtils.smoothstep(t, 0, .22));
    const updateStrip = (mesh, profile) => {
      mesh.geometry.dispose();
      mesh.geometry = makeStrip(frames, profile);
    };
    updateStrip(this.channelBed, () => [[half + .14, 0], [-half - .14, 0]]);
    updateStrip(this.channelWalls, () => [
      [half + .14, -.13], [half + .14, .58], [half, .58], [half, 0],
      [-half, 0], [-half, 1.13], [-half - .14, 1.13], [-half - .14, -.13],
    ]);
    updateStrip(this.channelWater, ({ t }) => [[half, wetDepth(t)], [-half, wetDepth(t)]]);
    this.channelCurrent.curve.points.forEach((point, i, points) => {
      const t = i / (points.length - 1);
      point.y = channelRoute[i][1] - drop * t + wetDepth(t) + .045;
    });
    this.channelCurrent.curve.updateArcLengths();
    if (this.forebayWater) {
      const floor = 1.87;
      this.forebayWater.scale.y = Math.max(.1, frames.at(-1).y + depthInScene - floor);
      this.forebayWater.position.y = floor + this.forebayWater.scale.y / 2;
    }
  }

  resize() {
    super.resize();
    this.padView(.83, .88);
  }

  updateWorked(values) {
    const { width, depth, slope, area, channelFlow, riverFlow, operatingFlow, plantMW, precipitation } = values;
    this.updateWeather(precipitation);
    this.buildChannel(width, depth, slope);
    const bypassFlow = riverFlow - operatingFlow;
    // Scale the design pipe speed by the fraction of design discharge in use.
    this.speeds = { inflow: riverFlow * .004, channel: operatingFlow / area * .27,
      bypass: bypassFlow * .007, turbine: .55 * operatingFlow / channelFlow, tailrace: operatingFlow * .008 };
    this.host.dataset.channelSpeed = String(this.speeds.channel);
    this.host.dataset.bypassSpeed = String(this.speeds.bypass);
    this.result = { turbineFlowM3s: operatingFlow / 12 };
    this.state = { load: plantMW / 50, energized: plantMW > 0 };
  }
}

const upstreamBank = -9.34;
const catchmentCenter = new T.Vector2(-11.72, -1.9);
const catchmentRim = [
  [-2, -.6], [-1.48, -2.28], [-.52, -2.62], [.98, -2.14],
  [1.89, -1.2], [2.08, .16], [1.52, 1.73], [.31, 2.27], [-1.28, 1.69],
];
const catchmentElevation = (x, z) =>
  5.82 + .33 * (upstreamBank - x) + .19 * Math.cos(z) + .18 * Math.sin(x * 1.6 + z);
const headDatum = -.46;
const schematicSurfaceAtHead = meanHead => visualState({
  inputs: { grossHeadM: 48 + meanHead * 1.8, ratingMW: 1 }, plantMW: 0,
}).surfaceY;

export class ReservoirWorkedScene extends WorkedScene {
  pipe(points, radius, label, type) {
    super.pipe(points, radius, type === 'spill' ? 'Spill release outlet' : label, type);
  }

  build() {
    super.build();
    this.camera.position.set(5.6, 13.6, 24);
    this.controls.target.set(-1.75, 2.72, -.15);
    this.controls.update();
    this.buildCatchmentTerrain();
    this.addWeather({ clouds: [[-12.85, 9.9, -2.56, .94], [-10.3, 9.36, -3.22, .72]], dropHeight: 3.35 });

    const catchmentMaterial = new T.LineDashedMaterial({
      color: '#fff0a6', dashSize: .23, gapSize: .1,
    });
    this.catchment = new T.LineLoop(new T.BufferGeometry(), catchmentMaterial);
    this.catchment.renderOrder = 2;
    this.scene.add(this.catchment);
    this.catchmentAnchor = new T.Vector3();
    this.runoffAnchor = new T.Vector3();
    this.runoffArrows = [-3.24, -2.10, -.91].map(z => {
      const x = -11.36;
      const y = catchmentElevation(x, z) + .22;
      const slope = (catchmentElevation(x + 1.65, z) - catchmentElevation(x, z)) / 1.65;
      const direction = new T.Vector3(1, slope, 0).normalize();
      const arrow = new T.ArrowHelper(direction, new T.Vector3(x, y, z), 1, '#136d80', .32, .28);
      this.scene.add(arrow);
      return arrow;
    });
    this.inflow = this.surfaceCurrent([[-9.2, 3, -1.8], [-7.7, 3, -1.8], [-6.1, 3, -1.8]], 'inflow');

    this.meanMarker = this.makeHeadMarker(-8.95, '#126483');
    this.designMarker = this.makeHeadMarker(-8.18, '#936019', true);
    this.ecoAnchor = new T.Vector3(4.8, .62, 2.78);
    this.buildEcologicalOutlet();
  }

  buildCatchmentTerrain() {
    const west = -14.16;
    const north = -5.12;
    const south = .96;
    const slope = new T.PlaneGeometry(upstreamBank - west, south - north, 28, 30).rotateX(-Math.PI / 2);
    const positions = slope.attributes.position;
    const grass = this.mat('#729686', 1);
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i) + (upstreamBank + west) / 2;
      const z = positions.getZ(i) + (north + south) / 2;
      positions.setXYZ(i, x, catchmentElevation(x, z), z);
    }
    slope.computeVertexNormals();
    this.mesh(slope, grass, [0, 0, 0], 'Upstream catchment slope');
    const corners = [[west, north], [upstreamBank, north], [upstreamBank, south], [west, south]];
    const bank = this.mat('#869b8e', .96);
    corners.forEach(([x0, z0], i) => {
      const [x1, z1] = corners[(i + 1) % corners.length];
      const edge = Array.from({ length: 31 }, (_, step) => {
        const x = T.MathUtils.lerp(x0, x1, step / 30);
        const z = T.MathUtils.lerp(z0, z1, step / 30);
        return { x, y: catchmentElevation(x, z), z, lateral: new T.Vector2() };
      });
      this.mesh(makeStrip(edge, ({ y }) => [[0, 0], [0, -.65 - y]]), bank, [0, 0, 0], 'Upstream bank cutaway');
    });
  }

  buildEcologicalOutlet() {
    const green = this.mat('#398e76', .36, .27);
    const route = [[-4.4, 1.12, 1.96], [-3.85, .96, 2.44], [-2.52, .46, 2.69],
      [-.7, .27, 2.69], [1.72, .13, 2.69], [3.4, .08, 2.69], [4.7, .07, 2.69],
      [5.44, -.22, 2.67], [6.72, -.46, 2.16]];
    this.pipe(route, .17, 'Environmental release through the ecological turbine', 'ecological');
    const stream = this.streams.at(-1);
    stream.offset = .1;
    stream.particles[0].material.color.set('#83f4b5');
    this.box([1.38, .19, .78], [4.18, -.37, 2.7], this.concrete, 'Ecological turbine foundation');
    this.cyl(.35, .53, [4.18, .02, 2.7], green, 'Ecological turbine');
    this.cyl(.23, .45, [4.18, .46, 2.7], this.steel, 'Ecological generator');
    this.ecoLight = this.mat('#6acb9d', .3);
    this.ecoLight.emissive.set('#2eab83');
    this.cyl(.132, .07, [4.18, .71, 2.7], this.ecoLight, 'Ecological release indicator');
  }

  makeHeadMarker(x, color, dashed = false) {
    const material = dashed
      ? new T.LineDashedMaterial({ color, dashSize: .26, gapSize: .16 })
      : new T.LineBasicMaterial({ color });
    material.depthTest = false;
    const geometry = new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(18), 3));
    const line = new T.LineSegments(geometry, material);
    line.frustumCulled = false;
    line.renderOrder = 2;
    this.scene.add(line);
    return { line, anchor: new T.Vector3(x, 0, 2.02) };
  }

  updateHeadMarker({ line, anchor }, height) {
    const { x, z } = anchor;
    const positions = line.geometry.attributes.position;
    positions.set([x, headDatum, z, x, height, z, x - .19, height, z, x + .19, height, z,
      x - .19, headDatum, z, x + .19, headDatum, z]);
    positions.needsUpdate = true;
    line.computeLineDistances();
    anchor.y = height;
  }

  updateCatchment(area, coefficient) {
    const scale = Math.sqrt(area / 400);
    if (this.catchmentScale !== scale) {
      this.catchmentScale = scale;
      const outline = catchmentRim.map(([dx, dz]) => [
        catchmentCenter.x + dx * scale, 0, catchmentCenter.y + dz * scale,
      ]);
      const rim = pathFrames([...outline, outline[0]], 42);
      this.catchment.geometry.dispose();
      this.catchment.geometry = new T.BufferGeometry().setFromPoints(rim.map(({ x, z }) =>
        new T.Vector3(x, catchmentElevation(x, z) + .12, z)
      ));
      this.catchment.computeLineDistances();
    }
    this.catchmentAnchor.set(catchmentCenter.x, catchmentElevation(catchmentCenter.x, catchmentCenter.y) + .2, catchmentCenter.y);
    for (const arrow of this.runoffArrows) {
      arrow.setLength(.43 + 1.07 * coefficient, .21 + .12 * coefficient, .16 + .14 * coefficient);
    }
    this.runoffAnchor.set(-10.7, catchmentElevation(-10.7, -2.1) + .2, -2.1);
    this.host.dataset.catchmentScale = String(scale);
    this.host.dataset.runoffCoefficient = String(coefficient);
  }

  controlAnchor(key) {
    switch (key) {
      case 'area': return this.catchmentAnchor;
      case 'coefficient': return this.runoffAnchor;
      case 'meanhead': return this.meanMarker.anchor;
      case 'designhead': return this.designMarker.anchor;
      case 'eco': return this.ecoAnchor;
      default: return super.controlAnchor(key);
    }
  }

  resize() {
    super.resize();
    this.padView(.87, 1.04);
  }

  updateWorked({ area, precipitation, coefficient, mean, spill, ecological, usable, meanHead, designHead, operatingMW, ratingMW }) {
    super.update({ inputs: { grossHeadM: 48 + meanHead * 1.8, ratingMW },
      turbineFlowM3s: usable, spillFlowM3s: spill, plantMW: operatingMW });
    this.updateWeather(precipitation);
    this.updateCatchment(area, coefficient);
    this.updateHeadMarker(this.meanMarker, this.state.surfaceY);
    this.updateHeadMarker(this.designMarker, schematicSurfaceAtHead(designHead));
    this.ecoLight.emissiveIntensity = ecological > 0 ? 1.5 : 0;
    this.inflow.curve.points.forEach(point => { point.y = this.state.surfaceY + .055; });
    this.inflow.curve.updateArcLengths();
    this.speeds = { inflow: .22 * Math.log1p(mean), turbine: .32 * Math.log1p(usable),
      spill: .35 * Math.log1p(spill), ecological: .4 * Math.log1p(ecological) };
    this.host.dataset.mainSpeed = String(this.speeds.turbine);
    this.host.dataset.ecologicalSpeed = String(this.speeds.ecological);
    this.host.dataset.designHead = String(designHead);
  }
}
