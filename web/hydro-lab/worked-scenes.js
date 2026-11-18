import * as T from 'three';
import { HydroScene } from './scene.js';

class WorkedScene extends HydroScene {
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
    this.landscape();
    const ground = this.mat('#91a79a');
    this.box([19, .5, 10], [0, -.9, -.55], ground, 'River valley and foundations');
    this.box([6, 2.68, 4.6], [-6.1, .65, -.6], ground, 'Raised upstream riverbed');
    this.box([6, .65, 4.6], [-6.1, 2.31, -.6], this.water, 'River upstream of the diversion weir');
    this.box([.42, 2.75, 4.8], [-3.2, .59, -.6], this.concrete, 'Diversion weir');
    this.box([6.4, .24, 1.45], [-.1, -.43, 3.7], this.water, 'Bypassed river reach');
    this.box([5.9, .24, 3], [6, -.43, 2.95], this.water, 'Tailrace');
    this.surfaceCurrent([[-9, 2.665, -.4], [-6, 2.665, -.4], [-3.5, 2.665, -.4]], 'inflow');

    this.channel = new T.Group();
    this.channel.position.set(0, 1.29, .35);
    this.scene.add(this.channel);
    this.box([7.2, .23, 3.45], [0, 0, 0], this.concrete, 'Concrete channel bed', this.channel);
    for (const [z, height] of [[-1.66, 1.25], [1.66, .45]]) {
      this.box([7.2, height, .17], [0, height / 2, z], this.cut, 'Open channel wall (front cut away)', this.channel);
    }
    this.channelWater = this.box([7.1, 1, 3.16], [0, .5, 0], this.water, 'Open channel water', this.channel);
    this.channelCurrents = [-.7, .05, .8].map(z => this.surfaceCurrent([[-3.45, 2, z], [3.45, 2, z]], 'channel', 12));

    this.box([.5, 1.6, 3.4], [3.6, 1.09, .35], this.concrete, 'Forebay and penstock intakes');
    for (const z of [.65, 1.45]) {
      this.pipe([[3.5, 1.55, z], [4.25, 1.35, z], [5.25, .65, z], [6.55, .41, z]], .27, 'One of two steel penstocks');
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
    this.pipe([[6.65, .3, 1.05], [6.8, -.08, 1.75], [7.2, -.24, 2.7]], .24, 'Turbine discharge', 'tailrace');
    this.box([.85, .65, .73], [8.25, .1, .05], this.steel, 'Transformer');
    this.energyPath([[6.65, 1.65, 1.05], [7.35, 1.35, .45], [8.25, .5, .05]], 'Electrical output');
  }

  updateWorked(values) {
    const { width, depth, area, channelFlow, riverFlow, operatingFlow, plantMW } = values;
    const waterDepth = depth * .22;
    this.channel.scale.z = width / 12;
    this.channelWater.scale.y = waterDepth;
    this.channelWater.position.y = .12 + waterDepth / 2;
    this.channelCurrents.forEach((stream, index) => {
      const z = .35 + [-.7, .05, .8][index] * width / 12;
      stream.curve.points.forEach(point => point.set(point.x, 1.44 + waterDepth, z));
      stream.curve.updateArcLengths();
    });
    // Scale the design pipe speed by the fraction of design discharge in use.
    this.speeds = { inflow: riverFlow * .004, channel: operatingFlow / area * .27, turbine: .55 * operatingFlow / channelFlow,
      tailrace: operatingFlow * .008 };
    this.host.dataset.channelSpeed = String(this.speeds.channel);
    this.result = { turbineFlowM3s: operatingFlow / 12 };
    this.state = { load: plantMW / 50, energized: plantMW > 0 };
  }
}

export class ReservoirWorkedScene extends WorkedScene {
  pipe(points, radius, label, type) {
    super.pipe(points, radius, type === 'spill' ? 'Spill release outlet' : label, type);
  }

  build() {
    super.build();
    this.pipe([[-4.35, 1.2, -.9], [-2, .83, -.9], [.6, .25, -.9], [2.4, -.29, -1.3], [6.6, -.35, -1.3]],
      .16, 'Environmental release outlet', 'ecological');
    this.inflow = this.surfaceCurrent([[-9.2, 3, -1.8], [-7.7, 3, -1.8], [-6.1, 3, -1.8]], 'inflow');
  }

  updateWorked({ mean, spill, ecological, usable, meanHead, operatingMW, ratingMW }) {
    super.update({ inputs: { grossHeadM: 48 + meanHead * 1.8, ratingMW },
      turbineFlowM3s: usable, spillFlowM3s: spill, plantMW: operatingMW });
    this.inflow.curve.points.forEach(point => { point.y = this.state.surfaceY + .055; });
    this.inflow.curve.updateArcLengths();
    this.speeds = { inflow: .22 * Math.log1p(mean), turbine: .32 * Math.log1p(usable),
      spill: .35 * Math.log1p(spill), ecological: .4 * Math.log1p(ecological) };
    this.host.dataset.mainSpeed = String(this.speeds.turbine);
  }
}
