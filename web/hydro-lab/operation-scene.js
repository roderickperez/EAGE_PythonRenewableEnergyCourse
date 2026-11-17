import * as T from 'three';

export function createOperationScene(kind, { scene, material, mesh, box, tank, level, steel, rock, palette }) {
  const blue = material(palette.water);
  const teal = material(palette.net);
  const bronze = material(palette.shaft);
  const waterDot = new T.MeshBasicMaterial({ color: '#9af1ff' });
  const bypassDot = new T.MeshBasicMaterial({ color: '#b3f1d9' });
  const gold = new T.MeshBasicMaterial({ color: palette.electricity });
  const bead = new T.SphereGeometry(.078, 12, 8);
  const makeParticles = (count, tint) => Array.from({ length: count }, () => mesh(bead, tint, 0, 0, 0));
  const path = points => new T.CatmullRomCurve3(points.map(([x, y, z]) => new T.Vector3(x, y, z)));
  function pipe(route, radius, tint) {
    return mesh(new T.TubeGeometry(route, 48, radius, 18, false), tint, 0, 0, 0);
  }
  function move(particles, route, phase, active, offset = .13) {
    for (const [index, particle] of particles.entries()) {
      particle.visible = active;
      if (active) {
        particle.position.copy(route.getPointAt((phase + index / particles.length) % 1));
        particle.position.z += offset;
      }
    }
  }
  function cylinderX(radius, width, tint, x, y, z) {
    const cylinder = mesh(new T.CylinderGeometry(radius, radius, width, 28), tint, x, y, z);
    cylinder.rotation.z = Math.PI / 2;
    return cylinder;
  }

  box(8.9, .18, 3.05, rock, 0, -.09, -.15);
  if (kind === 'hydraulic-loss') {
    box(1.1, .18, .85, steel, -2.85, .11, .12);
    box(.3, 1.13, .36, rock, -2.85, .69, .12);
    box(1.1, .18, .85, steel, .35, .11, .12);
    box(.3, 1.13, .36, rock, .35, .69, .12);
    const pipeY = 1.34;
    const tapSpan = 2.05;
    const glass = new T.MeshStandardMaterial({
      color: '#b4d6d9', transparent: true, opacity: .28, depthWrite: false, side: T.DoubleSide,
    });
    function tap(x, z, tint, name) {
      const connector = mesh(new T.CylinderGeometry(.112, .112, .35, 14), steel, x, pipeY + .16, z);
      const tube = mesh(new T.CylinderGeometry(.105, .105, tapSpan, 18, 1, true), glass,
        x, pipeY + tapSpan / 2, z);
      const column = mesh(new T.CylinderGeometry(.078, .078, 1, 18), tint, x, pipeY, z);
      tube.renderOrder = 2;
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 64;
      const context = canvas.getContext('2d');
      context.fillStyle = '#f8fcfa';
      context.beginPath();
      context.arc(32, 32, 29, 0, 2 * Math.PI);
      context.fill();
      context.fillStyle = '#24584e';
      context.font = 'bold 40px system-ui';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(name, 32, 34);
      const texture = new T.CanvasTexture(canvas);
      texture.colorSpace = T.SRGBColorSpace;
      const label = new T.Sprite(new T.SpriteMaterial({ map: texture }));
      label.position.set(x, pipeY + tapSpan + .21, z);
      label.scale.set(.38, .38, 1);
      scene.add(label);
      return {
        moveTo(nextX, nextZ) {
          for (const item of [connector, tube, column, label]) {
            item.position.x = nextX;
            item.position.z = nextZ;
          }
        },
        setHead(head, scale) { level(column, head * scale, pipeY); },
      };
    }
    const inlet = tap(-3.38, .12, blue, 'A');
    const outlet = tap(1.2, 1.23, teal, 'B');
    const fitting = mesh(new T.TorusGeometry(1, .085, 12, 28), material('#8a645f'), 0, pipeY, .25);
    fitting.rotation.y = Math.PI / 2;
    const flow = makeParticles(15, waterDot);
    const cutaway = steel.clone();
    cutaway.transparent = true;
    cutaway.opacity = .26;
    cutaway.depthWrite = false;
    let route, shell, lining, lastLength, lastDiameter;
    return ([length, velocity, diameter, inletHead, outletHead], phase) => {
      if (lastLength !== length || lastDiameter !== diameter) {
        const end = .78 + 1.85 * length / 1000;
        route = path([[-3.8, pipeY, .12], [-1.4, pipeY, .12], [end - .52, pipeY, .12],
          [end, pipeY, .25], [end + .19, pipeY, .65], [end + .23, pipeY, 1.38]]);
        for (const old of [shell, lining]) {
          if (!old) continue;
          old.geometry.dispose();
          old.removeFromParent();
        }
        const radius = .12 + diameter * .057;
        shell = pipe(route, radius, cutaway);
        lining = pipe(route, radius * .72, blue);
        fitting.position.set(end - .22, pipeY, .15);
        fitting.scale.setScalar(radius * 1.08);
        outlet.moveTo(end + .22, 1.18);
        lastLength = length;
        lastDiameter = diameter;
      }
      const scale = tapSpan * .91 / Math.max(5, inletHead);
      inlet.setHead(inletHead, scale);
      outlet.setHead(outletHead, scale);
      move(flow, route, phase, velocity > 0, .17 + diameter * .036);
    };
  }

  const shaftY = 1.37;
  const shaftZ = .76;
  const tailwater = tank(3.25, 1.65, .73, 1.5);
  level(tailwater, .56);
  const center = new T.Vector3(-.43, shaftY, shaftZ);
  const turbinePath = [
    path([[-4, 1.95, .72], [-2.35, 1.95, .72], [-1.02, 1.91, .66], [-.46, 1.9, .69],
      [-.33, .94, .85], [.75, .53, .3], [2.7, .46, .1]]),
    path([[-4, 1.53, .5], [-2.55, 1.53, .5], [-1.1, 1.68, .92], [-.5, 1.48, 1.17],
      [-.43, .88, .86], [.75, .53, .3], [2.7, .46, .1]]),
    path([[-4, shaftY, shaftZ], [-2.27, shaftY, shaftZ], [-.9, shaftY, shaftZ], [.03, shaftY, shaftZ],
      [.62, .69, .5], [1.45, .48, .15], [2.7, .46, .1]]),
  ];
  const routeTubes = turbinePath.map(route => pipe(route, .13, blue));
  const bypass = path([[-3.9, 1.31, -.35], [-2.95, 2.42, -.8], [-2.15, 2.5, -.82],
    [.85, 2.5, -.82], [2.7, 2.38, -.74], [3.45, 1.86, -.5], [3.45, .55, -.48]]);
  pipe(bypass, .11, teal);
  const conduit = path([[2.18, 1.47, .76], [2.65, 1.74, .7], [3.13, 1.89, .56], [4.2, 1.93, .55]]);
  pipe(conduit, .04, steel);
  const rotor = new T.Group();
  rotor.position.copy(center);
  scene.add(rotor);
  const ring = new T.Mesh(new T.TorusGeometry(.55, .055, 12, 36), steel);
  ring.rotation.y = Math.PI / 2;
  rotor.add(ring);
  const profiles = [10, 8, 4].map((count, family) => {
    const group = new T.Group();
    const geometry = family === 0
      ? new T.SphereGeometry(.125, 14, 10)
      : new T.BoxGeometry(.14, family === 1 ? .34 : .45, family === 1 ? .13 : .18);
    for (let index = 0; index < count; index++) {
      const angle = index * 2 * Math.PI / count;
      const radius = [.47, .33, .26][family];
      const blade = new T.Mesh(geometry, bronze);
      blade.position.set(0, radius * Math.cos(angle), radius * Math.sin(angle));
      blade.rotation.x = angle + .23;
      if (family === 0) blade.scale.set(.72, .62, 1);
      group.add(blade);
    }
    rotor.add(group);
    return group;
  });
  cylinderX(.11, 1.53, steel, .46, shaftY, shaftZ);
  cylinderX(.46, 1.07, bronze, 1.65, shaftY, shaftZ);
  for (const x of [1.24, 1.64, 2.03]) cylinderX(.48, .065, steel, x, shaftY, shaftZ);
  box(1.3, .78, 1.18, rock, 1.65, .39, shaftZ);
  const generatingFlow = makeParticles(16, waterDot);
  const bypassFlow = makeParticles(14, bypassDot);
  const electricity = makeParticles(6, gold);
  return ([family, turbineQ, bypassQ, power], phase) => {
    profiles.forEach((profile, index) => { profile.visible = index === family; });
    routeTubes.forEach((tube, index) => { tube.visible = index === family; });
    if (turbineQ > 0) rotor.rotation.x = phase * Math.PI * 4;
    move(generatingFlow, turbinePath[family], phase, turbineQ > 0, .15);
    move(bypassFlow, bypass, phase, bypassQ > 0, .11);
    move(electricity, conduit, (phase * 2) % 1, power > 0, .05);
  };
}
