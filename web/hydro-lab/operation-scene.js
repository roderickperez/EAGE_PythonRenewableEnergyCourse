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
    box(1.1, .18, .85, steel, -2.85, .11, .48);
    box(.3, 1.2, .36, rock, -2.85, .72, .48);
    box(1.1, .18, .85, steel, .35, .11, .48);
    box(.3, 1.2, .36, rock, .35, .72, .48);
    const gaugeBase = .35;
    box(.98, .20, .86, steel, 3.14, gaugeBase - .1, .42);
    box(.45, 2.85, .22, material('#c4d9d0'), 3.14, gaugeBase + 1.425, .2);
    const frictionBar = box(.46, 1, .19, material(palette.loss), 3.14, 0, .39);
    const fittingBar = box(.46, 1, .19, material('#8a645f'), 3.14, 0, .39);
    const flow = makeParticles(15, waterDot);
    const cutaway = steel.clone();
    cutaway.transparent = true;
    cutaway.opacity = .26;
    cutaway.depthWrite = false;
    let route, shell, lining, lastLength, lastDiameter;
    return ([length, velocity, diameter, friction, fittings], phase) => {
      if (lastLength !== length || lastDiameter !== diameter) {
        const end = .7 + 1.45 * length / 1000;
        route = path([[-3.8, 1.58, .4], [-1.4, 1.58, .4], [end - .4, 1.58, .4],
          [end, 1.53, .57], [end + .22, 1.15, .75], [end + .22, .94, .75]]);
        for (const old of [shell, lining]) {
          if (!old) continue;
          old.geometry.dispose();
          old.removeFromParent();
        }
        const radius = .12 + diameter * .057;
        shell = pipe(route, radius, cutaway);
        lining = pipe(route, radius * .72, blue);
        lastLength = length;
        lastDiameter = diameter;
      }
      const total = friction + fittings;
      const scale = 2.75 * Math.log1p(total) / Math.log1p(4000);
      const frictionHeight = total > 0 ? scale * friction / total : 0;
      level(frictionBar, frictionHeight, gaugeBase);
      level(fittingBar, scale - frictionHeight, gaugeBase + frictionHeight);
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
