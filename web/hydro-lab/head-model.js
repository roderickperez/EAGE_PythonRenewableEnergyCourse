export function computeHead({ elevation, pressureKPa, velocity, density, gravity }) {
  const pressure = 1000 * pressureKPa / (density * gravity);
  const kinetic = velocity ** 2 / (2 * gravity);
  return { elevation, pressure, kinetic, total: elevation + pressure + kinetic };
}
