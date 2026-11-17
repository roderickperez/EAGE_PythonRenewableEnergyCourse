// Canadian Solar CS5P-220M reference parameters from pvlib's single-diode example.
const reference = { lightCurrent: 5.114, saturation: 8.196e-10, alpha: .004539,
  diodeVoltage: 2.6373, series: 1.065, shunt: 381.68, bandgap: 1.121, kelvin: 298.15 };
const boltzmannEV = 8.617333262e-5;

function bisect(low, high, aboveZero) {
  for (let i = 0; i < 40; i++) {
    const middle = (low + high) / 2;
    if (aboveZero(middle)) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
}

export function moduleCurve(irradiance, cellTemperatureC) {
  if (irradiance === 0) {
    return { isc: 0, voc: 0, peak: null, samples: [[0, 0, 0]] };
  }

  const temperature = cellTemperatureC + 273.15;
  const delta = temperature - reference.kelvin;
  const ratio = temperature / reference.kelvin;
  const lightCurrent = irradiance / 1000 * (reference.lightCurrent + reference.alpha * delta);
  const saturation = reference.saturation * ratio ** 3 * Math.exp(
    (reference.bandgap / reference.kelvin - reference.bandgap * (1 - .0002677 * delta) / temperature) / boltzmannEV,
  );
  const a = reference.diodeVoltage * ratio;
  const shunt = reference.shunt * 1000 / irradiance;
  // Parameterize by voltage across the diode and shunt to evaluate current directly.
  const current = diodeVoltage => lightCurrent - saturation * Math.expm1(diodeVoltage / a) - diodeVoltage / shunt;
  const voltage = diodeVoltage => diodeVoltage - reference.series * current(diodeVoltage);
  const voc = bisect(0, a * Math.log1p(lightCurrent / saturation), v => current(v) > 0);
  const shortCircuit = bisect(0, voc, v => voltage(v) < 0);
  const isc = current(shortCircuit);
  const power = diodeVoltage => voltage(diodeVoltage) * current(diodeVoltage);

  let low = shortCircuit, high = voc;
  for (let i = 0; i < 42; i++) {
    const left = (2 * low + high) / 3;
    const right = (low + 2 * high) / 3;
    if (power(left) < power(right)) low = left;
    else high = right;
  }
  const optimum = (low + high) / 2;
  const peak = { voltage: voltage(optimum), current: current(optimum), power: power(optimum) };
  const diodeSamples = Array.from({ length: 95 }, (_, i) => shortCircuit + (voc - shortCircuit) * (i + 1) / 96);
  diodeSamples.push(optimum);
  diodeSamples.sort((a, b) => a - b);
  const samples = [[0, isc, 0], ...diodeSamples.map(v => [voltage(v), current(v), power(v)]), [voc, 0, 0]];
  return { isc, voc, peak, samples };
}
