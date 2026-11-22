export const RAD = Math.PI / 180;
export const YEAR_H = 8760;
const DRY_AIR_R = 287.05;
export const BETZ = 16 / 27;

export function dryAirDensity(pressureKPa, temperatureC) {
  return pressureKPa * 1000 / (DRY_AIR_R * (temperatureC + 273.15));
}

export function powerLawWind(speed, height, referenceHeight, exponent) {
  return speed * (height / referenceHeight) ** exponent;
}

export function logLawWind(speed, height, referenceHeight, roughness, displacement) {
  return speed * Math.log((height - displacement) / roughness) /
    Math.log((referenceHeight - displacement) / roughness);
}

export function twoIntervalWind(first, second) {
  const mean = (first + second) / 2;
  const deviation = Math.abs(second - first) / 2;
  const meanCubedSpeeds = (first ** 3 + second ** 3) / 2;
  return { mean, deviation, intensity: mean > 0 ? deviation / mean : null,
    meanCubedSpeeds, cubedMeanSpeed: mean ** 3 };
}

export function weibullPowerBins({ shape, scale, count, maximum, turbine }) {
  const cdf = speed => -Math.expm1(-((speed / scale) ** shape));
  const width = maximum / count;
  const bins = Array.from({ length: count }, (_, index) => {
    const lower = index * width, upper = (index + 1) * width;
    const probability = cdf(upper) - cdf(lower);
    return { lower, upper, probability, contribution: electricalPower((lower + upper) / 2, turbine) * probability };
  });
  return { bins, meanMW: bins.reduce((sum, bin) => sum + bin.contribution, 0), tail: 1 - cdf(maximum) };
}

export function annualEnergyScenarios(gross, availability, wake, other, relativeDeviation) {
  const afterAvailability = gross * availability;
  const afterWake = afterAvailability * (1 - wake);
  const p50 = afterWake * (1 - other);
  const deviation = p50 * relativeDeviation;
  return { afterAvailability, afterWake, p50, deviation, p90: p50 - 1.2815515655 * deviation };
}

export function availableWind(speed, density, radius) {
  const area = Math.PI * radius ** 2;
  const massFlow = density * area * speed;
  const specificEnergy = speed ** 2 / 2;
  return { area, massFlow, specificEnergy, windMW: massFlow * specificEnergy / 1e6 };
}

export function aerodynamicConversion(windMW, powerCoefficient, drivetrainEfficiency) {
  const rotorMW = windMW * Math.min(powerCoefficient, BETZ);
  const electricalMW = rotorMW * drivetrainEfficiency;
  return { rotorMW, electricalMW, drivetrainLossMW: rotorMW - electricalMW, betzMW: windMW * BETZ };
}

export function electricalPower(speed, { cutIn, ratedSpeed, cutOut, ratedMW }) {
  if (speed <= cutIn || speed >= cutOut) return 0;
  if (speed >= ratedSpeed) return ratedMW;
  return ratedMW * (speed ** 3 - cutIn ** 3) / (ratedSpeed ** 3 - cutIn ** 3);
}

export function powerModel(input) {
  if (input.outputMode !== 'aero') return speed => electricalPower(speed, input);
  const density = dryAirDensity(input.pressure, input.temperature);
  const factor = .5 * density * Math.PI * input.radius ** 2 * Math.min(input.cp, BETZ) * input.efficiency / 1e8;
  return speed => speed <= input.cutIn || speed >= input.cutOut ? 0 : Math.min(input.ratedMW, factor * speed ** 3);
}

// The course curve has its greatest P/v³ ratio at rated speed. The electrical
// bound includes the selected drivetrain efficiency, as does shaft torque.
export function curveFeasibility(input, point) {
  if (input.outputMode !== 'curve') return { ratedViolation: false, currentViolation: false };
  const idealElectrical = speed => availableWind(speed, point.density, input.radius).windMW * BETZ * input.efficiency / 100;
  const exceeds = (power, limit) => power > limit + 1e-12 * Math.max(1, Math.abs(power), Math.abs(limit));
  return { ratedViolation: exceeds(input.ratedMW, idealElectrical(input.ratedSpeed)),
           currentViolation: exceeds(point.gridMW, idealElectrical(point.speed)) };
}

export function operatingPoint(input) {
  const ratio = input.hub / input.reference;
  const logRatio = Math.log((input.hub - input.displacement) / input.roughness) /
    Math.log((input.reference - input.displacement) / input.roughness);
  const shear = input.heightMode === 'power' ? ratio ** input.alpha : input.heightMode === 'log' ? logRatio : 1;
  const speed = input.speed * shear;
  const density = dryAirDensity(input.pressure, input.temperature);
  const { area, massFlow, windMW } = availableWind(speed, density, input.radius);
  const { rotorMW, electricalMW: aeroMW } = aerodynamicConversion(windMW, input.cp, input.efficiency / 100);
  const gridMW = powerModel(input)(speed);
  const actualRatedSpeed = input.outputMode === 'aero' && aeroMW > 0 ?
    Math.max(input.cutIn, speed * Math.cbrt(input.ratedMW / aeroMW)) : input.ratedSpeed;
  const region = speed <= input.cutIn ? 'Below or at cut-in' :
    speed >= input.cutOut ? 'Cut-out shutdown' : gridMW === input.ratedMW ? 'Rated output' : gridMW > 0 ? 'Generating' : 'Zero generation';
  // Illustrative variable-speed operation, limited to rated-speed rotor RPM.
  const omega = gridMW > 0 ? input.tsr * Math.min(speed, actualRatedSpeed) / input.radius : 0;
  return { speed, density, area, massFlow, windMW, rotorMW, aeroMW, gridMW, region, omega,
    rpm: omega * 60 / (2 * Math.PI), tsr: speed ? omega * input.radius / speed : null,
    torque: omega ? gridMW * 1e6 / (input.efficiency / 100 * omega) : null,
    frictionVelocity: input.speed * .4 / Math.log((input.reference - input.displacement) / input.roughness) };
}

export function annualEnergy(input) {
  // Only the generating range contributes. Use exact Weibull bin probabilities.
  const survival = speed => Math.exp(-((speed / input.scale) ** input.shape));
  const powerAt = powerModel(input);
  const count = Math.ceil((input.cutOut - input.cutIn) / .05);
  const step = (input.cutOut - input.cutIn) / count;
  let meanMW = 0;
  for (let i = 0; i < count; i++) {
    const lower = input.cutIn + i * step;
    const upper = lower + step;
    meanMW += powerAt((lower + upper) / 2) * (survival(lower) - survival(upper));
  }
  const gross = YEAR_H * meanMW;
  const retained = input.availability / 100 * (1 - input.wake / 100) * (1 - input.loss / 100);
  const net = gross * retained;
  return { meanMW, gross, retained, net, capacityFactor: net / (YEAR_H * input.ratedMW) };
}

export function circularMean(a, b) {
  const sine = (Math.sin(a * RAD) + Math.sin(b * RAD)) / 2;
  const cosine = (Math.cos(a * RAD) + Math.cos(b * RAD)) / 2;
  const length = Math.hypot(sine, cosine);
  return { length, degrees: length < 1e-10 ? null : (Math.atan2(sine, cosine) / RAD + 360) % 360 };
}
