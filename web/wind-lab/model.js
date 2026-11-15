export const RAD = Math.PI / 180;
export const YEAR_H = 8760;
const DRY_AIR_R = 287.05;
const BETZ = 16 / 27;

export function electricalPower(speed, { cutIn, ratedSpeed, cutOut, ratedMW }) {
  if (speed <= cutIn || speed >= cutOut) return 0;
  if (speed >= ratedSpeed) return ratedMW;
  return ratedMW * (speed ** 3 - cutIn ** 3) / (ratedSpeed ** 3 - cutIn ** 3);
}

export function powerModel(input) {
  if (input.outputMode !== 'aero') return speed => electricalPower(speed, input);
  const density = input.pressure * 1000 / (DRY_AIR_R * (input.temperature + 273.15));
  const factor = .5 * density * Math.PI * input.radius ** 2 * Math.min(input.cp, BETZ) * input.efficiency / 1e8;
  return speed => speed <= input.cutIn || speed >= input.cutOut ? 0 : Math.min(input.ratedMW, factor * speed ** 3);
}

export function operatingPoint(input) {
  const ratio = input.hub / input.reference;
  const logRatio = Math.log((input.hub - input.displacement) / input.roughness) /
    Math.log((input.reference - input.displacement) / input.roughness);
  const shear = input.heightMode === 'power' ? ratio ** input.alpha : input.heightMode === 'log' ? logRatio : 1;
  const speed = input.speed * shear;
  const density = input.pressure * 1000 / (DRY_AIR_R * (input.temperature + 273.15));
  const area = Math.PI * input.radius ** 2;
  const massFlow = density * area * speed;
  const windMW = massFlow * speed ** 2 / 2e6;
  const rotorMW = windMW * Math.min(input.cp, BETZ);
  const aeroMW = rotorMW * input.efficiency / 100;
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
