export const RAD = Math.PI / 180;
export const STC = 1000;

// East, up and north components in local solar time (no civil-time offset).
export function sunDirection(hour, latitude, declination) {
  const lat = latitude * RAD;
  const dec = declination * RAD;
  const h = (hour - 12) * 15 * RAD;
  return [
    -Math.cos(dec) * Math.sin(h),
    Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(h),
    Math.cos(lat) * Math.sin(dec) - Math.sin(lat) * Math.cos(dec) * Math.cos(h),
  ];
}

export function solarPower(input, hour = input.hour) {
  const sun = sunDirection(hour, input.latitude, input.declination);
  const elevation = Math.asin(Math.max(-1, Math.min(1, sun[1]))) / RAD;
  const tilt = input.tilt * RAD;
  const az = input.azimuth * RAD;
  const normal = [Math.sin(tilt) * Math.sin(az), Math.cos(tilt), Math.sin(tilt) * Math.cos(az)];
  const incidence = Math.max(-1, Math.min(1, sun.reduce((dot, component, i) => dot + component * normal[i], 0)));
  const up = Math.max(0, sun[1]);
  // Synthetic daylight profile. Sliders set the beam and diffuse scales.
  const dni = input.dni * Math.min(1, 4 * up);
  const dhi = input.dhi * up;
  const ghi = dni * up + dhi;
  const beam = dni * Math.max(0, incidence);
  const sky = dhi * (1 + normal[1]) / 2;
  const ground = input.albedo * ghi * (1 - normal[1]) / 2;
  const poa = beam + sky + ground;
  const rated = input.area * STC * input.efficiency / 100;
  const cell = input.ambient + poa / 800 * (input.noct - 20);
  const tempFactor = Math.max(0, 1 + input.coefficient / 100 * (cell - 25));
  const areaPower = poa * input.area * input.efficiency / 100;
  const retained = (1 - input.soiling / 100) * (1 - input.wiring / 100);
  const dc = areaPower * tempFactor;
  const potential = dc * retained * input.inverterEfficiency / 100;
  const ac = Math.min(potential, input.rating);
  return { sun, normal, elevation, incidence, dni, dhi, ghi, beam, sky, ground, poa, rated, cell, tempFactor,
    areaPower, retained, dc, potential, ac, clip: potential - ac, conversionLoss: dc * retained - potential };
}

export function solarDay(input) {
  const dt = 1 / 12; // Five-minute midpoint integration, including night.
  const samples = Array.from({ length: 288 }, (_, i) => solarPower(input, (i + .5) * dt));
  const integrate = key => samples.reduce((sum, sample) => sum + sample[key], 0) * dt / 1000;
  const energy = integrate('ac');
  const irradiation = integrate('poa');
  const yieldFinal = energy / (samples[0].rated / 1000);
  const yieldRef = irradiation / (STC / 1000);
  return { samples, energy, irradiation, yieldFinal, yieldRef, pr: yieldRef ? yieldFinal / yieldRef : null,
    cf: energy / (input.rating / 1000 * 24), clipping: integrate('clip') };
}

export function batteryDay(samples, { load, storage, batteryPower, batteryEfficiency }) {
  const dt = 1 / 12;
  const efficiency = batteryEfficiency / 100;
  let stored = 0, unmet = 0, surplus = 0, used = 0;
  for (const { ac } of samples) {
    const direct = Math.min(ac, load);
    const charging = Math.min(ac - direct, batteryPower, Math.max(0, (storage - stored) / (efficiency * dt)));
    const discharge = Math.min(load - direct, batteryPower, Math.max(0, stored * efficiency / dt));
    stored += dt * (efficiency * charging - discharge / efficiency);
    unmet += dt * (load - direct - discharge);
    surplus += dt * (ac - direct - charging);
    used += dt * (direct + discharge);
  }
  return { stored: stored / 1000, unmet: unmet / 1000, surplus: surplus / 1000, used: used / 1000 };
}

export function lifetime({ firstYear, degradation, years, capital, maintenance, discount }) {
  let energy = 0, discountedEnergy = 0, discountedCost = capital, lastYear;
  for (let year = 1; year <= years; year++) {
    const factor = (1 + discount / 100) ** year;
    lastYear = firstYear * (1 - degradation / 100) ** (year - 1);
    energy += lastYear;
    discountedEnergy += lastYear / factor;
    discountedCost += maintenance / factor;
  }
  return { energy, lastYear, lcoe: discountedCost / discountedEnergy };
}
