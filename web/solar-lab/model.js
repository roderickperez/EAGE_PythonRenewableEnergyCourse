export const RAD = Math.PI / 180;
export const STC = 1000;

// Angles in degrees, with azimuth clockwise from north.
export function solarVector(angleFromUp, azimuth) {
  const angle = angleFromUp * RAD;
  const heading = azimuth * RAD;
  return [Math.sin(angle) * Math.sin(heading), Math.cos(angle), Math.sin(angle) * Math.cos(heading)];
}

export function incidenceCosine(sun, normal) {
  const projection = sun.reduce((dot, component, i) => dot + component * normal[i], 0);
  return Math.max(-1, Math.min(1, projection));
}

export function beamOnPlane(dni, cosine) {
  // Treat a grazing ray as zero despite roundoff in the trigonometry.
  return cosine > 1e-12 ? dni * cosine : 0;
}

export function isotropicIrradiance(dni, dhi, albedo, sun, normal) {
  const ghi = beamOnPlane(dni, sun[1]) + dhi;
  const incidence = incidenceCosine(sun, normal);
  const beam = beamOnPlane(dni, incidence);
  const sky = dhi * (1 + normal[1]) / 2;
  const ground = albedo * ghi * (1 - normal[1]) / 2;
  return { ghi, incidence, beam, sky, ground, poa: beam + sky + ground };
}

export function intervalIrradiation(intervals) {
  return intervals.reduce((total, { irradiance, hours }) => total + irradiance * hours, 0) / 1000;
}

export function estimateCellTemperature(irradiance, ambient, noct) {
  return ambient + irradiance / 800 * (noct - 20);
}

export function temperatureMultiplier(cellTemperature, coefficient) {
  return Math.max(0, 1 + coefficient * (cellTemperature - 25));
}

export function retainedFraction(...losses) {
  return losses.reduce((fraction, loss) => fraction * (1 - loss), 1);
}

export function inverterOutput(dcAtInverter, efficiency, rating) {
  const potential = dcAtInverter * efficiency;
  const ac = Math.min(potential, rating);
  return { potential, ac, clip: potential - ac, conversionLoss: dcAtInverter - potential };
}

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
  const normal = solarVector(input.tilt, input.azimuth);
  const up = Math.max(0, sun[1]);
  // Synthetic daylight profile. Sliders set the beam and diffuse scales.
  const dni = input.dni * Math.min(1, 4 * up);
  const dhi = input.dhi * up;
  const { ghi, incidence, beam, sky, ground, poa } = isotropicIrradiance(dni, dhi, input.albedo, sun, normal);
  const rated = input.area * STC * input.efficiency / 100;
  const cell = estimateCellTemperature(poa, input.ambient, input.noct);
  const tempFactor = temperatureMultiplier(cell, input.coefficient / 100);
  const areaPower = poa * input.area * input.efficiency / 100;
  const retained = retainedFraction(input.soiling / 100, input.wiring / 100);
  const dc = areaPower * tempFactor;
  const inverter = inverterOutput(dc * retained, input.inverterEfficiency / 100, input.rating);
  return { sun, normal, elevation, incidence, dni, dhi, ghi, beam, sky, ground, poa, rated, cell, tempFactor,
    areaPower, retained, dc, ...inverter };
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
