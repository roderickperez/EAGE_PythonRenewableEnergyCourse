export const HYDRO_CONSTANTS = Object.freeze({
  efficiency: 0.85,
  densityKgM3: 1000,
  gravityMS2: 9.81,
  durationHours: 6,
});

export const DEFAULT_INPUTS = Object.freeze({
  grossHeadM: 193,
  usableFlowM3s: 10,
  headLossM: 2,
  ratingMW: 2,
});

const ranges = Object.freeze({
  grossHeadM: [5, 250],
  usableFlowM3s: [0, 30],
  headLossM: [0, 50],
  ratingMW: [0.1, 25],
});

function finiteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function normalizeHydroInputs(values = {}) {
  const grossHeadM = clamp(
    finiteNumber(values.grossHeadM, DEFAULT_INPUTS.grossHeadM),
    ...ranges.grossHeadM,
  );
  const usableFlowM3s = clamp(
    finiteNumber(values.usableFlowM3s, DEFAULT_INPUTS.usableFlowM3s),
    ...ranges.usableFlowM3s,
  );
  const maximumLoss = Math.min(ranges.headLossM[1], Math.max(0, grossHeadM - 0.5));
  const headLossM = clamp(
    finiteNumber(values.headLossM, DEFAULT_INPUTS.headLossM),
    ranges.headLossM[0],
    maximumLoss,
  );
  const ratingMW = clamp(
    finiteNumber(values.ratingMW, DEFAULT_INPUTS.ratingMW),
    ...ranges.ratingMW,
  );
  return { grossHeadM, usableFlowM3s, headLossM, ratingMW };
}

/**
 * Calculate the steady-state educational hydro model. All public values use the
 * units encoded in their property names.
 */
export function computeHydro(values = DEFAULT_INPUTS) {
  const inputs = normalizeHydroInputs(values);
  const { efficiency, densityKgM3, gravityMS2, durationHours } = HYDRO_CONSTANTS;
  const netHeadM = Math.max(0, inputs.grossHeadM - inputs.headLossM);
  const hydraulicMW = densityKgM3 * gravityMS2 * inputs.usableFlowM3s * netHeadM / 1e6;
  const uncappedMW = efficiency * hydraulicMW;
  const plantMW = Math.min(uncappedMW, inputs.ratingMW);
  const flowForRating = netHeadM > 0
    ? inputs.ratingMW * 1e6 / (efficiency * densityKgM3 * gravityMS2 * netHeadM)
    : 0;
  const turbineFlowM3s = Math.min(inputs.usableFlowM3s, flowForRating);
  const spillFlowM3s = Math.max(0, inputs.usableFlowM3s - turbineFlowM3s);
  return {
    inputs,
    netHeadM,
    hydraulicMW,
    uncappedMW,
    plantMW,
    turbineFlowM3s,
    spillFlowM3s,
    energy6hMWh: plantMW * durationHours,
    clipped: uncappedMW > inputs.ratingMW + 1e-9,
  };
}
