const hoursPerYear = 8760;
const secondsPerYear = hoursPerYear * 3600;

function channelDesign({ width, depth, slope }) {
  const gradient = slope / 100;
  const area = width * depth;
  const radius = area / (width + 2 * depth);
  const velocity = radius ** (2 / 3) * Math.sqrt(gradient) / .014;
  const channelFlow = area * velocity;
  const pipeRadius = Math.sqrt((channelFlow / 5 / 2) / Math.PI);
  const pipeGradient = .012 ** 2 * 5 ** 2 / (pipeRadius / 2) ** (4 / 3);
  const channelLoss = 2500 * gradient + 1;
  const pipeLoss = 500 * pipeGradient + .5;
  const netHead = 50e6 / (1000 * 9.81 * channelFlow * .92 * .97);
  return { area, radius, velocity, channelFlow, channelLoss, pipeLoss, netHead, grossHead: netHead + channelLoss + pipeLoss };
}

export const diversionSiteHead = channelDesign({ width: 12, depth: 4.5, slope: .01 }).grossHead;

export function diversionCase(inputs) {
  const design = channelDesign(inputs);
  const catchment = inputs.flow * secondsPerYear / (inputs.precip / 1000 * inputs.runoff) / 1e6;
  const operatingFlow = Math.min(inputs.flow, design.channelFlow);
  // A fixed-site comparison uses the selected case's computed design losses.
  // Sizing the head afresh to hit 50 MW at each setting would mask sensitivity.
  const operatingHead = Math.max(0, diversionSiteHead - design.channelLoss - design.pipeLoss);
  const plantMW = Math.min(50, 9.81 * operatingFlow * operatingHead * .92 * .97 / 1000);
  return { ...design, width: inputs.width, depth: inputs.depth, riverFlow: inputs.flow, slope: inputs.slope, catchment,
    operatingFlow, plantMW, energy: plantMW * hoursPerYear / 1000 };
}

export function reservoirCase({ area, rain, coefficient, meanhead: meanHead, designhead: designHead, eco }) {
  const annual = area * 1e6 * rain / 1000 * coefficient;
  const mean = annual / secondsPerYear;
  const designFlow = 2 * mean;
  const spill = .1 * mean;
  const ecological = Math.min(eco, mean - spill);
  const usable = mean - spill - ecological;
  const efficiency = .91 * .95 * .99;
  const designShaft = 9.81 * designFlow * designHead * .91;
  const designGenerator = designShaft * .95;
  const designTransformer = designGenerator * .99;
  const hours = hoursPerYear * usable * meanHead / (designFlow * designHead);
  const operatingMW = 9.81 * usable * meanHead * efficiency / 1000;
  const generatorEnergy = designGenerator * hours / 1e6;
  const transformerEnergy = operatingMW * hoursPerYear / 1000;
  const gridEnergy = transformerEnergy * .985;
  const ecologicalPower = 9.81 * ecological * designHead * efficiency;
  const ecologicalEnergy = 9.81 * ecological * meanHead * efficiency * hoursPerYear / 1e6;
  return { annual, mean, spill, ecological, shortfall: eco - ecological, usable, designFlow, meanHead, designHead,
    designShaft, designGenerator, designTransformer, hours, operatingMW, ratingMW: designTransformer / 1000,
    generatorEnergy, transformerEnergy, gridEnergy, ecologicalPower, ecologicalEnergy,
    energy: gridEnergy, totalEnergy: gridEnergy + ecologicalEnergy * .985 };
}
