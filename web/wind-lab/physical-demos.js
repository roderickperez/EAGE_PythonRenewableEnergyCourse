import { BETZ, aerodynamicConversion, availableWind, dryAirDensity, electricalPower,
  logLawWind, powerLawWind, twoIntervalWind } from './model.js';
import { axesFor, mw, n, operatingMarker, percent, sample, windColors as c } from './demo-plots.js';

const speed = ['speed', 'Undisturbed wind speed v', 0, 16, .1, 8, 'm/s'];
const density = ['density', 'Air density ρ', .8, 1.4, .005, 1.225, 'kg/m³'];
const diameter = ['diameter', 'Rotor diameter D', 40, 140, 1, 80, 'm'];
const pressure = ['pressure', 'Air pressure', 80, 105, .005, 101.325, 'kPa'];
const temperature = ['temperature', 'Air temperature', -30, 50, 1, 15, '°C'];
const powerInputs = input => availableWind(input.speed, input.density, input.diameter / 2);
export const referenceTurbine = { cutIn: 3, ratedSpeed: 12, cutOut: 25, ratedMW: 3 };

export const physicalDemos = {
  'cube-law': {
    title: 'Why wind power contains v³',
    plot: 'Available wind power versus wind speed',
    caption: 'The orange dot marks the selected speed. The second dot uses twice that speed with the same area and density.',
    note: 'This calculation is the undisturbed kinetic-energy flux across the swept area. The turbine rotation is illustrative.',
    legend: [[c.wind, 'Available wind power'], [c.electrical, 'Selected wind speed']],
    controls: [speed, density, diameter],
    draw(svg, input) {
      const { speed, density, diameter } = input;
      const result = powerInputs(input);
      const atSpeed = v => availableWind(v, density, diameter / 2).windMW;
      const maxSpeed = Math.max(16, 2 * speed);
      const axes = axesFor(svg, 0, maxSpeed, atSpeed(maxSpeed) * 1.12, 'Wind speed (m/s)', 'Available power (MW)');
      axes.series(sample(0, maxSpeed, atSpeed), c.wind);
      operatingMarker(axes, speed, result.windMW);
      if (speed > 0) axes.dot(2 * speed, atSpeed(2 * speed), c.wind);
      return {
        headline: `P<sub>wind</sub> = ${mw(result.windMW)}`,
        rows: [
          `A = πD²/4 ≈ ${n(result.area)} m² · ṁ = ρAv ≈ ${n(result.massFlow)} kg/s`,
          `Specific kinetic energy v²/2 = ${n(result.specificEnergy)} J/kg · ṁ × v²/2 ≈ <strong>${mw(result.windMW)}</strong>`,
          `At ${n(2 * speed)} m/s, available power is ${mw(atSpeed(2 * speed))}. ${speed > 0 ? 'Doubling the selected speed multiplies power by eight.' : 'Both speeds are zero.'}`,
        ],
        speed, radius: diameter / 2,
      };
    },
  },
  betz: {
    title: 'From available wind power to electricity',
    plot: 'Rotor and electrical power versus power coefficient',
    caption: 'The vertical dashed line marks the ideal Betz bound. The separation between rotor and electrical output is drivetrain loss.',
    note: 'A fixed coefficient and efficiency isolate the conversion stages. The electrical estimate is before turbine operating limits.',
    legend: [[c.wind, 'Available wind power'], [c.rotor, 'Extracted rotor power'], [c.electrical, 'Electrical estimate'], [c.reference, 'Ideal Betz bound']],
    controls: [speed, density, diameter, ['cp', 'Power coefficient C<sub>p</sub>', 0, BETZ, 'any', .44, ''],
      ['efficiency', 'Drivetrain efficiency', 0, 100, 1, 92, '%']],
    draw(svg, input) {
      const { windMW } = powerInputs(input);
      const { cp, efficiency } = input;
      const loss = aerodynamicConversion(windMW, cp, efficiency / 100);
      const axes = axesFor(svg, 0, .64, windMW * 1.13, 'Rotor power coefficient', 'Power (MW)');
      axes.line(axes.x(0), axes.y(windMW), axes.x(.64), axes.y(windMW), c.wind, '6 4');
      axes.line(axes.x(BETZ), axes.y(0), axes.x(BETZ), axes.y(windMW), c.reference, '4 4');
      axes.series([[0, 0], [BETZ, loss.betzMW]], c.rotor);
      axes.series([[0, 0], [BETZ, loss.betzMW * efficiency / 100]], c.electrical);
      axes.dot(cp, loss.rotorMW, c.rotor);
      operatingMarker(axes, cp, loss.electricalMW);
      return {
        headline: `P<sub>e</sub> = ${mw(loss.electricalMW)}`,
        rows: [
          `Available ${mw(windMW)} · Rotor = ${n(cp)} × ${n(windMW)} ≈ <strong>${mw(loss.rotorMW)}</strong>`,
          `Electrical = ${n(efficiency / 100)} × ${n(loss.rotorMW)} ≈ <strong>${mw(loss.electricalMW)}</strong> · Drivetrain loss ${mw(loss.drivetrainLossMW)}`,
          `Ideal Betz bound 16/27 ≈ ${percent(BETZ)} · Maximum ideal rotor extraction at this speed ${mw(loss.betzMW)}`,
        ],
        speed: input.speed, radius: input.diameter / 2, electrical: loss.electricalMW,
      };
    },
  },
  'air-density': {
    title: 'Dry-air density and available wind power',
    plot: 'Air density versus temperature',
    caption: 'Pressure is held at the selected value along the curve. The dot marks the selected temperature.',
    note: 'The dry-air gas constant is 287.05 J/(kg·K). The wind-power calculation uses the selected speed and rotor diameter.',
    legend: [[c.wind, 'Dry-air density'], [c.electrical, 'Selected condition']],
    controls: [pressure, temperature, speed, diameter],
    draw(svg, input) {
      const { pressure, temperature, speed, diameter } = input;
      const rhoAt = t => dryAirDensity(pressure, t);
      const rho = rhoAt(temperature);
      const { windMW } = availableWind(speed, rho, diameter / 2);
      const axes = axesFor(svg, -30, 50, 1.6, 'Air temperature (°C)', 'Density (kg/m³)');
      axes.series(sample(-30, 50, rhoAt), c.wind);
      operatingMarker(axes, temperature, rho);
      return {
        headline: `ρ = ${n(rho)} kg/m³`,
        rows: [
          `ρ = p/(R<sub>d</sub>T) = ${n(pressure * 1000)} Pa / (287.05 × ${n(temperature + 273.15)} K) ≈ <strong>${n(rho)} kg/m³</strong>`,
          `At ${n(speed)} m/s and D = ${n(diameter)} m, available wind power is <strong>${mw(windMW)}</strong>`,
        ],
        speed, radius: diameter / 2,
      };
    },
  },
  shear: {
    title: 'Extrapolating wind speed to hub height',
    plot: 'Power-law and neutral log-law wind profiles',
    caption: 'Both curves pass through the same reference measurement. The paired markers show the two estimates at the selected hub height.',
    note: 'The turbine animation uses the power-law hub speed. The neutral log-law curve is normalized to the same reference speed. The selected heights are above the displacement and roughness lengths.',
    legend: [[c.wind, 'Power law'], [c.rotor, 'Neutral log law (dashed)'], [c.reference, 'Reference measurement']],
    controls: [
      ['referenceSpeed', 'Reference wind speed', 0, 15, .1, 7, 'm/s'],
      ['referenceHeight', 'Reference height', 10, 80, 1, 30, 'm'],
      ['hub', 'Hub height', 20, 160, 1, 90, 'm'],
      ['alpha', 'Shear exponent α', 0, .5, .01, .14, ''],
      ['roughness', 'Roughness length z<sub>0</sub>', .01, .5, .01, .12, 'm'],
      ['displacement', 'Displacement height d', 0, 5, .1, 0, 'm'],
    ],
    draw(svg, input) {
      const { referenceSpeed, referenceHeight, hub, alpha, roughness, displacement } = input;
      const power = z => powerLawWind(referenceSpeed, z, referenceHeight, alpha);
      const log = z => logLawWind(referenceSpeed, z, referenceHeight, roughness, displacement);
      const hubSpeed = power(hub);
      const logSpeed = log(hub);
      const axes = axesFor(svg, 10, 160, Math.max(1, 1.12 * Math.max(power(160), log(160))), 'Height above ground (m)', 'Wind speed (m/s)');
      axes.series(sample(10, 160, power), c.wind);
      axes.series(sample(10, 160, log), c.rotor, '6 4');
      axes.dot(referenceHeight, referenceSpeed, c.reference, 5);
      axes.dot(hub, logSpeed, c.rotor, 4.5);
      operatingMarker(axes, hub, hubSpeed, c.wind);
      return {
        headline: `Hub wind = ${n(hubSpeed)} m/s`,
        rows: [
          `Power law · ${n(referenceSpeed)} × (${hub}/${referenceHeight})<sup>${n(alpha)}</sup> ≈ <strong>${n(hubSpeed)} m/s</strong>`,
          `Neutral log law at the same hub height ≈ <strong>${n(logSpeed)} m/s</strong> · Difference (log − power) ${n(logSpeed - hubSpeed)} m/s`,
        ],
        speed: hubSpeed, hub,
      };
    },
  },
  turbulence: {
    title: 'Wind fluctuations and averaging',
    plot: 'Two equal-duration wind intervals',
    caption: 'The vertical marker advances through the two intervals. The dashed line is the window mean.',
    note: 'The two values are equal-duration interval means. The example uses population standard deviation and compressed playback.',
    legend: [[c.wind, 'Interval wind speed'], [c.reference, 'Window mean (dashed)'], [c.electrical, 'Playback position']],
    controls: [['first', 'First interval speed', 0, 20, .1, 4, 'm/s'],
      ['second', 'Second interval speed', 0, 20, .1, 8, 'm/s'], ['interval', 'Each interval lasts', 1, 30, 1, 5, 'min']],
    draw(svg, { first, second, interval }) {
      const summary = twoIntervalWind(first, second);
      const top = Math.max(1, Math.max(first, second) * 1.18);
      const axes = axesFor(svg, 0, interval * 2, top, 'Elapsed time (min)', 'Wind speed (m/s)');
      axes.series([[0, first], [interval, first], [interval, second], [2 * interval, second]], c.wind);
      axes.line(axes.x(0), axes.y(summary.mean), axes.x(interval * 2), axes.y(summary.mean), c.reference, '5 4');
      const mark = axes.line(axes.x(0), axes.y(0), axes.x(0), axes.y(top), c.electrical, '4 3');
      return {
        headline: `TI = ${summary.intensity === null ? 'undefined' : percent(summary.intensity)}`,
        rows: [
          `Mean wind ${n(summary.mean)} m/s · Population standard deviation ${n(summary.deviation)} m/s · Window ${interval * 2} min`,
          `Mean of cubed speeds = <strong>${n(summary.meanCubedSpeeds)}</strong> · Cube of mean speed = <strong>${n(summary.cubedMeanSpeed)}</strong> (m³/s³)`,
        ],
        atPhase(phase) {
          const x = axes.x(interval * 2 * phase);
          mark.setAttribute('x1', x);
          mark.setAttribute('x2', x);
          return phase < .5 ? first : second;
        },
      };
    },
  },
  'electrical-curve': {
    title: 'Cut-in, rated output and cut-out',
    plot: 'Electrical output versus wind speed',
    caption: 'The curve rises to the rating and stays flat up to, but excluding, cut-out. The closed point at cut-out marks zero output.',
    note: 'This is the lesson’s idealized electrical curve. Slider ranges keep cut-in below rated speed and rated speed below cut-out.',
    legend: [[c.wind, 'Electrical curve'], [c.reference, 'Operating thresholds (dashed)'], [c.electrical, 'Selected operating point']],
    controls: [['speed', 'Wind speed', 0, 32, .1, 10, 'm/s'],
      ['cutIn', 'Cut-in speed', 1, 6, .1, 3, 'm/s'], ['ratedSpeed', 'Rated speed', 8, 16, .1, 12, 'm/s'],
      ['cutOut', 'Cut-out speed', 18, 30, .1, 25, 'm/s'], ['ratedMW', 'Rated electrical power', .5, 10, .1, 3, 'MW']],
    draw(svg, input) {
      const { speed, cutIn, ratedSpeed, cutOut, ratedMW } = input;
      const power = value => electricalPower(value, input);
      const output = power(speed);
      const axes = axesFor(svg, 0, 32, ratedMW * 1.12, 'Wind speed (m/s)', 'Electrical output (MW)');
      for (const x of [cutIn, ratedSpeed, cutOut]) axes.line(axes.x(x), axes.y(0), axes.x(x), axes.y(ratedMW), c.reference, '3 4');
      axes.series([[0, 0], ...sample(cutIn, ratedSpeed, power), [cutOut, ratedMW]], c.wind);
      axes.series([[cutOut, 0], [32, 0]], c.wind);
      const open = axes.dot(cutOut, ratedMW, '#f8fcfa');
      open.setAttribute('stroke', c.wind);
      axes.dot(cutOut, 0, c.wind);
      operatingMarker(axes, speed, output);
      const region = speed <= cutIn ? 'At or below cut-in' : speed >= cutOut ? 'Cut-out shutdown'
        : speed >= ratedSpeed ? 'Rated output' : 'Cubic ramp';
      return {
        headline: `P = ${mw(output)}`,
        rows: [`<strong>${region}</strong> · Selected speed ${n(speed)} m/s`,
          `Cut-in ${n(cutIn)} m/s → rated ${n(ratedSpeed)} m/s → cut-out ${n(cutOut)} m/s · Rating ${mw(ratedMW)}`],
        speed, electrical: output, stopped: output === 0,
      };
    },
  },
};
