import { areas, conduction, energy } from './closed-loop-model.js';

const fmt = (n, digits = 2) => n.toLocaleString('en-US', { maximumFractionDigits: digits });

function comparison(title, entries, unit) {
  const maximum = Math.max(...entries.map(([, value])=>Math.abs(value)), .01);
  const extent = maximum * 1.15;
  const negative = entries.some(([,value])=>value < 0);
  const x = value => 150 + (value + (negative ? extent : 0)) * 260 / (extent * (negative ? 2 : 1));
  const origin = x(0);
  return `<figure class="loop-bars"><figcaption>${title}</figcaption>
    <svg viewBox="0 0 470 130" role="img" aria-label="${title}: ${entries.map(([label,value])=>`${label} ${fmt(value,3)} ${unit}`).join(', ')}">
      <path d="M${origin} 12V102" stroke="#829fa8"/>
      ${entries.map(([label,value,color],i)=>{
        const y=19+i*43, end=x(value);
        return `<text x="8" y="${y+16}">${label}</text><rect x="${Math.min(end,origin)}" y="${y}" width="${Math.abs(end-origin)}" height="24" rx="3" fill="${color}"/>`;
      }).join('')}
      <text x="${origin}" y="119" text-anchor="middle">0</text>
      ${negative ? `<text x="150" y="119" text-anchor="middle">${fmt(-extent)} ${unit}</text>` : ''}
      <text x="410" y="119" text-anchor="middle">${fmt(extent)} ${unit}</text>
    </svg></figure>`;
}

function accumulation(result, years) {
  const horizon = 40;
  const durationHours = horizon*8760;
  const scale = Math.max(result.thermal, Math.abs(result.available), .01)*durationHours*1.12;
  const min = result.available < 0 ? -scale : 0;
  const x = year => 62 + year/horizon*378;
  const y = energy => 104 - (energy-min)/(scale-min)*83;
  const origin = y(0);
  const point = (energy,color) => `<path d="M62 ${origin}L${x(years)} ${y(energy)}" fill="none" stroke="${color}" stroke-width="3"/>
     <circle cx="${x(years)}" cy="${y(energy)}" r="4" fill="${color}"/>`;
  return `<figure class="loop-bars"><figcaption>Accumulated over time</figcaption>
    <svg viewBox="0 0 470 149" role="img" aria-label="After ${years} years, ${fmt(result.thermalMWh,0)} MWh thermal and ${fmt(result.availableMWh,0)} MWh available work">
      <path d="M62 21V104H440M62 ${origin}H440" fill="none" stroke="#829fa8"/>
      <text x="55" y="25" text-anchor="end">${fmt(scale/1000,1)}</text>
      <text x="55" y="${origin+4}" text-anchor="end">0</text>
      ${min < 0 ? `<text x="55" y="108" text-anchor="end">${fmt(min/1000,1)}</text>` : ''}
      <text x="10" y="13">GWh</text>
      ${[0,10,20,30,40].map(year=>`<text x="${x(year)}" y="123" text-anchor="middle">${year} yr</text>`).join('')}
      ${point(result.thermalMWh,'#ca6640')}${point(result.availableMWh,'#25819b')}
      <text x="62" y="143" fill="#a54c2e">● Thermal</text><text x="153" y="143" fill="#1f6e84">● Available work</text>
    </svg></figure>`;
}

function conductionChart(points, input, temperature) {
  const x = distance => 55 + distance/10*500;
  const y = value => 180 - (value-input.wall)/(input.far-input.wall)*155;
  const ticks = [0,2.5,5,7.5,10].map(distance =>
    `<path d="M${x(distance)} 25V180" stroke="#e5eff0"/>
     <text x="${x(distance)}" y="199" text-anchor="middle">${distance}</text>`).join('');
  const trace = points.map(([distance,value],index) =>
    `${index ? 'L' : 'M'}${x(distance).toFixed(1)} ${y(value).toFixed(1)}`).join(' ');
  return `<svg viewBox="0 0 600 235" role="img" aria-label="Rock temperature versus distance from the wall">
    <g font-size="12" fill="#526b77" font-family="system-ui">
      ${ticks}
      <path d="M55 25V180H555" fill="none" stroke="#9db5bb"/>
      <text x="3" y="32">${input.far}</text>
      <text x="3" y="182">${input.wall}</text>
      <text x="300" y="224" text-anchor="middle">Distance from wall (m)</text>
      <text x="60" y="16">Rock temperature (°C)</text>
      <path d="${trace}" fill="none" stroke="#16879d" stroke-width="3"/>
      <circle cx="${x(input.distance)}" cy="${y(temperature)}" r="5" fill="#c44f3e"/>
    </g>
  </svg>`;
}

export function mountClosedLoopEquation(root, mode, spec, mountCutaway) {
  const isArea = mode === 'area';
  const metrics = {
    area: ['Inner tube area', 'Annular area', 'Annulus / tube'],
    conduction: ['Thermal diffusivity α', 'Thermal effusivity', 'Rock temperature at selected distance'],
    energy: ['Heat rate', 'Available work rate', 'Thermal energy', 'Available work'],
  }[mode];
  root.innerHTML = `<section class="loop-card">
    <h2>${spec.title}</h2><p>${spec.intro}</p>
    <div class="loop-equation-grid" data-equation="${mode}">
      <div class="loop-equation-visual">${isArea ? `<div class="loop-pipe-canvas" role="img" aria-label="Three-dimensional cutaway of a concentric geothermal pipe"></div>
        <div class="loop-pipe-key"><span>Blue · annulus</span><span>Gray · pipe wall</span><span>Orange · inner tube</span></div>
        <small>Drag to orbit and scroll to zoom. All three diameters use the same scale.</small>` : ''}</div>
      <section class="loop-equation-panel" aria-label="Inputs and results">
        <dl class="loop-metrics">${metrics.map((metric,i)=>`<div><dt>${metric}</dt><dd data-metric="${i}"></dd></div>`).join('')}</dl>
        <div class="loop-controls">${spec.controls.map(([id,label,min,max,step,value,unit])=>`<label for="loop-${id}">${label}<output for="loop-${id}" data-out="${id}"></output><input id="loop-${id}" type="range" data-id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${label} (${unit})"></label>`).join('')}</div>
      </section>
    </div>
    <p class="loop-note"></p>
  </section>`;

  const visual = root.querySelector('.loop-equation-visual');
  const note = root.querySelector('.loop-note');
  const inputs = [...root.querySelectorAll('input')];
  const outputs = [...root.querySelectorAll('[data-metric]')];
  let pipe;
  if (isArea) {
    const host = root.querySelector('.loop-pipe-canvas');
    try {
      pipe = mountCutaway(host);
      window.addEventListener('pagehide', ()=>pipe.destroy(), {once:true});
    } catch {
      host.textContent = '3D view unavailable. The cross-section and area calculations remain available.';
    }
  }

  function update() {
    const data = Object.fromEntries(inputs.map(el=>[el.dataset.id,el.valueAsNumber]));
    for (const [id,,,,,,unit] of spec.controls) {
      const value = `${fmt(data[id],4)} ${unit}`;
      root.querySelector(`[data-out="${id}"]`).textContent = value;
      root.querySelector(`[data-id="${id}"]`).setAttribute('aria-valuetext',value);
    }
    let values;
    if (isArea) {
      const [inner,outer,bore] = inputs;
      const bound = (value, delta) => (value + delta).toFixed(3);
      inner.max = Math.min(.14,bound(data.outer,-.005));
      outer.min = Math.max(.06,bound(data.inner,.005));
      outer.max = Math.min(.2,bound(data.bore,-.005));
      bore.min = Math.max(.12,bound(data.outer,.005));
      const result = areas(data);
      pipe?.update(data);
      if (!pipe) {
        const host = visual.querySelector('.loop-pipe-canvas');
        const circles = [['bore','#72bfd7'],['outer','#87989e'],['inner','#d27150']];
        host.innerHTML = `<svg viewBox="0 0 320 270" role="img" aria-label="Coaxial cross-section fallback">${circles.map(([key,color])=>`<circle cx="160" cy="135" r="${400*data[key]}" fill="${color}"/>`).join('')}</svg>`;
      }
      values = [`${result.tube.toFixed(4)} m²`, `${result.annulus.toFixed(4)} m²`, fmt(result.ratio,2)];
      note.textContent = 'The sliders maintain a positive inner-pipe wall thickness and flow area in both passages. Pressure loss also depends on flow and fluid properties.';
    } else if (mode === 'conduction') {
      const result = conduction(data);
      const points = Array.from({length:81}, (_,index) => {
        const distance = index/8;
        return [distance,conduction({...data,distance}).temperature];
      });
      visual.innerHTML = `<div class="loop-plot">${conductionChart(points,data,result.temperature)}</div>`;
      values = [
        `${result.alpha.toExponential(2)} m²/s`,
        `${fmt(result.effusivity,0)} J/(m² K √s)`,
        `${fmt(result.temperature,1)} °C at ${fmt(data.distance,1)} m`,
      ];
      note.textContent = 'The red point marks the selected distance on the one-dimensional profile for a fixed wall temperature.';
    } else {
      const result = energy(data);
      const rows = (thermal, available) => [['Thermal',thermal,'#ca6640'],['Available work',available,'#25819b']];
      visual.innerHTML = comparison('Instantaneous rates', rows(result.thermal,result.available), 'MW') +
        accumulation(result, data.years);
      values = [`${fmt(result.thermal,3)} MW`,`${fmt(result.available,3)} MW`,
        `${fmt(result.thermalMWh,0)} MWh`,`${fmt(result.availableMWh,0)} MWh`];
      note.textContent = 'The duration changes accumulated energy while the rates stay constant. Available work is signed. The enthalpy and entropy differences are teaching inputs and must come from consistent fluid states for a real calculation.';
    }
    outputs.forEach((el,i)=>el.textContent=values[i]);
  }
  inputs.forEach(input=>input.addEventListener('input',update));
  update();
}
