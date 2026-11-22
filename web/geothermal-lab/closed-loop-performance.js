import { benchmark, outletAt, outletReadings, scenarios, levelizedCost } from './closed-loop-performance-model.js';

const heat = '#c45a3c', water = '#257d9d', carbon = '#94608b', green = '#267762';
const fmt = (value, digits=1) => value.toLocaleString('en-US',{maximumFractionDigits:digits});
const metrics = [
  ['electric','Electrical output','MW electric'], ['thermal','Thermal output','MW thermal'],
  ['outlet','Outlet temperature','°C'], ['lcoe','Levelized electricity cost','USD/MWh'],
  ['flow','Selected mass flow','kg/s'], ['length','Horizontal extent','km'],
];
const costInputs = [
  ['capital','Upfront capital at year 0',0,100,1,30,'million USD'],
  ['operating','Annual operating cost',0,8,.1,1,'million USD/year'],
  ['generation','First-year electricity',1,75,1,30,'GWh/year'],
  ['decline','Annual electricity decline',0,10,.1,0,'%'],
  ['discount','Annual discount rate',0,15,.1,7,'%'],
  ['years','Project lifetime',1,50,1,40,'yr'],
];

function range([id,label,min,max,step,value,unit]) {
  return `<label for="perf-${id}">${label}<output for="perf-${id}"></output>
    <input type="range" id="perf-${id}" data-input="${id}" min="${min}" max="${max}" step="${step}" value="${value}" data-unit="${unit}">
  </label>`;
}

function frame(label, content) {
  return `<svg viewBox="0 0 560 286" role="img" aria-label="${label}" font-family="system-ui, sans-serif">
    <g font-size="13" fill="#315462">${content}</g></svg>`;
}

function timeline({ series, years, yMax, yMin=0, unit, cursor, label, xLabel }) {
  const x = year => 65+year/years*462;
  const y = value => 227-(value-yMin)/(yMax-yMin)*192;
  const xTicks = years < 5 ? Array.from({length:years+1},(_,i)=>i)
    : [...new Set([0,Math.round(years/4),Math.round(years/2),Math.round(years*.75),years])];
  const yTicks = Array.from({length:5},(_,index)=>yMin+(yMax-yMin)*index/4);
  const grid = yTicks.map(value => `<path d="M65 ${y(value)}H527" stroke="#dce8e8"/>
    <text x="57" y="${y(value)+4}" text-anchor="end">${fmt(value)}</text>`).join('')+
    xTicks.map(year=>`<text x="${x(year)}" y="246" text-anchor="middle">${year}</text>`).join('');
  return frame(label,`
    ${grid}
    <text x="65" y="20">${unit}</text>
    <text x="295" y="271" text-anchor="middle">${xLabel}</text>
    <path d="M65 35V227H527" fill="none" stroke="#96abad"/>
    ${cursor ? `<path d="M${x(cursor.year)} 35V227" stroke="#547781" stroke-dasharray="4 5"/>` : ''}
    ${series.map(({ points,color },index) => {
      const path = points.map(([year,value],i)=>`${i?'L':'M'}${x(year)} ${y(value)}`).join(' ');
      const dot = cursor ? [cursor.year,cursor.values[index]] : points.at(-1);
      return `<path d="${path}" fill="none" stroke="${color}" stroke-width="3"/>
        <circle cx="${x(dot[0])}" cy="${y(dot[1])}" r="5" fill="${color}" stroke="white" stroke-width="1.5"/>`;
    }).join('')}`);
}

function scenarioBars(metric) {
  const [key,heading,unit] = metric;
  const maxValue = Math.max(...scenarios.map(row=>row[key]));
  const step = maxValue > 100 ? 50 : maxValue > 10 ? 25 : maxValue > 5 ? 5 : 1;
  const maximum = Math.ceil(maxValue/step)*step;
  const x = value => 149+value/maximum*360;
  const grid = [0,maximum/2,maximum].map(value=>`<path d="M${x(value)} 25V239" stroke="#dce8e8"/>
    <text x="${x(value)}" y="260" text-anchor="middle">${fmt(value)} ${unit}</text>`).join('');
  const bars = scenarios.map((row,i) => {
    const y=33+i*53;
    return `<text x="13" y="${y+12}" font-weight="650">${row.geometry}</text>
      <text x="13" y="${y+28}" font-size="12">${row.fluid}</text>
      <rect x="149" y="${y}" width="${x(row[key])-149}" height="31" rx="4" fill="${i<2?water:carbon}"/>
      <text x="${Math.max(x(row[key])-7,190)}" y="${y+20}" text-anchor="end" fill="white" font-weight="650">${fmt(row[key])}</text>`;
  }).join('');
  const label = heading + '. ' + scenarios.map(row=>`${row.geometry}, ${row.fluid}: ${fmt(row[key])} ${unit}`).join('. ');
  return frame(label,grid+bars);
}

export function mountClosedLoopPerformance(root) {
  root.innerHTML = `<section class="loop-card loop-performance">
    <h2>Explore closed-loop performance</h2>
    <div class="loop-performance-tabs" role="group" aria-label="Performance view">
      <button type="button" data-show="outlet" aria-pressed="true">Outlet temperature</button>
      <button type="button" data-show="scenarios" aria-pressed="false">Reported scenarios</button>
      <button type="button" data-show="lcoe" aria-pressed="false">LCOE calculator</button>
    </div>
    <div data-view="outlet">
      <div class="loop-equation-grid">
        <div class="loop-bars" data-chart="outlet"></div>
        <section class="loop-equation-panel" aria-label="Outlet temperatures and time">
          <dl class="loop-metrics">
            ${benchmark.map(({label,color},i)=>`<div class="loop-case"><dt style="--series:${color}">${label}</dt><dd data-outlet="${i}"></dd></div>`).join('')}
          </dl>
          <div class="loop-controls">${range(['time','Years of operation',.25,40,.25,10,'yr'])}</div>
        </section>
      </div>
      <p class="loop-note">Approximate readings from the overlapping model curves in White et al., Figure 3. Both cases use a 10 km U-shaped water loop. Flow rate and insulation length differ together. The chart begins at 0.25 years to leave out the rapid startup transient.</p>
    </div>
    <div data-view="scenarios" hidden>
      <div class="loop-equation-grid">
        <div class="loop-bars" data-chart="scenarios"></div>
        <section class="loop-equation-panel" aria-label="Reported scenario comparison">
          <div class="loop-option loop-metric-select"><label for="perf-metric">Compare</label>
            <select id="perf-metric">${metrics.map(([key,label])=>`<option value="${key}">${label}</option>`).join('')}</select>
          </div>
          <p data-scenario-insight aria-live="polite"></p>
          <small>40 years · 5 km vertical depth · 7% discount rate · assumed drilling cost of $1,000/m</small>
        </section>
      </div>
      <p class="loop-note">White et al., Table 3. These four modeled designs use individually selected settings and extreme resource assumptions. The U-shaped CO₂ case reaches the 100 kg/s database limit, so its flow is not a proven unconstrained optimum. The 20 km extent reaches the study limit. Costs exclude exploration, redrilling, well-field maintenance and land leasing. Scenario differences reflect complete designs and surface cycles; they do not establish a universal winner.</p>
    </div>
    <div data-view="lcoe" hidden>
      <div class="loop-equation-grid">
        <div class="loop-bars" data-chart="lcoe"></div>
        <section class="loop-equation-panel" aria-label="Discounted results and assumptions">
          <dl class="loop-metrics">
            <div class="loop-case"><dt>Levelized electricity cost</dt><dd data-cost="rate"></dd></div>
            <div><dt>Present value of costs</dt><dd data-cost="cost"></dd></div>
            <div><dt>Discounted electricity</dt><dd data-cost="energy"></dd></div>
          </dl>
          <div class="loop-controls">${costInputs.map(range).join('')}</div>
        </section>
      </div>
      <p class="loop-note">Teaching calculation with capital paid upfront, constant annual operating cost, and electricity delivered at the end of each year. Each point divides cumulative discounted costs by cumulative discounted electricity through that year. Values use the slider assumptions.</p>
    </div>
    <small class="loop-performance-source"><a href="https://doi.org/10.1016/j.geothermics.2023.102852" target="_blank" rel="noopener">White et al. (2024), Figure 3 and Table 3</a></small>
  </section>`;
  const find = selector => root.querySelector(selector);
  const read = id => find(`[data-input="${id}"]`).valueAsNumber;
  const sliders = [...root.querySelectorAll('input[type="range"]')];
  function updateSliderLabel(input) {
    const value = `${fmt(input.valueAsNumber,2)} ${input.dataset.unit}`;
    find(`output[for="${input.id}"]`).textContent = value;
    input.setAttribute('aria-valuetext',value);
  }

  function updateOutlet() {
    const year = read('time'), values = outletAt(year);
    values.forEach((value,i) => find(`[data-outlet="${i}"]`).textContent = `≈ ${fmt(value,0)} °C`);
    find('[data-chart="outlet"]').innerHTML = timeline({
      series:benchmark.map(({color},i)=>({color,points:outletReadings.map(row=>[row[0],row[i+1]])})),
      years:40, yMin:50, yMax:170, unit:'Outlet temperature (°C)', cursor:{year,values},
      label:`Approximate outlet temperatures at year ${year}, ${fmt(values[0],0)} and ${fmt(values[1],0)} degrees Celsius`,
      xLabel:'Years of operation',
    });
  }

  function updateScenarios() {
    const metric = metrics.find(([key])=>key===find('#perf-metric').value);
    const [key, heading, unit] = metric;
    const values = scenarios.map(row=>row[key]);
    const best = key === 'lcoe' ? Math.min(...values) : Math.max(...values);
    const selected = scenarios.filter(row=>row[key]===best).map(row=>row.geometry+' with '+row.fluid);
    find('[data-chart="scenarios"]').innerHTML = scenarioBars(metric);
    const lead = key === 'lcoe' ? 'Lowest reported' : 'Highest reported';
    find('[data-scenario-insight]').textContent = lead+' '+heading.toLowerCase()+' is '+fmt(best)+' '+unit+' ('+selected.join(', ')+').';
  }

  function updateCost() {
    const inputs = Object.fromEntries(costInputs.map(([id])=>[id,read(id)]));
    const series = levelizedCost(inputs);
    const latest = series.at(-1);
    const max = Math.max(10,Math.ceil(Math.max(...series.map(row=>row.lcoe))/20)*20);
    find('[data-cost="rate"]').textContent = `$${fmt(latest.lcoe,2)}/MWh`;
    find('[data-cost="cost"]').textContent = `$${fmt(latest.cost/1e6,2)} million`;
    find('[data-cost="energy"]').textContent = `${fmt(latest.energy/1000,1)} GWh`;
    find('[data-chart="lcoe"]').innerHTML = timeline({
      series:[{color:green,points:series.map(row=>[row.year,row.lcoe])}],
      years:inputs.years,yMax:max,unit:'Cumulative LCOE (USD/MWh)',
      label:`Cumulative levelized electricity cost through year ${inputs.years}: ${fmt(latest.lcoe,2)} dollars per megawatt-hour`,
      xLabel:'Years included in the calculation',
    });
  }

  for (const input of sliders) {
    updateSliderLabel(input);
    input.addEventListener('input',()=>{
      updateSliderLabel(input);
      if (input.dataset.input==='time') updateOutlet();
      else updateCost();
    });
  }
  find('#perf-metric').addEventListener('change',updateScenarios);
  const buttons = [...root.querySelectorAll('[data-show]')];
  buttons.forEach(button=>button.addEventListener('click',()=>{
    buttons.forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
    root.querySelectorAll('[data-view]').forEach(panel=>panel.hidden=panel.dataset.view!==button.dataset.show);
  }));
  updateOutlet();
  updateScenarios();
  updateCost();
}
