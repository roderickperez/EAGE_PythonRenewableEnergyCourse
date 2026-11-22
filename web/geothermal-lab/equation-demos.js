// Each example follows the stated lesson boundary and uses explicit SI conversions.
const c = (id, label, min, max, step, initial, unit) => [id, label, min, max, step, initial, unit];
const fmt = (x, digits = 3) => Number(x.toFixed(digits)).toLocaleString('en', { maximumFractionDigits: digits });
const flow = c('flow', 'Mass flow ṁ', 0, 150, 1, 50, 'kg/s');
const hot = c('reservoir', 'Production temperature Tₚ', 90, 220, 1, 150, '°C');
const cold = c('reinjection', 'Return temperature Tᵣ', 15, 80, 1, 70, '°C');
const cp = c('cp', 'Specific heat cₚ', 3000, 4600, 10, 4180, 'J/(kg K)');
const thermal = a => a.flow * a.cp * (a.reservoir - a.reinjection) / 1e6;
const out = (headline, equation, scene = {}, metric = 0) => ({ headline, equation, scene, metric });
export const demos = {
  darcy: {
    title: 'Darcy flow through reservoir rock', note: 'Single-phase horizontal Darcy flux; the animation illustrates relative flow, not pore-scale velocities.',
    controls: [c('permeability','Permeability',1,300,1,50,'mD'), c('viscosity','Dynamic viscosity',.1,2,.1,.5,'mPa s'),c('pressureGradient','Pressure-gradient magnitude',0,30,1,10,'kPa/m')],
    calculate(a) { const q = a.permeability*9.869233e-16/(a.viscosity/1000)*a.pressureGradient*1000;
      return out(`|qᴅ| = ${q.toExponential(2)} m/s`, `|qᴅ| = (k/μ)|dp/dx| = ${q.toExponential(2)} m/s. Flow is toward lower pressure.`, {flow: Math.min(150,q*1e7)}, q); }
  },
  gradient: {
    title: 'Temperature gradient and conductive heat flux', note: 'A local linear profile and one-dimensional conductive heat transfer; the rock colors are illustrative.',
    controls: [c('surface','Surface temperature',0,35,1,15,'°C'),c('depth','Depth',.5,6,.1,3,'km'),c('gradient','Geothermal gradient',10,75,1,35,'K/km'),c('conductivity','Thermal conductivity',.5,5,.1,2.5,'W/(m K)')],
    calculate(a) {const t=a.surface+a.depth*a.gradient, q=a.conductivity*a.gradient/1000;
      return out(`T(z) = ${fmt(t,1)} °C`, `T = ${a.surface} + ${a.gradient} × ${a.depth} = ${fmt(t,1)} °C · Upward |q| = ${fmt(q,4)} W/m²`, {reservoir:Math.min(220,Math.max(65,t))}, t);}
  },
  heat: {
    title: 'Heat extracted from produced fluid', note: 'Single-phase, constant-specific-heat approximation; a steam or flashing mixture requires actual enthalpies.',
    controls:[flow,hot,cold,cp],
    calculate(a) {const p=thermal(a), dh=a.cp*(a.reservoir-a.reinjection)/1000;
      return out(`Pₜₕ = ${fmt(p)} MW`, `Δh ≈ cₚΔT = ${fmt(dh,1)} kJ/kg · ṁΔh = ${fmt(p)} MW thermal`,a,p);}
  },
  carnot: {
    title:'Carnot heat-engine limit', note:'The ideal fixed-reservoir limit uses kelvin and is not a plant efficiency.',
    controls:[hot,c('sink','Cold reservoir temperature T꜀',5,45,1,25,'°C')],
    calculate(a) {const eta=1-(a.sink+273.15)/(a.reservoir+273.15);return out(`ηC = ${fmt(eta*100,1)}%`, `1 − (${fmt(a.sink+273.15,2)} K / ${fmt(a.reservoir+273.15,2)} K) = ${fmt(eta*100,1)}%`,a,eta);}
  },
  conversion: {
    title:'Thermal, gross and net power', note:'Teaching model with a specified conversion efficiency and parasitic load as a share of gross electricity. The fixed-reservoir Carnot bound is an upper limit; cooling a finite fluid stream generally imposes a stricter limit.' ,
    controls:[flow,hot,cold,c('sink','Heat-sink temperature',5,45,1,25,'°C'),c('efficiency','Conversion efficiency',0,20,.5,12,'%'),c('parasitic','Parasitic share of gross',0,50,1,10,'%')],
    calculate(a) {const p=a.flow*4180*(a.reservoir-a.reinjection)/1e6, gross=p*a.efficiency/100, load=gross*a.parasitic/100;
      const limit=1-(a.sink+273.15)/(a.reservoir+273.15);
      const warning=a.efficiency/100>limit ? ' Warning: efficiency exceeds this ideal Carnot limit; positive output at this efficiency would be physically infeasible for the specified sink.' : ' Passing this upper bound alone does not establish feasibility.';
      const returnWarning=a.reinjection<=a.sink ? ' Return temperature is at or below the specified sink: passive heat exchange with that sink cannot cool the fluid to this return temperature; a colder sink or additional work would be needed. The displayed net power does not account for such work.' : '';
      return out(`Pₙₑₜ = ${fmt(gross-load)} MW`, `Pₜₕ = ${fmt(p)} MW · Pgross = ηPₜₕ = ${fmt(gross)} MW · Pnet = ${fmt(gross)} − ${fmt(load)} = ${fmt(gross-load)} MW · Carnot upper bound = ${fmt(limit*100,1)}%.${warning}${returnWarning}`,a,gross-load);}
  },
  pump: {
    title:'Hydraulic pumping power', note:'Constant liquid density; combined motor–drive–pump efficiency converts hydraulic power to electrical input. Volume flow is mass flow divided by density.',
    controls:[flow,c('density','Fluid density',800,1200,10,1000,'kg/m³'),c('pressure','Pressure rise',0,15,.1,2,'MPa'),c('pumpEfficiency','Combined motor–pump efficiency',30,95,1,75,'%')],
    calculate(a) {const q=a.flow/a.density,p=a.pressure*q/(a.pumpEfficiency/100);return out(`Ppump = ${fmt(p)} MW`, `Q = ṁ/ρ = ${fmt(q,4)} m³/s · ΔpQ/ηpump = ${fmt(p)} MW`,{...a,loadMode:'explicit'},p);}
  },
  heatpump: {
    title:'Direct heat and heat pump', note:'This heating example moves heat from a source; it is not electrical power generation.',
    controls:[c('heatPumpPower','Electrical input',0,5,.1,.5,'MW'),c('cop','Heating coefficient of performance',1,6,.1,3.5,'')],
    calculate(a) {const q=a.cop*a.heatPumpPower;return out(`Heat delivered = ${fmt(q)} MW`, `Q̇delivered = COPₕ × Pelectric = ${fmt(q)} MW thermal · Source heat = ${fmt(q-a.heatPumpPower)} MW`,{flow:Math.min(150,q*30)},q);}
  },
  storage: {
    title:'Sensible heat stored in the reservoir', note:'Uniform bulk volume and properties; all pore volume is assumed water-filled, with fixed water density 1,000 kg/m³ and heat capacity 4,180 J/(kg K). This is heat in place, not recoverable electricity.',
    controls:[c('volume','Bulk reservoir volume',.5,30,.5,5,'million m³'),c('porosity','Porosity',0,35,1,12,'%'),hot,c('surface','Reference temperature',0,35,1,15,'°C'),c('rockDensity','Solid-rock density',2000,3200,10,2600,'kg/m³'),c('rockCp','Rock heat capacity',600,1400,10,900,'J/(kg K)')],
    calculate(a) {const phi=a.porosity/100, e=a.volume*1e6*((1-phi)*a.rockDensity*a.rockCp+phi*1000*4180)*(a.reservoir-a.surface)/3.6e12;
      return out(`Heat in place = ${fmt(e,1)} GWh`, `V[(1−φ)ρᵣcᵣ + φρꜰcꜰ](T−Tref) = ${fmt(e,1)} GWh thermal`,a,e);}
  },
  decline: {
    title:'Compounded annual power decline', note:'A prescribed scenario, not a reservoir prediction. The exponent is zero in year 1.',
    controls:[c('initial','Initial net power P₁',0,20,.1,5,'MW'),c('decline','Annual decline',0,20,.1,2,'%'),c('year','Operating year',1,40,1,10,'')],
    calculate(a) {const p=a.initial*(1-a.decline/100)**(a.year-1);return out(`Year ${a.year}: ${fmt(p)} MW`, `Pᵧ = ${a.initial} × (1 − ${a.decline}/100)<sup>${a.year-1}</sup> = ${fmt(p)} MW`,{flow:Math.min(150,p*15)},p);}
  },
  energy: {
    title:'Net power, operating time and energy', note:'Availability is applied once to a constant operating-power scenario; do not apply it again to measured power that includes downtime. A capacity factor above 100% indicates operating power exceeds the specified rating.',
    controls:[c('power','Net operating power',0,20,.1,5,'MW'),c('hours','Period',1,8760,1,8760,'h'),c('availability','Availability',0,100,1,95,'%'),c('rating','Net nameplate rating',1,20,.1,10,'MW')],
    calculate(a) {const e=a.power*a.hours*a.availability/100, cf=e/(a.rating*a.hours);return out(`E = ${fmt(e,1)} MWh`, `E = P × hours × availability = ${fmt(e,1)} MWh · CF = E/(rating × hours) = ${fmt(cf*100,1)}%`,{flow:Math.min(150,a.power*10)},e);}
  }
};
