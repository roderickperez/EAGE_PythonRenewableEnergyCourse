// Capture financial state at the same instant as the physical snapshot.
export function fieldRecord(snapshot, budget, event = 'sample') {
 const finance = budget.state(snapshot.revenue, snapshot.capacity);
 return {...snapshot, event, investment: finance.investment, balance: finance.balance,
         upgrades: finance.upgrades, extraWells: finance.extraWells};
}

// One row is one ordered snapshot or configuration event. Producer details allow
// independent replay of the separately clipped thermal contributions.
export function fieldCsv(history) {
 const names = ['record_index','event','years','balanced_depth_count','active_depth_count','rate_kg_s_per_active_depth','total_flow_kg_s','production_C','injection_C','thermal_MW','conversion_efficiency','auxiliary_fraction','availability','price_USD_per_MWh','net_power_MW','capacity_MW','energy_MWh','revenue_USD','cumulative_investment_USD','project_balance_USD','capacity_upgrades','additional_wells_drilled','producer_streams_json','wells_json'];
 const quote = value => '"' + JSON.stringify(value).replaceAll('"', '""') + '"';
 const rows = history.map((p, i) => [i,p.event,p.years,p.activeLayers,p.rate>0?p.activeLayers:0,p.rate,p.activeLayers*p.rate,p.production??'',p.injection,p.thermal,p.efficiency,p.auxiliary,p.availability,p.price,p.net,p.capacity,p.energy,p.revenue,p.investment,p.balance,p.upgrades,p.extraWells,quote(p.wellTemperatures),quote(p.wells)].join(','));
 return [names.join(','),...rows].join('\n');
}
