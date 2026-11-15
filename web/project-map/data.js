// Each file is small, numeric and unquoted, with an explicit UTC timestamp.
export const schemas = {
  solar: ['irradiance_wm2','ambient_c'],
  wind: ['wind_speed_ms'],
  hydro: ['river_flow_m3s'],
  geothermal: ['mass_flow_kgs','production_c','reinjection_c'],
  demand: ['demand_mw'],
};

export function joinProjectInputs(files) {
  const tables = Object.entries(schemas).map(([name, fields]) => {
    const [header, ...records] = files[name].trim().split(/\r?\n/);
    if (header !== ['timestamp', ...fields].join(',')) throw new Error(`Unexpected columns in ${name}.csv`);
    const table = new Map();
    for (const record of records) {
      const [date, ...values] = record.split(',');
      const time = Date.parse(date.replace(' ', 'T'));
      if (!date.endsWith('+00:00') || !Number.isFinite(time) || table.has(time) ||
          values.length !== fields.length || values.some(value => !value.trim() || !Number.isFinite(Number(value)))) {
        throw new Error(`Invalid hourly record in ${name}.csv`);
      }
      table.set(time,Object.fromEntries(fields.map((field,i) => [field,Number(values[i])])));
    }
    return table;
  });
  const start = Date.UTC(2025,5,1);
  const count = 168;
  if (tables.some(table => table.size !== count)) throw new Error('The project needs 168 records per CSV.');
  return Array.from({length:count},(_,hour)=>{
    const time = start + hour*3600000;
    const matched = tables.map(table => table.get(time));
    if (matched.some(row => !row)) throw new Error('The CSVs must cover the same UTC hours.');
    return { time, ...Object.assign({},...matched) };
  });
}
