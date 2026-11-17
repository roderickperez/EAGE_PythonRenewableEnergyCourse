import { solarPower, solarDay } from './model.js';

export const SITES = [
  { region:'North America', name:'Phoenix, USA', lat:33.448, lon:-112.074 },
  { region:'South America', name:'Santiago, Chile', lat:-33.448, lon:-70.669 },
  { region:'Africa', name:'Nairobi, Kenya', lat:-1.292, lon:36.822 },
  { region:'Europe', name:'Vienna, Austria', lat:48.208, lon:16.374 },
  { region:'Asia', name:'New Delhi, India', lat:28.614, lon:77.209 },
];
export const PANEL_COST = 1000, SALVAGE = 500, PRICE = .10;
export function dayNumber(date) {
  const start = Date.UTC(date.getUTCFullYear(), 0, 0);
  return Math.floor((Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()) - start) / 86400000);
}
export function declination(date) {
  return 23.44 * Math.sin(2 * Math.PI * (284 + dayNumber(date)) / 365.2422);
}
export function conditions(date, lat, variation = 0) {
  const season = Math.cos(2 * Math.PI * (dayNumber(date) - 172) / 365.2422) * Math.sin(lat * Math.PI / 180);
  const clouds = Math.max(.08, Math.min(.82, .34 - .12 * season + variation));
  return { clouds, ambient: 18 + 18 * season - .18 * Math.abs(lat), dni: 950 * (1 - .75 * clouds), dhi: 120 + 130 * clouds };
}
// A slowly varying illustration of dust and other surface losses; separate from cloud attenuation.
export function surfaceLoss(variation = 0) {
  return Math.max(2, Math.min(9, 5 + variation * 18));
}
export function inputs(site, date, settings, weather) {
  return { latitude:site.lat, declination:declination(date), tilt:settings.tilt, azimuth:settings.azimuth,
    area:2 * settings.count, efficiency:20, dni:weather.dni, dhi:weather.dhi, albedo:.2,
    ambient:weather.ambient, noct:45, coefficient:-.4, soiling:settings.shading,
    wiring:3, inverterEfficiency:96, rating:settings.count * 400 };
}
export function calculate(site, date, hour, settings, weather) {
  const input=inputs(site,date,settings,weather);
  // solarDay divides by rating; avoid undefined zero-panel metrics.
  return { current:solarPower(input,hour), day:settings.count ? solarDay(input) : { samples:[],energy:0 } };
}
