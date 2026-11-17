// The sun is below the horizon at up <= 0; blend through low daylight smoothly.
export const daylightFactor=up=>Math.max(0,Math.min(1,up/.08));
