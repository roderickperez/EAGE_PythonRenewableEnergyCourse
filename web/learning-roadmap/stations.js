export const stations = [
  {
    id: 'python', name: 'Python Island', x: 0, z: 8, color: '#3676a5', quiz: 'generalPython',
    description: 'In the Python Course, run a notebook, store measurements in variables and lists, and use loops and functions to repeat a calculation. NumPy, pandas and Matplotlib take you from individual readings to arrays, tables and plots.',
    task: 'Calculate energy from a short power series before crossing to the energy hub.',
    question: 'In Python, what do 7 / 2 and 7 // 2 return?', options: ['3 and 3', '3.5 and 3', '3.5 and 4'], correct: 1,
    why: 'True division gives 3.5. Floor division rounds the quotient down to 3.'
  },
  {
    id: 'hydro', name: 'Hydropower', x: -8, z: 3, color: '#237fb6', quiz: 'hydroelectricEnergy',
    description: 'Trace water from the reservoir to the turbine. Use discharge, net head and efficiency to estimate electrical power, then integrate power over time to get energy.',
    task: 'Change the flow in the plant lab and watch the turbine and power readout.',
    question: 'Which pair gives the units of electrical power and delivered energy, in that order?', options: ['MWh and MW', 'MW and MW', 'MW and MWh'], correct: 2,
    why: 'A plant operating at 1 MW for 1 hour delivers 1 MWh.'
  },
  {
    id: 'solar', name: 'Solar', x: -5, z: -6, color: '#d78b23', quiz: 'solarEnergy',
    description: 'Follow sunlight from the receiving plane to the panel and inverter. Calculate DC power from irradiance, module area and efficiency, then account for temperature and conversion losses.',
    task: 'Compare panel output under two irradiance settings.',
    question: 'At 780 W/m², how much ideal DC power does a 2.1 m² panel with 21% efficiency produce?', options: ['343.98 W', '1,638 W', '78 W'], correct: 0,
    why: '780 × 2.1 × 0.21 = 343.98 W before system losses.'
  },
  {
    id: 'wind', name: 'Wind', x: 4, z: -7, color: '#318976', quiz: 'windEnergy',
    description: 'Follow kinetic energy through the rotor and drivetrain. Apply a turbine power curve to a wind-speed series to estimate annual energy and capacity factor.',
    task: 'Vary the wind speed and compare the resource with turbine output.',
    question: 'Wind speed doubles from 5 to 10 m/s. With fixed density and rotor area, how does available wind power change?', options: ['It doubles', 'It increases eightfold', 'It quadruples'], correct: 1,
    why: 'Available power scales with speed cubed, so the factor is 2³ = 8.'
  },
  {
    id: 'geothermal', name: 'Geothermal', x: 8, z: 2, color: '#be6547', quiz: 'geothermalEnergy',
    description: 'Trace production and reinjection through a geothermal reservoir. Use mass flow and the temperature drop to estimate extracted heat, then distinguish thermal output from electrical generation.',
    task: 'Follow the fluid circuit in the lab and identify where heat leaves the system.',
    question: 'Which unit describes the geothermal gradient?', options: ['W/m²', 'MWh', '°C/km'], correct: 2,
    why: 'Gradient measures temperature change per unit depth. W/m² measures heat flux.'
  }
];
