# Renewable-energy Python test

**25 questions · 100 marks · suggested duration 90 minutes.** Each question is worth 4 marks. Show units, assumptions and Python where requested. Use a fresh notebook for calculations. This assessment contains questions only. Practice answers elsewhere in the course are study material, not an answer key for this test.

## General

### Question 01 — 4 marks

Explain the distinction between power, energy, conversion efficiency and capacity factor; give units for the first two.

### Question 02 — 4 marks

A 4 MW plant delivers 36 MWh over 12 h. Write Python to compute its capacity factor and mean output.

### Question 03 — 4 marks

Interval-average powers[1,3,2] MW persist for[0.5,2,1.5] h. Write code for energy and time-weighted mean power.

### Question 04 — 4 marks

Two source CSV files have unique timestamps, but one lacks an hour. Explain why an inner join and filling missing values with zero can give misleading generation totals. Propose a validation rule.

### Question 05 — 4 marks

Hourly supply[2,6] MW serves demand[4,4] MW without storage. Compute served, curtailed, unmet MWh and demand coverage.

## Solar

### Question 06 — 4 marks

Distinguish irradiance and irradiation, including units; explain why adding hourly irradiance readings requires a time-step factor.

### Question 07 — 4 marks

At 900 W/m², area 10 m² and efficiency 18%, calculate DC kW in Python.

### Question 08 — 4 marks

A 10 kW STC array at 1000 W/m² has cell temperature 50°C and gamma−0.004/°C. A 97%-efficient inverter is capped at 8 kW AC. Compute DC, unclipped AC and exported AC.

### Question 09 — 4 marks

A 50 kW DC array exports 6000 kWh during a period with 150 kWh/m² POA irradiation. Calculate PR using reference irradiance 1 kW/m². Is PR cell efficiency?

### Question 10 — 4 marks

Explain how to compare two inverter ratings using the same hourly DC data, and identify two reasons that the rating with least clipping may not be the best design.

## Hydro

### Question 11 — 4 marks

Explain gross head, hydraulic head loss and net head. Which enters the electrical power equation?

### Question 12 — 4 marks

Calculate electrical MW for flow 6 m³/s, net head 40 m and efficiency 85%, using rho 1000 and g 9.81.

### Question 13 — 4 marks

River flows[1,5,10] m³/s must first reserve up to 3 m³/s. Write NumPy code for environmental release, shortage and usable turbine flow.

### Question 14 — 4 marks

A reservoir starts with 10000 m³, receives 2 m³/s for 1 h and releases 3 m³/s for 1 h. With no spill or evaporation, calculate final storage and state the water-balance equation.

### Question 15 — 4 marks

A complete pumped-storage cycle uses pump efficiency 80% and generation efficiency 90%, with equal head both ways and no other losses. Calculate round-trip efficiency and explain why output is not new primary renewable energy.

## Wind

### Question 16 — 4 marks

Explain why the cubic available-wind-power law must not be extrapolated to turbine electrical output above rated and cut-out speeds.

### Question 17 — 4 marks

Available wind power is 600 kW, Cp 0.42 and drivetrain efficiency 0.95. Calculate electrical output and compare Cp against 16/27.

### Question 18 — 4 marks

A 2 MW turbine uses cut-in 3, rated 12 and cut-out 25 m/s. Write a function returning zero below 3, 2(v³−27)/(1728−27) from 3 to 12, rated below 25, and zero at 25 or higher. Give outputs at[0,3,12,25].

### Question 19 — 4 marks

Directions are 355° and 5°. Explain the problem with an arithmetic mean and give the circular-mean procedure.

### Question 20 — 4 marks

Explain why a Weibull AEP calculation needs a turbine curve, wind-distribution probabilities and a convergence check. State how you would apply one combined 10% loss assumption.

## Geothermal

### Question 21 — 4 marks

Distinguish geothermal gradient, conductive heat flux, thermal power and net electrical power, including units.

### Question 22 — 4 marks

Use 40 kg/s, production 140°C, reinjection 60°C and cp 4180 J/(kg K). Calculate thermal MW, gross electrical MW at 10% and net MW after 15% parasitic fraction of gross.

### Question 23 — 4 marks

A 4 MW net plant has 90% availability during a 365-day year. Calculate annual GWh. Would the same answer apply to a 366-day year?

### Question 24 — 4 marks

Explain how to fit exponential decline and evaluate future-year predictions without leakage. Why does a perfect fit to noiseless synthetic data not establish reservoir forecasting accuracy?

### Question 25 — 4 marks

For production temperatures[150,60]°C, reinjection 70°C and positive flow, explain how to handle the second record in a power-generation model and what is needed for two-phase fluids.

## Submission

Submit one executed notebook with question numbers, workings and concise explanations. Do not include the final-project submission in this test.
