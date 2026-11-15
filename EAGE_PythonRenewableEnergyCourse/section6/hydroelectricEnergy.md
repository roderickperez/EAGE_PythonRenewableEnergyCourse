---
kernelspec:
  name: python3
  display_name: Python 3
---

# Hydroelectric Energy

## Learning goals

By the end of this lesson you should be able to:

- distinguish head, flow, power, and energy;
- calculate gross and net hydraulic power with consistent SI units;
- implement reusable Python functions with input checks;
- convert a flow time series into power and energy;
- interpret capacity factor, turbine limits, and pumped-storage efficiency;
- create and explain engineering plots rather than only calculate one number.

Hydropower converts the gravitational potential energy of water into shaft power and then electricity. Conventional storage plants, run-of-river plants, and pumped-storage plants use the same energy balance but have different operating constraints. Head and flow determine the resource; turbine-generator efficiency and equipment limits determine the electrical output. See the U.S. Department of Energy overview and turbine guidance [@doeHydropowerWorks; @doeHydroTurbines].

:::{iframe} ../../hydro-lab/index.html
:width: 100%
:title: Interactive three-dimensional hydroelectric plant and engineering dashboard
:::

<!-- expanded-theory:hydro:start -->

## Physical description: water, elevation and electricity

Hydropower converts part of the mechanical energy of moving water into electricity. In the plants considered here, gravity drives water from a higher hydraulic-energy level to a lower one. A turbine converts part of that energy into shaft rotation, and a generator converts shaft power into electrical power. The ongoing hydrological cycle replenishes river flow, but the amount of water available at a particular location is finite and varies in time [@jica2011; @ifc2015].

A useful first description of a site therefore needs **both flow and head**. A large river is not necessarily a high-power site if little usable head is available. A mountain site may have high head but limited flow. Plant rating, annual energy and dependable dry-season output are separate design quantities. A station can have a large generator and a low capacity factor if water availability or operating constraints prevent continuous rated operation. Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

### Main system arrangements

| Arrangement | Defining feature | Main modelling implication |
|---|---|---|
| Run-of-river | Uses river flow with limited regulation in the simplified course model | Output closely follows available flow after required releases |
| Storage reservoir | Retains water so some inflow can be used later | Track storage, releases, spill and changing head chronologically |
| Diversion scheme | Conveys water through a canal, tunnel or penstock to gain useful head | Include conveyance losses and impacts along the bypassed river reach |
| Pumped storage | Uses electricity to lift water, then generates during release | Track both pumping consumption and recovered generation |

*Source for these definitions/calculations:* [@ifc2015], Chapters 2–3; [@jrc2025hydro], §2 (pumped storage); the equations are energy balances with the stated loss boundaries.

These categories can overlap: a run-of-river project may have some pondage, and a reservoir plant may also be part of a cascade. A pumped-storage scheme can have natural inflows as well as pumped water. Its total generation cannot automatically be treated as primary renewable production without appropriate accounting [@jica2011; @jrc2025hydro].

### Components and the flow path

The typical sequence is **intake → screen/trash rack → conveyance system → turbine → tailrace**, with **turbine shaft → generator → transformer → electrical network** on the energy-conversion side. Gates regulate flow; a spillway passes water that cannot or should not enter the turbines. An intake screen intercepts debris, while sediment-management facilities may be needed to limit abrasion and deposition. [@jica2011], Chapters 3 and 9; [@ifc2015], Chapter 3.

The penstock conveys pressurized water to the turbine. A surge arrangement may be required to manage pressure transients. The tailrace returns water downstream. In a reaction-turbine installation, the draft tube is part of the pressure and velocity-energy recovery process; it is not an additional source of energy. Hydraulic design must consider the full water passage rather than only the turbine nameplate. [@jica2011], Chapters 3 and 9; [@ifc2015], Chapter 3.

## Head, flow and the hydraulic energy balance

### What “head” means

Head is mechanical energy per unit **weight** of water, expressed in metres. For a steady incompressible flow, the elevation, pressure and velocity terms are [@jica2011], §8.3 and Chapter 9; [@ifc2015], Chapter 7. The velocity-head expression assumes kinetic-energy correction factor one.

$$H=z+\frac{p}{\rho g}+\frac{v^2}{2g}.$$

*Equation basis:* [@jica2011], §8.3 and Chapter 9; [@ifc2015], Chapter 7. The velocity-head expression assumes kinetic-energy correction factor one.

$z$ is elevation relative to a common datum; $p$ is pressure using a consistent pressure reference; $v$ is mean flow velocity; $\rho$ is water density; and $g$ is gravitational acceleration. A difference in water-surface elevation is a useful approximation to gross head between large upstream and downstream water bodies when their pressure and velocity terms largely cancel. More general installations require the complete hydraulic-head difference. [@jica2011], §8.3 and Chapter 9; [@ifc2015], Chapter 7. The velocity-head expression assumes kinetic-energy correction factor one.

The course uses

$$H_{net}=H_{gross}-h_{loss}.$$

*Equation basis:* [@jica2011], §8.3 and Chapter 9; [@ifc2015], Chapter 7. The velocity-head expression assumes kinetic-energy correction factor one.

The reference points must be specified consistently. Installation geometry and unrecovered outlet kinetic energy may matter as well as pipe friction; the JICA manual discusses effective head for different turbine arrangements. Do not subtract a loss twice if it has already been included in a supplied net-head value (Chapter 8) [@jica2011].

### Flow and volume are different quantities

Volume flow $Q$ is measured in m³/s. Water volume over an interval is $V=Q\Delta t$ for constant or interval-average flow, with time in seconds. Mass flow is $\dot m=\rho Q$ in kg/s. For example, 3 m³/s sustained for two hours moves $3\times7200=21{,}600$ m³; the water volume is not 6 m³. Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Unit definitions/conversions: [NIST SP 811, Appendix B](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).

The same distinction matters in reservoir routing. Electrical energy may be integrated using MW and hours, while the water balance uses m³/s and seconds. Keeping separate conversion factors in Python prevents a factor-of-3600 error. Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Unit definitions/conversions: [NIST SP 811, Appendix B](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).

### Deriving the power equation

A water mass $m$ descending through usable head $H$ has available mechanical energy $mgH$. For a continuously flowing mass rate $\dot m=\rho Q$, the available hydraulic power is Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

$$P_{hyd}=\rho gQH_{net}.$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

The electrical conversion chain gives

$$P_e=\eta_t\eta_g\rho gQH_{net},$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

where $\eta_t$ is turbine efficiency and $\eta_g$ is generator efficiency. A combined efficiency $\eta$ may include other explicitly identified losses. If station auxiliaries are not included in $\eta$, subtract their power separately when calculating net export. A quoted turbine efficiency must not be silently treated as the efficiency of the complete station [@jica2011; @ifc2015].

| Symbol | Definition | Unit used here |
|---|---|---|
| $Q$ | Flow actually passing through operating turbines | m³/s |
| $H_{gross}$, $H_{net}$ | Gross and usable turbine head | m |
| $h_{loss}$ | Hydraulic head dissipated or otherwise unavailable under the chosen boundary | m |
| $\rho$ | Water density, approximated as 1000 in these exercises | kg/m³ |
| $g$ | Gravitational acceleration, approximated as 9.81 | m/s² |
| $\eta_t$, $\eta_g$ | Conversion efficiencies | Fractions, not percentages in code |
| $P_e$, $P_r$ | Electrical output and electrical rating at a stated boundary | W or MW |

*Source for these definitions/calculations:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

Dimensional check: $(\mathrm{kg/m^3})(\mathrm{m/s^2})(\mathrm{m^3/s})(\mathrm m)=\mathrm{kg\,m^2/s^3}=\mathrm W$. Divide by $10^6$ to obtain MW. Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

## Hydraulic losses and turbine operation

For a circular pipe, a useful friction model is the Darcy–Weisbach equation, [@jica2011], §8.3 and Chapter 10; [@ifc2015], Chapter 3.

$$h_f=f_D\frac{L}{D}\frac{v^2}{2g},\qquad
v=\frac{Q}{A_p},\qquad A_p=\frac{\pi D^2}{4}.$$

*Equation basis:* [@jica2011], §8.3 and Chapter 10; [@ifc2015], Chapter 3.

$f_D$ is the dimensionless **Darcy** friction factor, $L$ and $D$ are pipe length and diameter in metres, and $A_p$ is pipe cross-sectional area. Do not substitute the numerically different Fanning friction factor without conversion. Fittings and transitions may be represented by $h_m=\sum K_jv^2/(2g)$ with appropriately defined local velocities. If geometry and coefficients are held fixed, losses scale approximately with $Q^2$. Thus doubling turbine flow need not double output: net head may fall as flow increases. [@jica2011], §8.3 and Chapter 10; [@ifc2015], Chapter 3.

The Python exercises sometimes use a supplied loss value or an illustrative coefficient $h_{loss}=aQ^2$. That coefficient is not a universal property of hydropower. A real estimate needs geometry, roughness, fluid properties and a defensible hydraulic model. [@jica2011], §8.3 and Chapter 10; [@ifc2015], Chapter 3.

### Turbine families

**Impulse turbines**, including Pelton machines, extract energy from a high-speed jet acting on the runner. **Reaction turbines**, including Francis and Kaplan/propeller arrangements, operate with pressure changes through the runner. Pelton machines are commonly associated with higher-head, lower-flow applications; Kaplan/propeller machines suit many lower-head, larger-flow settings; Francis machines occupy a broad intermediate range. These are qualitative tendencies, not rigid selection thresholds [@jica2011; @doeHydroTurbines].

Selection also depends on flow variation, rotational speed, cavitation, sediment, mechanical constraints and cost. Efficiency changes away from the design operating point. Minimum stable flow and turbine start/stop behavior mean that a constant-efficiency model may overstate output at very low discharge. Cavitation occurs when local pressure permits vapor cavities to form and subsequently collapse; it can damage components and constrain installation and operation. [@jica2011], Chapter 10; [@doeHydroTurbines].

The simple power cap $P=\min(\eta\rho gQH,P_r)$ is adequate for introductory energy calculations. It does not explicitly model gate position, turbine flow or pressure transients. At fixed head and efficiency, a corresponding flow limit is $Q_r=P_r/(\eta\rho gH)$. If water routing matters, allocate no more than that flow to the turbine and route the remainder to a bypass, spillway or storage. [@jica2011], Chapter 10; [@doeHydroTurbines].

## Water allocation, hydrology and storage

### Environmental releases before generation

An elementary allocation rule is

$$Q_{env,served}=\min(Q_{river},Q_{env,target}),$$
$$Q_{usable}=\max(Q_{river}-Q_{env,target},0).$$

*Equation basis:* [@ifc2015], Chapters 7 and 12; the fixed reservation and routing rule are course assumptions.

If river flow is below the target, generation cannot create the missing environmental water. Record the shortfall separately. Actual environmental-flow rules may be seasonal, ecological and legally defined, and water may also be allocated to irrigation, water supply, flood control or navigation. A fixed reservation used in an exercise is an assumption, not a site prescription [@ifc2015].

### Hydrographs and flow-duration curves

A **hydrograph** retains the time sequence of flow. A **flow-duration curve** sorts flows from largest to smallest and plots them against an exceedance-frequency convention. The latter helps describe how often a flow is equalled or exceeded, but discards chronological information. The course uses plotting positions $r/(n+1)$ for descending rank $r$; other conventions exist and should be named. [@jica2011], Chapters 7–8; [@ifc2015], Chapter 7.

A duration curve can support a simplified run-of-river energy estimate, but it cannot on its own determine reservoir operation or a sequence of shortages. Long-term assessment should cover wet and dry conditions, missing observations, upstream abstractions and changes in the catchment. A four-day exercise is not evidence of firm year-round production. [@jica2011], Chapters 7–8; [@ifc2015], Chapter 7.

### Reservoir conservation and variable head

For a time step with constant or interval-average flows,

$$S_{t+1}=S_t+
(Q_{in}-Q_{turb}-Q_{env}-Q_{spill})\Delta t
-V_{evap}-V_{other}.$$

*Equation basis:* [@jica2011], §8.3.2; the time-discrete water balance follows conservation under the stated separate flow paths.

Storage $S$ and losses $V$ are in m³, flows are m³/s, and $\Delta t$ is seconds. Environmental flow and turbine release must be defined as distinct water paths; do not count the same discharge twice. A route with minimum storage or dead storage must limit withdrawals to water above that bound. Enforce $S_{min}\le S\le S_{max}$ and preserve mass balance at every step. [@jica2011], §8.3.2; the time-discrete water balance follows conservation under the stated separate flow paths.

Reservoir level and tailwater can change, so head is generally a function of storage and discharge. Constant head is a useful simplification, particularly for short demonstrations, but can bias energy in a large drawdown. State whether within-step inflow can be used immediately and whether head is evaluated at the beginning, midpoint or end of the step. [@jica2011], §8.3.2; the time-discrete water balance follows conservation under the stated separate flow paths.

## Pumped storage and energy accounting

Moving a volume $V$ upward through constant head $H$ stores approximately $E_{pot}=\rho gVH$ joules. Pumping and generation require separate efficiency directions: [@jrc2025hydro], §2 (pumped storage); the equations are energy balances with the stated loss boundaries.

$$E_{pump}=\frac{E_{pot}}{\eta_p},\qquad
E_{gen}=\eta_t\eta_gE_{pot}.$$

*Equation basis:* [@jrc2025hydro], §2 (pumped storage); the equations are energy balances with the stated loss boundaries.

With the same head and water volume on both legs and no additional losses, $\eta_{RT}=\eta_p\eta_t\eta_g$. In a partial cycle, remaining water still contains stored energy; comparing generated energy with pumping input before accounting for that remaining storage gives an incomplete efficiency calculation. Pumped storage provides timing and flexibility services while consuming net electrical energy over a complete cycle [@jrc2025hydro].

## Worked calculation before coding

Take a synthetic river flow of 12 m³/s, environmental reservation of 2 m³/s, gross head of 30 m, hydraulic loss of 2 m and combined turbine-generator efficiency of 0.85. Original numerical substitution; Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

1. Available turbine flow is $12-2=10$ m³/s.
2. Net head is $30-2=28$ m.
3. Uncapped electrical output is $1000(9.81)(10)(28)(0.85)=2{,}334{,}780$ W, or **2.33478 MW**.
4. A 2 MW generator limits exported power to **2 MW** under the simplified boundary.
5. Six hours at this output deliver **12 MWh**, not 12 MW.

*Source for these definitions/calculations:* Original numerical substitution; Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

If the model also routes water, calculate the flow needed for the capped output and handle the remaining available flow explicitly. Do not assume the generator cap eliminates water from the balance. This example omits separate auxiliary consumption and assumes unchanged conditions over six hours. Original numerical substitution; Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

## Environmental context and the route to Python

Hydropower does not burn fuel in the turbine, but the project can affect river continuity, sediment transport, fish movement, inundated land, water quality and downstream flow timing. Reservoir greenhouse-gas behavior is site-specific; “renewable” does not mean zero life-cycle impact. The IFC guide is useful for connecting engineering calculations to project development and environmental assessment [@ifc2015].

Exercises 1–5 establish head, power, energy and water-volume units. Exercises 6–15 introduce equipment limits, variable flows, losses and duration curves. Exercises 16–20 require chronological storage and conservation checks. Read JICA Chapters 3 and 8 before these calculations; consult the supplied IFC guide for hydrology and project context, and Paish's technical review for further small-hydro background [@jica2011; @ifc2015; @paish2002hydro].


<!-- expanded-theory:hydro:end -->

## Concepts and equations: compact reference

### Gross head, head loss, and net head

The gross head is the upstream-to-downstream elevation or hydraulic-head difference. Friction and local losses reduce the head available at the turbine: Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

$$H_{net}=H_{gross}-h_{loss}$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

Always use $H_{net}$ in the plant power equation. Treating gross head as net head overestimates production. Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.

### Power and energy

For water density $\rho$, gravity $g$, volume flow $Q$, net head $H_{net}$, and combined turbine-generator efficiency $\eta$: Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

$$P_e=\eta\rho gQH_{net}$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

| Quantity | SI unit |
|---|---|
| $\rho$ | kg/m$^3$ |
| $g$ | m/s$^2$ |
| $Q$ | m$^3$/s |
| $H$ | m |
| $P$ | W |

*Source for these definitions/calculations:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

Power is an instantaneous rate. Energy over intervals $\Delta t_i$ is Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

$$E=\sum_i P_i\Delta t_i$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

If $P$ is in MW and $\Delta t$ in hours, $E$ is in MWh. For rated power $P_r$ and capacity factor $CF$ over $T$ hours: Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

$$E=P_r\,CF\,T,\qquad CF=\frac{E}{P_rT}$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

Capacity factor includes hydrology, outages, dispatch, and other constraints; it is not turbine efficiency. Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

### Equipment limits and storage

The generator limits electrical output:

$$P_{plant}=\min(P_e,P_r)$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. [@jrc2025hydro], §2 (pumped storage); the equations are energy balances with the stated loss boundaries.

For pumped storage, round-trip efficiency compares recovered generation with pumping energy: Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. [@jrc2025hydro], §2 (pumped storage); the equations are energy balances with the stated loss boundaries.

$$\eta_{RT}=\frac{E_{generated}}{E_{pumping}}$$

*Equation basis:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. [@jrc2025hydro], §2 (pumped storage); the equations are energy balances with the stated loss boundaries.

It is below one. Pumped storage shifts energy in time; it is not a net energy source. Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. [@jrc2025hydro], §2 (pumped storage); the equations are energy balances with the stated loss boundaries.

## A tested Python model

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

RHO_WATER = 1000.0  # kg/m3
G = 9.81            # m/s2

def hydro_power_mw(flow_m3s, gross_head_m, efficiency=0.90,
                   head_loss_m=0.0, rated_power_mw=None):
    """Return hydroelectric power in MW for scalar or array inputs."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [flow_m3s, gross_head_m, efficiency, head_loss_m]):
        raise ValueError("model inputs must be finite")
    flow = np.asarray(flow_m3s, dtype=float)
    net_head = np.asarray(gross_head_m, dtype=float) - head_loss_m
    if np.any(flow < 0):
        raise ValueError("flow_m3s must be non-negative")
    if np.any(np.asarray(head_loss_m) < 0):
        raise ValueError("head loss must be non-negative")
    if np.any(net_head < 0):
        raise ValueError("head loss cannot exceed gross head")
    if not 0 <= efficiency <= 1:
        raise ValueError("efficiency must be between 0 and 1")
    power = efficiency * RHO_WATER * G * flow * net_head / 1e6
    if rated_power_mw is not None:
        if not np.isfinite(rated_power_mw) or rated_power_mw < 0:
            raise ValueError("rated_power_mw must be non-negative")
        power = np.minimum(power, rated_power_mw)
    return float(power) if power.ndim == 0 else power

assert hydro_power_mw(0, 50) == 0
assert np.isclose(hydro_power_mw(100, 50, 0.90), 44.145)
```

The function combines basic Python ideas: arguments, defaults, conditionals, exceptions, NumPy arrays, and a return value.

## Worked time-series example

The following synthetic daily flows are teaching data, not a site assessment. A real study needs measured flow, an environmental-flow rule, a flow-duration curve covering wet and dry years, head-loss estimates, and an equipment-efficiency curve.

```{code-cell} python
rng = np.random.default_rng(7)
days = np.arange(1, 366)
seasonal_flow = 75 + 35 * np.sin(2 * np.pi * (days - 60) / 365)
flow_m3s = np.clip(seasonal_flow + rng.normal(0, 7, len(days)), 10, None)

power_mw = hydro_power_mw(
    flow_m3s, gross_head_m=48, head_loss_m=3,
    efficiency=0.90, rated_power_mw=35,
)
energy_gwh = power_mw.sum() * 24 / 1000
capacity_factor = energy_gwh * 1000 / (35 * 8760)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(days, flow_m3s, color="tab:blue")
axes[0].set(ylabel="Flow (m³/s)", title="Synthetic daily river flow")
axes[1].plot(days, power_mw, color="tab:green")
axes[1].axhline(35, color="black", linestyle="--", label="Rated power")
axes[1].set(xlabel="Day of year", ylabel="Power (MW)",
            title=f"Energy = {energy_gwh:.1f} GWh; CF = {capacity_factor:.1%}")
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
```

The flat segments at 35 MW are generator clipping, not constant river flow. This is why resource and output plots should be viewed together.

[Download the theory notebook](hydroelectricEnergy.ipynb).

## Common mistakes

- Using gross head when the equation requires net head.
- Confusing efficiency with capacity factor.
- Multiplying MW by seconds and labelling the result MWh.
- Ignoring generator clipping or environmental-flow requirements.
- Applying capacity factor to power that is already a time average.
- Treating pumped storage as a source of net electricity.

(chapter-practice-20-python-exercises)=
## Practice

[Open the hydroelectric exercises](../section7/workbook/hydroelectric.md) in the Applied Python Workbook.

## Check your understanding

Continue with the [hydropower quiz and Python challenge](../section7/renewableEnergyquizzes/hydroelectricEnergy.md).
