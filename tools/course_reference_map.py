"""Concept-specific sources for the original course exercise bank.

The books support the physical relationships, not the synthetic input numbers.
Keep these locators next to the exercise rather than a chapter-wide citation list.
"""
UNITS = 'Unit definitions/conversions: [NIST SP 811, Appendix B](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).'
ENERGY = 'Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).'
HYDRO = 'Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7.'
WIND = 'Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.'
PV = 'PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2.'
GEO = 'Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.'
PANDAS = 'Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).'
NUMPY = 'Array calculations: [@numpyDocs].'
LCOE = 'Discounted cost and energy: [@ifc2015], §13.2.5 and Chapter 14; [@goswami2015], §1.4.3, Eq. 1.6; [@irena2026], methodology annex.'
CARNOT = 'Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle).'
CIRCULAR = 'Circular mean and zero-resultant limitation: [SciPy circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).'
WEIBULL = 'Wind distribution and energy: [@manwell2009], §§2.4–2.5; [SciPy Weibull density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html).'
NOCT = 'Specified simplified NOCT approximation; compare the fuller [Sandia/SAM thermal model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/noct-cell-temperature/).'
PR = 'Yield and performance-ratio definitions: [Sandia PV performance metrics](https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/).'
STORAGE = 'Storage losses: [@wade2003], Chapter 5; [@manwell2009], §10.7. The dispatch policy is an original course assumption.'

EXERCISE_REFERENCES = {
 'general': [UNITS, ENERGY, WIND, ENERGY, ENERGY, ENERGY, PANDAS, PANDAS, PANDAS,
             ENERGY+' Original no-storage balance.', PANDAS+' '+ENERGY, WIND,
             PANDAS+' [@sqliteDocs].', 'Chronological evaluation: [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).',
             'Plotting: [@matplotlibDocs]. '+ENERGY, STORAGE, LCOE, PANDAS,
             'Original discrete design search under the stated shortfall target; '+NUMPY,
             'Original uncertainty propagation with stated distributions; [NumPy random sampling](https://numpy.org/doc/stable/reference/random/index.html).'],
 'solar': [PV, PV, ENERGY+' '+PV, PV, 'Capacity-factor energy accounting: [@manwell2009], §2.5; this exercise explicitly uses the AC plant rating.', NOCT, 'Linear temperature correction: [@pvlibDocs].',
           PV+' The constant-efficiency clipping rule is specified here.', PV, PR, PR, ENERGY,
           PV+' '+NUMPY, 'Original compounded-degradation scenario; '+NUMPY,
           PV+' Plotting: [@matplotlibDocs].', PV+' Original discrete inverter search.', NOCT+' [@pvlibDocs].',
           STORAGE, LCOE+' Original degradation scenario.', PANDAS+' Chronological baseline evaluation; thresholds are course assumptions.'],
 'hydro': [HYDRO,HYDRO,ENERGY, 'Environmental/ecological releases: [@ifc2015], Chapters 7 and 12; the reservation is a teaching assumption.',
           HYDRO,HYDRO,HYDRO,HYDRO+' The quadratic loss coefficient is assumed.', ENERGY+' '+PANDAS,
           'Flow-duration ranking: [@jica2011], Chapters 7–8; the plotting-position formula is stated in the exercise.',HYDRO,
           'Pumped-storage conversion: [@jrc2025hydro], §2; efficiencies are illustrative.',HYDRO,
           '[ @ifc2015 ], Chapters 7 and 12; separate environmental demand from water physically available.',
           HYDRO+' Plotting: [@matplotlibDocs].',HYDRO+' Original chronological water-balance policy.',
           HYDRO+' Storage–head relationship is synthetic.',HYDRO+' Environmental flow is a scenario, not a regulatory recommendation.',
           HYDRO+' Original capacity search.', 'Pumped storage: [@jrc2025hydro], §2; original dispatch and water balance.'],
 'wind': [WIND,WIND,WIND,WIND,WIND, 'Height profiles: [@manwell2009], §2.3.4.2.',WIND,
          WIND+' Cubic ramp is a teaching approximation, not a measured curve.',WIND,
          'Turbulence intensity: [@manwell2009], §2.3; population standard deviation is explicitly assumed.', CIRCULAR,
          'Direction-sector convention specified in this exercise; [@manwell2009], Chapter 2; '+NUMPY,WIND,ENERGY,
          'Wind statistics: [@manwell2009], §2.4; [@matplotlibDocs].',WEIBULL,CIRCULAR,
          'Power-law shear: [@manwell2009], §2.3.4.2; logarithmic least squares is the stated estimator.',WIND,
          'Bootstrap sampling: [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Independence is assumed here; serial wind data may require block resampling.'],
 'geothermal': [GEO,GEO,GEO,ENERGY+' '+GEO,UNITS,GEO,GEO,GEO,GEO,GEO,CARNOT,GEO,
                'Original compounded-decline scenario; '+GEO,GEO+' '+NUMPY,GEO+' Plotting: [@matplotlibDocs].',
                'Original exponential decline fit with a chronological holdout; '+NUMPY,GEO+' The shared plant limit is assumed.',
                GEO+' Pump-load polynomial is a synthetic penalty, not a measured pump curve.',
                GEO+' Intervention timing and restoration are hypothetical.',GEO+' Independent uniform distributions are hypothetical; '+NUMPY]
}
EXERCISE_REFERENCES['hydro'][13]=EXERCISE_REFERENCES['hydro'][13].replace('[ @ifc2015 ]','[@ifc2015]')
assert all(len(refs)==20 for refs in EXERCISE_REFERENCES.values())
