"""Create five separate, deterministic input datasets; contains no project solution."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'EAGE_PythonRenewableEnergyCourse/section8/data'
ROOT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(2026)
time=pd.date_range('2025-06-01',periods=168,freq='h',tz='UTC')
hour=time.hour.to_numpy(); elapsed=np.arange(len(time))
solar_shape=np.maximum(np.sin(np.pi*(hour-5.5)/14),0)
cloud=np.clip(0.82+rng.normal(0,0.12,len(time)),0.35,1.05)
tables={
 'solar.csv':{'irradiance_wm2':950*solar_shape*cloud,'ambient_c':19+8*np.sin(2*np.pi*(hour-8)/24)},
 'wind.csv':{'wind_speed_ms':np.clip(rng.weibull(2.1,len(time))*8,0,32)},
 'hydro.csv':{'river_flow_m3s':np.clip(24+3*np.sin(2*np.pi*elapsed/168)+rng.normal(0,1.2,len(time)),0,None)},
 'geothermal.csv':{'mass_flow_kgs':70+3*np.sin(2*np.pi*elapsed/168),'production_c':160+2*np.cos(2*np.pi*elapsed/168),'reinjection_c':np.full(len(time),70.)},
 'demand.csv':{'demand_mw':23+5*np.sin(2*np.pi*(hour-16)/24)**2},
}
for name,columns in tables.items():
    pd.DataFrame({'timestamp':time,**columns}).to_csv(ROOT/name,index=False,float_format='%.8f')
    print(name,len(time),'rows')
