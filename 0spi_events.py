#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from pathlib import Path
from itertools import product, groupby
import numpy as np
import xarray as xr
from pandas import to_datetime

tic = time.time()
datanm = "spimv2"  # Define dataset name for later use
data = xr.open_dataset("0data/SPI1_monthly_0.250deg_1950_2016.nc").sel(t=slice("1950", "2016"))
# datanm = "spimv2fkt"  # For Fekete grid
# data = xr.open_dataset("0data/Fekete/SPI1_monthly_fekete025_1948_2016.nc").rename({"time": "t", 
#                                                                                    "SPI1": "spi1"}).sel(t=slice("1950", "2016"))

# %% Load data
lat = data.lat.to_numpy()
lon = data.lon.to_numpy()
np.save('0data/{}_lat.npy'.format(datanm), np.array(lat))
np.save('0data/{}_lon.npy'.format(datanm), np.array(lon))
latlon = np.array(list(product(lat, lon))) if data.spi1.ndim == 3 else np.array([lat, lon]).T
np.save('0data/{}_latlon.npy'.format(datanm), latlon)
ddate = to_datetime(data.t.to_numpy())
np.save('0data/{}_date.npy'.format(datanm), ddate)  # [ddate.year >= 1990]
spi = data.spi1.stack(latlon=("lat", "lon")).to_numpy().T if data.spi1.ndim == 3 else data.spi1.to_numpy().T # 先滚经度再滚纬度
# ddate.year >= 1990
# ddate = ddate[ddate.year >= 1990]
# pcp = xr.open_dataarray("data/PGFv3_prec_yearly_0.250deg_1948_2016.nc").datanm[2:, :, :]

t = ddate.shape[0]  # Time points
la = lat.shape[0]  # Latitude points
lo = lon.shape[0]  # Longitude points
n = la * lo  # Space points
if np.ma.isMaskedArray(spi):
    spi = spi.filled(0)

y = ddate.year.max() - ddate.year.min() + 1  # Years


# %% 计算bursts
def drought_time(ts, th, burst=False):  # events
    te = ts <= th  # time of events
    if burst:
        tb = te.copy()  # time of bursts
        tb0 = np.roll(tb, 1)
        tb0[:, 0] = False
        tb[tb & tb0] = False
        return te, tb
    else:
        return te


def durations(te):
    du_num = np.concatenate([[len(list(j)) for i, j in groupby(x) if i] for x in te]).astype('uint16')
    due = np.zeros_like(te, dtype='uint16')
    due[te] = np.repeat(du_num, du_num)
    return due


def flood_time(ts, th, burst=False):
    te = ts >= th
    if burst:
        tb = te.copy()  # time of bursts
        tb0 = np.roll(tb, 1)
        tb0[:, 0] = False
        tb[tb & tb0] = False
        return te, tb
    else:
        return te


print("Event timing starts: {:.3f}s".format(time.time() - tic))

event_dir = Path("1event")
if not event_dir.exists():
    event_dir.mkdir()
    print("Created directory: 1event")

for perc in [-1.5]:  # [-1, -1.5, -2, -3]:
    ev, eb = drought_time(spi, perc, burst=True)
    np.savez_compressed("1event/{}_glb_spi1_event_drt{}".format(datanm, perc), ev=ev)  # event timing
    np.savez_compressed("1event/{}_glb_spi1_burst_drt{}".format(datanm, perc), eb=eb)  # event timing
    evdu = durations(ev)
    np.savez_compressed("1event/{}_glb_spi1_duration_drt{}".format(datanm, perc), evdu=evdu)
    print("Threshold {}th: {:.3f}s".format(perc, time.time() - tic))

for perc in [1.5]:  # [1, 1.5, 2, 3]:
    ev, eb = flood_time(spi, perc, burst=True)
    np.savez_compressed("1event/{}_glb_spi1_event_fld{}".format(datanm, perc), ev=ev)  # event timing
    np.savez_compressed("1event/{}_glb_spi1_burst_fld{}".format(datanm, perc), eb=eb)  # event timing
    evdu = durations(ev)
    np.savez_compressed("1event/{}_glb_spi1_duration_drt{}".format(datanm, perc), evdu=evdu)
    print("Threshold {}th: {:.3f}s".format(perc, time.time() - tic))

print()