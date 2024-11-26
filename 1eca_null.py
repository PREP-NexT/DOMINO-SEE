#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pandas import to_datetime
from scipy.stats import binom

datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
vp = np.load("0data/prcp_validpoint_annual_100.npy")

th = -1.5  # Null结果与th无关
dT = 1

ev = (np.load("1event/{}_glb_spi1_event_drt{}.npz".format(datanm, th))["ev"]).sum(axis=1)
print("actual max:", ev.max())
evmax = 100
print("evmax set:", evmax)
na = np.arange(evmax).reshape(evmax, 1)
nb = np.arange(evmax).reshape(1, evmax)

for sig in [0.1, 0.05, 0.01, 0.005, 0.001]:
    prec = na - binom.ppf(sig, n=na, p=(1 - (dT * 2 + 1) / ddate.size)**nb)
    trig = nb - binom.ppf(sig, n=nb, p=(1 - (dT * 2 + 1) / ddate.size)**na)
    np.save("2eca/null/ecanull_{}_win{}_sig{}_evmax{}".format(datanm, dT, sig, evmax), np.stack((prec, trig), axis=2))
