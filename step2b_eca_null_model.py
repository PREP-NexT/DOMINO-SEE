#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pandas import to_datetime
from scipy.stats import binom

datanm = "spimv2"
# datanm = "spimv2fkt"  # For Fekete grid
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))

th = 1.5  # Null结果与th无关，但需要知道最大值
dT = 1

ev = (np.load("1event/{}_glb_spi1_event_drt{}.npz".format(datanm, -th))["ev"]).sum(axis=1)
print("actual max (drought):", ev.max())
ev = (np.load("1event/{}_glb_spi1_event_fld{}.npz".format(datanm, th))["ev"]).sum(axis=1)
print("actual max (pluvial):", ev.max())
evmax = 110
print("evmax set:", evmax)
na = np.arange(evmax).reshape(evmax, 1)
nb = np.arange(evmax).reshape(1, evmax)

for sig in [0.1, 0.05, 0.01, 0.005, 0.001]:
    prec = na - binom.ppf(sig, n=na, p=(1 - (dT * 2 + 1) / ddate.size)**nb)
    trig = nb - binom.ppf(sig, n=nb, p=(1 - (dT * 2 + 1) / ddate.size)**na)
    np.save("2eca/null/ecanull_{}_win{}_sig{}_evmax{}".format(datanm, dT, sig, evmax), np.stack((prec, trig), axis=2))
