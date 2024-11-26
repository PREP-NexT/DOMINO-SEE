#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Create   : 25/10/22 10:50 PM
@Author   : WANG HUI-MIN
@Update   : Get to links in one file
"""
import numpy as np
import time
from pandas import to_datetime
# from scipy.stats import binom
import multiprocessing
import scipy.sparse as sp
from numba import njit, prange, set_num_threads
set_num_threads(24)

datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
vp = np.load("0data/prcp_validpoint_annual_100.npy")
vp = vp.reshape(-1)

noc = 450  # 50
th = 1.5
dT = 1  # TODO: add an examination to this
print("ECA Window is dT =", dT)
sig = 0.005

direc = "01"
print('Direction ', direc)
if direc[0] == "0":
    ev0 = np.load("1event/{}_glb_spi1_event_drt{}.npz".format(datanm, -th))["ev"]
elif direc[0] == "1":
    ev0 = np.load("1event/{}_glb_spi1_event_fld{}.npz".format(datanm, th))["ev"]
if direc[1] == direc[0]:
    ev1 = ev0
else:
    if direc[1] == "0":
        ev1 = np.load("1event/{}_glb_spi1_event_drt{}.npz".format(datanm, -th))["ev"]
    elif direc[1] == "1":
        ev1 = np.load("1event/{}_glb_spi1_event_fld{}.npz".format(datanm, th))["ev"]
NA = np.sum(ev0, axis=1).reshape(ev0.shape[0], 1)
NB = np.sum(ev1, axis=1).reshape(1, ev1.shape[0])
del ev0, ev1
Nnull = 100
assert (NA.max() <= Nnull) & (NB.max() <= Nnull), "Event number in null model is not sufficiently large"
ecanull = np.load("2eca/null/ecanull_{}_win{}_sig{}_evmax{}.npy".format(datanm, dT, sig, Nnull))
print("Reading Data th{}: {:.2f}s".format(th, np.nan))

path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1'


@njit(parallel=True)
def null_compare_nb(nu, NA, NB):
    assert nu.shape[0] == NA.size
    assert nu.shape[1] == NB.size
    link = np.zeros(nu.shape, dtype='bool')
    for x in range(nu.shape[0]):
        for j in range(nu.shape[1]):
            link[x, j, 0] = nu[x, j, 0] > ecanull[NA[x], NB[j], 0]  # TODO: 选择的大于是正确的
            link[x, j, 1] = nu[x, j, 1] > ecanull[NA[x], NB[j], 1]
    return link


def null_compare_mp(core):
    # print("core {}: start".format(core))
    rows = np.arange(int(latlon.shape[0] / noc * core), int(latlon.shape[0] / noc * (core + 1)))
    nu = np.load("{}/2eca/ecaevents_{}_glb_event{}_{}_c{}.npz".format(path, datanm, direc, th, core))["nu"]
    na = NA[rows, :]
    link = null_compare_nb(nu, na.ravel(), NB.ravel())
    link = np.all(link, axis=2)  # TODO: 这里采用了ALL Rule，需要确认
    link[~vp[rows], :] = False  # Memory explode if put outside1
    link[:, ~vp] = False
    link = sp.coo_array(link)  # TODO: 原来这里是csr
    print("core {}: frac {:.5f}%".format(core, link.sum() / np.product(link.shape) * 100))
    return link


# def null_compare(core):
#     rows = np.arange(int(latlon.shape[0] / noc * core), int(latlon.shape[0] / noc * (core + 1)))
#     na = NA[rows, :]
#     nu = np.load("{}/2eca/ecaevents_{}_glb_event{}_{}_c{}.npz".format(path, datanm, direc, th, core))["nu"]
#     # cnull = ecanull[NA, NB, :]  # 慢一些
#     pre_null = ecanull[na, NB, 0]  # 广播没有问题,需要12s

#     tri_null = ecanull[na, NB, 1]  # 这个选用numba节省内存也更快
#     link = np.zeros(nu.shape, dtype='bool')
#     link[:, :, 0] = nu[:, :, 0] > pre_null
#     link[:, :, 1] = nu[:, :, 1] > tri_null
#
#     np.savez_compressed("{}/3link/link_{}_glb_event{}_{}_c{}.npz".format(path, datanm, direc, th, core), link=link)
#     print("core {}: frac {}%".format(core, link.sum()/link.size * 100))


print("Start Time: ", time.asctime())
# link = null_compare_mp(0)
with multiprocessing.Pool(processes=22) as p:
    link = sp.vstack(p.map(null_compare_mp, np.arange(noc), chunksize=1))
print("Significance End: ", time.asctime())
# sp.save_npz("{}/3link/link{}_{}_glb_event{}_{}_all.npz".format(path, sig, datanm, direc, th), link, compressed=False)
# link = sp.load_npz("{}/3link/link{}_{}_glb_event{}_{}_all.npz".format(path, sig, datanm, direc, th))
link.setdiag(False)
link.eliminate_zeros()
sp.save_npz("{}/3link/link{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th), link, compressed=False)
print("End Time: ", time.asctime(), "Process Time: ", time.process_time())

print()
# 单核内存峰值 4.8G
# 10核完成multiprocessing要1h
# 可以用15核完成
