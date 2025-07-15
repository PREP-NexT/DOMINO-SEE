#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Create   : 25/10/22 10:50 PM
@Author   : WANG HUI-MIN
@Update   : Get to links in one file
"""
import numpy as np
import time
import multiprocessing
import scipy.sparse as sp
from numba import njit, prange, set_num_threads
set_num_threads(24)

datanm = "spimv2"
# datanm = "spimv2fkt"  # For Fekete grid
vp = np.load("0data/prcp_validpoint_annual_100.npy")
# vp = np.load("0data/prcpfkt_validpoint_annual_100.npy")  # For Fekete grid
vp = vp.reshape(-1)
latlon = np.load('0data/{}_latlon.npy'.format(datanm))

noc = 450  # 50
th = 1.5
dT = 1
print("ECA Window is dT =", dT)
sig = 0.005

for direc in ["00", "01", "11"]:
    # direc = "01"
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
    Nnull = 110
    assert (NA.max() <= Nnull) & (NB.max() <= Nnull), "Event number in null model is not sufficiently large"
    ecanull = np.load("2eca/null/ecanull_{}_win{}_sig{}_evmax{}.npy".format(datanm, dT, sig, Nnull))
    print("Reading Data th{}: {:.2f}s".format(th, np.nan))

    path = '.'


    @njit(parallel=True)
    def null_compare_nb(nu, NA, NB):
        assert nu.shape[0] == NA.size
        assert nu.shape[1] == NB.size
        link = np.zeros(nu.shape, dtype='bool')
        for x in range(nu.shape[0]):
            for j in range(nu.shape[1]):
                link[x, j, 0] = nu[x, j, 0] > ecanull[NA[x], NB[j], 0]
                link[x, j, 1] = nu[x, j, 1] > ecanull[NA[x], NB[j], 1]
        return link


    def null_compare_mp(core):
        start_idx = int(latlon.shape[0] / noc * core)
        end_idx = int(latlon.shape[0] / noc * (core + 1)) if core < noc - 1 else latlon.shape[0]
        rows = np.arange(start_idx, end_idx)
        nu = np.load("{}/2eca/ecaevents_{}_glb_event{}_{}_c{}.npz".format(path, datanm, direc, th, core))["nu"]
        na = NA[rows, :]
        link = null_compare_nb(nu, na.ravel(), NB.ravel())
        link = np.all(link, axis=2)  # TODO: 这里采用了ALL Rule，需要确认
        link[~vp[rows], :] = False  # Memory explode if put outside1
        link[:, ~vp] = False
        link = sp.coo_array(link)  
        print("core {}: frac {:.5f}%".format(core, link.sum() / np.prod(link.shape) * 100))
        return link


    print("Start Time: ", time.asctime())
    # link = null_compare_mp(0)
    with multiprocessing.Pool(processes=40) as p:
        link = sp.vstack(p.map(null_compare_mp, np.arange(noc), chunksize=1))
    print("Significance End: ", time.asctime())
    # sp.save_npz("{}/3link/link{}_{}_glb_event{}_{}_all.npz".format(path, sig, datanm, direc, th), link, compressed=False)
    # link = sp.load_npz("{}/3link/link{}_{}_glb_event{}_{}_all.npz".format(path, sig, datanm, direc, th))
    link.setdiag(False)
    link.eliminate_zeros()
    sp.save_npz("{}/3link/link{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th), link, compressed=False)
    print("End Time: ", time.asctime(), "Process Time: ", time.process_time())

    del link, ecanull, NA, NB
    print()
