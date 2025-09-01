#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
import mpi


@njit
def eca_nb(b1, b2, b1w, b2w):
    KRprec = np.zeros((b1.shape[0], b2.shape[0]), dtype='uint8')
    KRtrig = np.zeros((b1.shape[0], b2.shape[0]), dtype='uint8')
    for j in range(b1.shape[0]):
        for k in range(b2.shape[0]):
            KRprec[j, k] = np.sum(b1[j, :] & b2w[k, :])
            KRtrig[j, k] = np.sum(b2[k, :] & b1w[j, :])
    return KRprec, KRtrig


def eca_window(b, symdelt=2):
    window = np.ones(2 * symdelt + 1)
    if symdelt == 0:
        bw = b  #.copy()
    else:
        bw = np.apply_along_axis(lambda x: np.convolve(x, window)[symdelt:-symdelt] >= 0.5, 1, b)
    return bw


def eca_mpnb_poisson(bX, bY, bwX, bwY, datanm, direc, th, core):
    path = '2eca'
    bX = bX.toarray()
    bY = bY.toarray()
    bwX = bwX.toarray()
    bwY = bwY.toarray()

    tic = time.time()
    b1 = bX
    b2 = bY
    b1w = bwX
    b2w = bwY
    KRprec25, KRtrig25 = eca_nb(b1, b2, b1w, b2w)
    print("core: {}, Coincidence Event: {:.2f}".format(core, time.time() - tic))
    np.savez_compressed("{}/ecaevents_{}_glb_event{}_{}_c{}".format(path, datanm, direc, th, core), nu=np.stack((KRprec25, KRtrig25), axis=2))
    return 0


def master():
    print("Start Time: ", time.asctime())

    datanm = "spimv2"
    # datanm = "spimv2fkt"  # For Fekete grid
    print("Dataset: ", datanm)
    latlon = np.load('0data/{}_latlon.npy'.format(datanm))

    noc = 450  # 50
    th = 1.5
    dT = 1

    for direc in ["00", "01", "11"]:
        """
        direc = "00": drought synchronization
        direc = "01": drought-pluvial synchronization
        direc = "11": pluvial synchronization
        """
        # direc = "01"
        print('Direction: ', direc)
        infileX = "{}_glb_spi1_event_{}.npz".format(datanm, "drt{}".format(-th) if direc[0] == "0" else "fld{}".format(th))
        infileY = "{}_glb_spi1_event_{}.npz".format(datanm, "drt{}".format(-th) if direc[1] == "0" else "fld{}".format(th))

        evX = np.load("1event/{}".format(infileX))["ev"]
        print("Data X: {}".format(infileX))
        evwX = eca_window(evX, symdelt=dT)
        evX = csr_matrix(evX)
        evwX = csr_matrix(evwX)
        if direc[1] == direc[0]:
            evY = evX
            print("Data Y is Data X")
            evwY = evwX
        else:
            evY = np.load("1event/{}".format(infileY))["ev"]
            print("Data Y: {}".format(infileY))
            evwY = eca_window(evY, symdelt=dT)
            evY = csr_matrix(evY)
            evwY = csr_matrix(evwY)
        print("Reading Data th{}: {:.2f}s".format(th, np.nan))

        total_rows = latlon.shape[0]
        for core in range(noc):
            start_idx = int(total_rows / noc * core)
            end_idx = int(total_rows / noc * (core + 1)) if core < noc - 1 else total_rows
            rows = np.arange(start_idx, end_idx)
            mpi.submit_call("eca_mpnb_poisson", (evX[rows, :], evY, evwX[rows, :], evwY, datanm, direc, th, core), id=f"{direc}_{core}")
            # eca_mpnb_poisson(ev, evw, 2, rows, datanm, th, core)
            print(f"batch {core} submitted for rows from {start_idx} to {end_idx}...")

        print("End Time: ", time.asctime(), "Process Time: ", time.process_time())


mpi.run()
