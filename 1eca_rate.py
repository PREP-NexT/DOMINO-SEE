#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from pandas import to_datetime
from scipy.sparse import csr_matrix
from numba import njit
import mpi
# import multiprocessing  # Slow compared to mpi


@njit
def eca_nb(b1, b2, b1w, b2w):
    KRprec = np.zeros((b1.shape[0], b2.shape[0]), dtype='uint8')
    KRtrig = np.zeros((b1.shape[0], b2.shape[0]), dtype='uint8')
    for j in range(b1.shape[0]):
        for k in range(b2.shape[0]):
            KRprec[j, k] = np.sum(b1[j, :] & b2w[k, :])
            KRtrig[j, k] = np.sum(b2[k, :] & b1w[j, :])
    # print(j)
    # KRprec[0, :] = np.sum(b1 & b2w, axis=1)  # 广播在大矩阵时浪费时间
    # KRtrig[0, :] = np.sum(b2 & b1w, axis=1)
    return KRprec, KRtrig


def eca_window(b, symdelt=2):
    window = np.ones(2 * symdelt + 1)
    if symdelt == 0:
        bw = b  #.copy()
    else:
        bw = np.apply_along_axis(lambda x: np.convolve(x, window)[symdelt:-symdelt] >= 0.5, 1, b)
    return bw


def eca_mpnb_poisson(bX, bY, bwX, bwY, datanm, direc, th, core):
    path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_dT0/2eca'
    bX = bX.toarray()
    bY = bY.toarray()
    # bY = bX if bY is bX else bY.toarray()  #目前无法避免
    # assert (bY is bX) == (direc[0] == direc[1]), "Error in different data!"
    bwX = bwX.toarray()
    bwY = bwY.toarray()
    # bwY = bwX if bwY is bwX else bwY.toarray()

    tic = time.time()
    b1 = bX #[rows, :]
    b2 = bY
    b1w = bwX #[rows, :]
    b2w = bwY
    KRprec25, KRtrig25 = eca_nb(b1, b2, b1w, b2w)
    print("core: {}, Coincidence Event: {:.2f}".format(core, time.time() - tic))
    np.savez_compressed("{}/ecaevents_{}_glb_event{}_{}_c{}".format(path, datanm, direc, th, core), nu=np.stack((KRprec25, KRtrig25), axis=2))
    return 0


def master():
    print("Start Time: ", time.asctime())

    datanm = "spimv2"
    print("Dataset: ", datanm)
    lon = np.load('0data/{}_lon.npy'.format(datanm))
    lat = np.load('0data/{}_lat.npy'.format(datanm))
    latlon = np.load('0data/{}_latlon.npy'.format(datanm))
    ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))

    noc = 450  # 50
    th = 1.5
    dT = 0

    direc = "01"
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

    for core in range(noc):
        rows = np.arange(int(latlon.shape[0] / noc * core), int(latlon.shape[0] / noc * (core + 1)))
        mpi.submit_call("eca_mpnb_poisson", (evX[rows, :], evY, evwX[rows, :], evwY, datanm, direc, th, core), id=core)  # TODO: 选行之后X is Y识别不出来
        # eca_mpnb_poisson(ev, evw, 2, rows, datanm, th, core)
        print("batch %d submitted ..." % core)

    print("End Time: ", time.asctime(), "Process Time: ", time.process_time())


mpi.run()
