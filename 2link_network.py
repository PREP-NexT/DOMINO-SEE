#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Create   : 25/10/22 22:50 PM
@Author   : WANG HUI-MIN
@Update   : This update is to separate dist calc with dist separation
"""
import time
import sys
import numpy as np
from pandas import to_datetime
import scipy.sparse as sp
from numba import njit, prange, set_num_threads
set_num_threads(24)


datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
vp = np.load("0data/prcp_validpoint_annual_100.npy")
vp = vp.reshape(vp.size)

path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_temp'
opath = ""
th = 1.5
sig = 0.005

# %% Distance distribution
sin_lon = np.sin(latlon[:, 1] * np.pi / 180)
cos_lon = np.cos(latlon[:, 1] * np.pi / 180)
sin_lat = np.sin(latlon[:, 0] * np.pi / 180)
cos_lat = np.cos(latlon[:, 0] * np.pi / 180)


@njit(parallel=True)
def angdist_mb(lnk_n2n):
    """
    :param lnk_n2n: (2 * no. of links) row node index to column node index
    :return: angdist: angular great distances
    """
    angdist = np.zeros(lnk_n2n.shape[1], dtype='uint16')
    for l in prange(lnk_n2n.shape[1]):
        t = sin_lat[lnk_n2n[0, l]] * sin_lat[lnk_n2n[1, l]] + cos_lat[lnk_n2n[0, l]] * cos_lat[lnk_n2n[1, l]] * (
                sin_lon[lnk_n2n[0, l]] * sin_lon[lnk_n2n[1, l]] + cos_lon[lnk_n2n[0, l]] * cos_lon[lnk_n2n[1, l]])
        if t > 1.0:
            t = 1.0
        elif t < -1.0:
            t = -1.0
        angdist[l] = np.round(np.arccos(t) * 6371)
    return angdist


def link_to_grtdst(link, verbose=True):
    link = sp.coo_array(link)
    # lnk_n2n = np.vstack((link.row, link.col))
    dist = angdist_mb(np.vstack((link.row, link.col)))
    dist = sp.coo_array((dist, (link.row, link.col)), shape=link.shape)
    if verbose:
        print("Great Distances Calculation Accomplished")
    return dist


# %% Link to Great Circle distances
def direc_dist(direc):
    print('Direction ', direc)
    tic = time.time()

    # %% link collection
    link = sp.load_npz("{}/3link/link{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))
    print("Link Fraction: {:.2f}%".format(link.size / (vp.sum() ** 2) * 100))

    # %% link to angular distance
    dist = link_to_grtdst(link)
    sp.save_npz("3link{}/linkdist{}_{}_glb_event{}_{}.npz".format(opath, sig, datanm, direc, th), dist)
    print("Link Distances: {:.2f}s".format(time.time() - tic))
    print("Link Ref: ", sys.getrefcount(link))

    # %% Plot Distance Distribution
    # plot_loghist(dist, cos_lat,
    #              title="Drought-Drought Network",
    #              figname='pics/dist{}/densities_angdist{}_{}_glb_event{}_{}.jpg'.format(opath, sig, datanm, direc, th))
    return dist


def dist_split(dist):
    distth = 2500
    linkidx = dist.data >= distth
    lnk_tel = sp.csr_array((np.ones(linkidx.sum(), dtype='bool'), (dist.row[linkidx], dist.col[linkidx])),
                           shape=dist.shape)
    sp.save_npz("{}/3link/linktel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th), lnk_tel)
    del lnk_tel
    lnk_sht = sp.csr_array((np.ones((~linkidx).sum(), dtype='bool'), (dist.row[~linkidx], dist.col[~linkidx])),
                           shape=dist.shape)
    sp.save_npz("{}/3link/linkshr{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th), lnk_sht)
    del lnk_sht
    print("Distribution Ref: ", sys.getrefcount(dist))
    # del dist
    # gc.collect()
    return None


if __name__ == "__main__":
    for direc in ["00", "01", "11"]:
        print("direc: ", direc)
        dist = direc_dist(direc)
        del dist