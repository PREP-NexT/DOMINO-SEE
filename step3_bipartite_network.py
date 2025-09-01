#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Create   : 28/2/23 9:09 PM
@Author   : WANG HUI-MIN
@Function : Regional bundle analysis for bipartite networks of breadbaskets
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from sklearn.neighbors import KernelDensity
import time
from functools import partial
from itertools import product
import multiprocessing
from utils.get_index_from_coord import get_index_for_square
from utils.distance import phaversine

datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
vp = np.load("0data/prcp_validpoint_annual_100.npy")
vp = vp.reshape(vp.size)
N = vp.sum()
print("Total Valid Points: {}".format(N))
path = ''

th = 1.5
sig = 0.005
distth = 2500

# %% 14 Regions
regions = {
           "North China": ([33.25, 43.0], [102.5, 121.0]),
           "South China": ([20.0, 33.0], [102.0, 122.5]),
           "East Europe": ([46.25, 55.5], [18.25, 58.0]),
           "West Europe": ([40.0, 56.0], [355.0, 18.0]),
           "Mediterranean": ([32.0, 46.0], [20.25, 44.0]),
           "East Africa": ([-3.0, 16.5], [23.5, 41.5]),
           "South Africa": ([-35.0, -15.0], [18.0, 35.0]),
           "Argentina": ([-39.0, -24.5], [295.0, 311.5]),
           "Canada": ([46.25, 55.0], [246.5, 264.0]),
           "East US": ([30.25, 46.0], [261.0, 275.0]),
           "West US": ([35.0, 48.0], [235.5, 246.0]),
           "Mexico": ([16.0, 30.0], [255.0, 266.0]),
           "Australia": ([-38.0, -26.0], [137.5, 153.0]),
           "India": ([8.0, 29.0], [70.0, 88.0])
           }
indices = {i: get_index_for_square(lat, lon, *c) for (i, c) in regions.items()}


def spherical_kde(values, xy, bw_opt):
    kde = KernelDensity(bandwidth=bw_opt, metric='haversine', kernel='gaussian', algorithm='ball_tree')
    kde.fit(values)
    datss = np.exp(kde.score_samples(xy))
    return datss


def null_spherical_kde(seed, pos, nol, xy, bw_opt):
    rng = np.random.default_rng(seed)
    values = pos[rng.choice(np.arange(pos.shape[0]), nol), :]
    datss = spherical_kde(values, xy, bw_opt)
    return datss

if __name__ == "__main__":
    print("Start Time: ", time.asctime())
    RERUN = False
    for direc in ["10", "00", "01", "11"]: 
        print('Direction ', direc)
        tic = time.time()
        if direc != "10":
            link = sp.load_npz("{}3link/linktel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th)).tocsr()
        else:
            link = sp.load_npz("{}3link/linktel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, "01", th)).tocoo()
            link = sp.csr_array((link.data, (link.col, link.row)), shape=link.shape)
        # del dist
        glb_frac = link.size / (vp.sum() ** 2)
        print("After Link Distance Threshold, Fraction: {:.2f}%".format(glb_frac * 100))

        # %% Region 1 and 2
        for rp in product(regions.keys(), regions.keys()):
            rx, ry = rp[0], rp[1]
            link_0_1 = link[indices[rx][2].reshape(-1, 1), indices[ry][2].reshape(1, -1)].toarray()
            nol = link_0_1.sum()
            print("Region {}-{}, Link Number:{}, Fraction: {:.2f}%".format(rx, ry, nol, nol / link_0_1.size * 100))
            
            savepath = "4bundle/bundle_{}-{}_{}_event{}_{}.npz".format(rx, ry, datanm, direc, th)
            if os.path.exists(savepath) and not RERUN:
                continue
            
            if nol > 0:
                rxdeg = link_0_1.sum(axis=1).reshape(indices[rx][0].size, indices[rx][1].size)
                rydeg = link_0_1.sum(axis=0).reshape(indices[ry][0].size, indices[ry][1].size)

                tic = time.time()
                pos = np.where(link_0_1)
                nsam = nol  # if nol < 50000 else 50000  
                bw_opt = 0.075 * nsam ** (-1. / (2 + 4))  
                # scott's rule = \sigma * n**(-1./(d+4)) 
                print("Bandwidth: {:.3f}".format(bw_opt))

                posx = pos[0]  # if nol != 50000 else np.random.choice(pos[0], nsam)
                # try: 
                #     print("KDE haversine std estimation: ", phaversine(indices[rx][3][posx, 1], indices[rx][3][posx, 0]).std())
                #     # TODO: This is really memory challenging
                # except:
                #     print()
                densx = spherical_kde(indices[rx][3][posx] * np.pi / 180., indices[rx][3] * np.pi / 180,
                                    bw_opt).reshape(indices[rx][0].size, indices[rx][1].size) * nsam
                print("NSam {}, Density in {}: {:.2f}s".format(nsam, rx, time.time() - tic))
                posy = pos[1]  # if nol != 50000 else np.random.choice(pos[1], nsam)
                # try:
                #     print("KDE haversine std estimation: ", phaversine(indices[ry][3][posy, 1], indices[ry][3][posy, 0]).std())
                # except:
                #     print()
                densy = spherical_kde(indices[ry][3][posy] * np.pi / 180., indices[ry][3] * np.pi / 180,
                                    bw_opt).reshape(indices[ry][0].size, indices[ry][1].size) * nsam
                print("NSam {}, Density in {}: {:.2f}s".format(nsam, ry, time.time() - tic))

                # Null density
                NNULL = 500
                noly = link[:, indices[ry][2]].sum() # 非对称临接矩阵
                nsam = np.round(noly * indices[rx][2].size / (N - indices[ry][2].size)).astype('int')  # 要用对方区域的期望links
                posxnull = indices[rx][3][vp[indices[rx][2]], :] * np.pi / 180.
                mp_kde_nullx = partial(null_spherical_kde, pos=posxnull, nol=nsam, xy=indices[rx][3] * np.pi / 180., bw_opt=bw_opt)
                with multiprocessing.Pool(processes=50) as p:
                    densxnull = np.vstack(p.map(mp_kde_nullx, np.arange(NNULL)*3, chunksize=1))
                densxnull = densxnull.reshape(-1, indices[rx][0].size, indices[rx][1].size) * nsam
                print("Nsam {}, Null KDE: {:.2f}s".format(nsam, time.time() - tic))

                nolx = link[indices[rx][2], :].sum()
                nsam = np.round(nolx * indices[ry][2].size / (N - indices[rx][2].size)).astype('int')  # 要用对方区域的期望links
                posynull = indices[ry][3][vp[indices[ry][2]], :] * np.pi / 180.
                mp_kde_nully = partial(null_spherical_kde, pos=posynull, nol=nsam, xy=indices[ry][3] * np.pi / 180., bw_opt=bw_opt)
                with multiprocessing.Pool(processes=50) as p:
                    densynull = np.vstack(p.map(mp_kde_nully, np.arange(NNULL)*3, chunksize=1))
                densynull = densynull.reshape(-1, indices[ry][0].size, indices[ry][1].size) * nsam
                print("Nsam {}, Null KDE: {:.2f}s".format(nsam, time.time() - tic))

                statsx = np.concatenate([np.mean(densxnull, axis=0, keepdims=True), np.std(densxnull, axis=0, keepdims=True),
                                        st.scoreatpercentile(densxnull, [90, 95, 99], axis=0)])
                statsy = np.concatenate([np.mean(densynull, axis=0, keepdims=True), np.std(densynull, axis=0, keepdims=True),
                                        st.scoreatpercentile(densynull, [90, 95, 99], axis=0)])
                sigdensx = np.zeros_like(densx)  # spatial link density
                sigdensx[densx > statsx[0] + 5 * statsx[1]] = 5.5
                sigdensx[densx <= statsx[0] + 5 * statsx[1]] = 4.5
                sigdensx[densx <= statsx[0] + 4 * statsx[1]] = 3.5
                sigdensx[densx <= statsx[0] + 3 * statsx[1]] = 2.5
                sigdensx[densx <= statsx[0] + 2 * statsx[1]] = 1.5
                # sigdensx[-1, :] = 1.5 
                # sigdensx[0, :] = 1.5
                bundlex = (densx > statsx[4])

                sigdensy = np.zeros_like(densy)  # spatial link density
                sigdensy[densy > statsy[0] + 5 * statsy[1]] = 5.5
                sigdensy[densy <= statsy[0] + 5 * statsy[1]] = 4.5
                sigdensy[densy <= statsy[0] + 4 * statsy[1]] = 3.5
                sigdensy[densy <= statsy[0] + 3 * statsy[1]] = 2.5
                sigdensy[densy <= statsy[0] + 2 * statsy[1]] = 1.5
                bundley = (densy > statsy[4])

                link_bundle = link_0_1 * bundlex.ravel().reshape((-1, 1)) * bundley.ravel()
                posx, posy = np.where(link_bundle)
                if posx.size > 0:
                    densx_bundle = spherical_kde(indices[rx][3][posx] * np.pi / 180., indices[rx][3] * np.pi / 180,
                                        bw_opt).reshape(indices[rx][0].size, indices[rx][1].size)
                    densy_bundle = spherical_kde(indices[ry][3][posy] * np.pi / 180., indices[ry][3] * np.pi / 180,
                                        bw_opt).reshape(indices[ry][0].size, indices[ry][1].size)
                else:
                    densx_bundle = np.zeros_like(densx)
                    densy_bundle = np.zeros_like(densy)

                np.savez("4bundle/bundle_{}-{}_{}_event{}_{}.npz".format(rx, ry, datanm, direc, th), rx=rx, ry=ry,
                        link_0_1=link_0_1, rxdeg=rxdeg, rydeg=rydeg,
                        densx=densx, densy=densy,
                        statsx=statsx, statsy=statsy, sigdensx=sigdensx, sigdensy=sigdensy,
                        bundlex=bundlex, bundley=bundley, densx_bundle=densx_bundle, densy_bundle=densy_bundle,
                        link_bundle=link_bundle)
                
                del densx, densy, densx_bundle, densy_bundle, link_bundle, bundlex, bundley, sigdensx, sigdensy, statsx, statsy
            else:
                np.savez("4bundle/bundle_{}-{}_{}_event{}_{}.npz".format(rx, ry, datanm, direc, th), rx=rx, ry=ry,
                        link_0_1=link_0_1, rxdeg=np.zeros((indices[rx][0].size, indices[rx][1].size)), rydeg=np.zeros((indices[ry][0].size, indices[ry][1].size)),
                        densx=np.zeros((indices[rx][0].size, indices[rx][1].size)), densy=np.zeros((indices[ry][0].size, indices[ry][1].size)),
                        sigdensx=np.zeros((indices[rx][0].size, indices[rx][1].size)), sigdensy=np.zeros((indices[ry][0].size, indices[ry][1].size)),
                        bundlex=np.zeros((indices[rx][0].size, indices[rx][1].size)), bundley=np.zeros((indices[ry][0].size, indices[ry][1].size)),
                        densx_bundle=np.zeros((indices[rx][0].size, indices[rx][1].size)), densy_bundle=np.zeros((indices[ry][0].size, indices[ry][1].size)),
                        link_bundle=link_0_1)
    
        del link
    print("End Time: ", time.asctime())
