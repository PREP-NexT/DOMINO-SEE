#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Create   : 30/6/22 9:09 PM
@Author   : WANG HUI-MIN
@         : This script
"""
# %%
import numpy as np
import pandas as pd
from pandas import to_datetime
import scipy.sparse as sp
import time
from get_index_from_coord import get_index_for_square
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle
import seaborn as sns
import proplot as pplt # TODO: 这个会改变rcParams
pplt.rc["font.sans-serif"] = "Myriad Pro"
pplt.rc["font.largesize"] = "large"
pplt.rc["tick.minor"] = False

# %%
datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
vp = np.load("0data/prcp_validpoint_annual_100.npy")
vp = vp.reshape(vp.size)
path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_temp'

th = 1.5
sig = 0.005

direc = "10"
print('Direction ', direc)
tic = time.time()

TYPEX = "Drought" if direc[0] == "0" else "Pluvial"
TYPEY = "Drought" if direc[1] == "0" else "Pluvial"
TITLES = {"00": "Drought synchronization", "01": "Drought-pluvial synchronization", "10": "Pluvial-drought synchronization", "11": "Pluvial synchronization"}
LABELS = {"00": "C", "01": "A", "10": "B", "11": "D"}
RECTS = {"00": (4.5, -0.5), "01": (6.5, 4.5), "10": (9.5, 1.5), "11": (5.5, 0.5)}

ff = np.load("4bundle/fracs_tele_{}_sig{}.npz".format(direc if direc != "10" else "01", sig))
fracsx = ff["fracsx"] if direc != "10" else ff["fracsy"].T
fracsy = ff["fracsy"] if direc != "10" else ff["fracsx"].T

# %%  Multiple Regions
regions = {"North China": ([33.25, 43.0], [102.5, 121.0]),
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
           "India": ([8.0, 29.0], [70.0, 88.0])}
region_list = list(regions.keys())
indices = {i: get_index_for_square(lat, lon, *c) for (i, c) in regions.items()}
NR = len(regions)

# %% Plot density plots
link = sp.load_npz("{}/3link/linktel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))  # .tocoo()
print("Tele, Number: {:.0f}, Fraction: {:.2f}%".format(link.size, link.size / (vp.size ** 2) * 100))

# plot link fractions
fracs = np.zeros((NR, NR))
for x in range(NR):
    for y in range(NR):
        rx = region_list[x]
        ry = region_list[y]
        link_x_y = link[indices[rx][2].reshape(-1, 1), indices[ry][2].reshape(1, -1)].A
        nol = link_x_y.sum()
        fracs[x, y] = nol / link_x_y.size * 100

np.fill_diagonal(fracs, 0.)
fracs = pd.DataFrame(data=fracs, index=region_list, columns=region_list)
mask = np.triu(np.ones_like(fracs, dtype=bool))

fig, ax = plt.subplots(figsize=(4.4, 3.65), tight_layout=True)
# cmap = mpl.cm.Reds
CMAP = {"00": mpl.cm.Reds, "11": mpl.cm.Blues, "01": mpl.colors.LinearSegmentedColormap.from_list("tab:purples", ["white", "darkslateblue"])}
sns.heatmap(fracs.iloc[:, :], cmap=CMAP[direc],  # mask=mask, # vmax=.3, center=0,
            square=True, cbar_kws ={"label": "Network Density [%]", "fraction": 0.05},
            ax=ax)  # "shrink": .5
ax.set_xlabel(TYPEY, fontsize="large", fontweight='bold')
ax.set_ylabel(TYPEX, fontsize="large", fontweight='bold')
ax.set_xticklabels(region_list, rotation=45, ha="right", rotation_mode="anchor")

for ax0 in fig.axes:
    for spine in ax0.spines.values():
        spine.set_visible(True)

fig.show()
fig.savefig("pics/pairwise_tele/pairfraction_{}_event{}_sig{}.png".format(datanm, direc, sig), dpi=300, bbox_inches='tight')
fig.savefig("pics/pairwise_tele/pairfraction_{}_event{}_sig{}.pdf".format(datanm, direc, sig), bbox_inches='tight')

# %% Plot bundle area fraction
M = NR
N = NR
CBLABELS = {"00": "Drought area fraction [-]", "11": "Pluvial area fraction [-]", 
            "01": "Area fraction [-]", "10": "Area fraction [-]"}
x = np.arange(M + 1) - 0.5
y = np.arange(N, -1, -1) - 0.5
xs, ys = np.meshgrid(x, y)

if True:  # direc[0] == direc[1]:
    triangles1 = [(i + j*(M+1), i+1 + j*(M+1), i + (j+1)*(M+1)) for j in range(N) for i in range(j+1)]  # upper tri
    triangles2 = [(i+1 + j*(M+1), i + (j+1)*(M+1), i+1 + (j+1)*(M+1)) for j in range(N) for i in range(j+1)]  # lower tri
    value1 = [fracsx[j, i] for j in range(N) for i in range(j+1)]  # j row -> rx; i col -> ry
    value2 = [fracsy[j, i] for j in range(N) for i in range(j+1)]
else:
    triangles1 = [(i + j*(M+1), i+1 + j*(M+1), i + (j+1)*(M+1)) for j in range(N) for i in range(M)]  # upper tri
    triangles2 = [(i+1 + j*(M+1), i + (j+1)*(M+1), i+1 + (j+1)*(M+1)) for j in range(N) for i in range(M)]  # lower tri
    # triangles1 = [(i + j*(M+1), i + (j+1)*(M+1), i+1 + (j+1)*(M+1)) for j in range(N) for i in range(M)]  # upper tri
    # triangles2 = [(i + j*(M+1), i+1 + j*(M+1), i+1 + (j+1)*(M+1)) for j in range(N) for i in range(M)]  # lower tri
    value1 = fracsx.ravel()
    value2 = fracsy.ravel()
triang1 = Triangulation(xs.ravel(), ys.ravel(), triangles1)
triang2 = Triangulation(xs.ravel(), ys.ravel(), triangles2)

# fig, ax = plt.subplots(figsize=(4.5, 4.5), tight_layout=True, subplot_kw={'aspect': 'equal'})
fig = plt.figure(figsize=(4.5, 3.8), tight_layout=True)
gs = fig.add_gridspec(1, 1, left=0.24, right=0.83, top=1, bottom=0.19, wspace=0.0)
ax = fig.add_subplot(gs[0], aspect='equal')
cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.03, ax.get_position().height])
cax2 = fig.add_axes([cax.get_position().x1, cax.get_position().y0, 0.03, cax.get_position().height])
colors0 = mpl.cm.RdPu if direc[0] == "0" else mpl.cm.GnBu
colors1 = mpl.cm.RdPu if direc[1] == "0" else mpl.cm.GnBu
img1 = ax.tripcolor(triang1, value1,
                    cmap=colors0, norm=mpl.colors.BoundaryNorm(np.linspace(0, 1, 11), ncolors=255))
img2 = ax.tripcolor(triang2, value2,
                    cmap=colors1, norm=mpl.colors.BoundaryNorm(np.linspace(0, 1, 11), ncolors=255))
# cbarax = make_axes_locatable(ax).append_axes("right", 0.25, pad=0.05, axes_class=plt.Axes)
# cbarax2 = make_axes_locatable(cbarax).append_axes("right", 0.25, pad=0.25, axes_class=plt.Axes)   这条路行不通
cb = fig.colorbar(img1, cax=cax2, fraction=0.05, pad=0., label=CBLABELS[direc])
if direc[0] == direc[1]:
    cax.axis('off')
else:
    cb2 = fig.colorbar(img2, cax=cax, fraction=0.05, pad=0.05)
    cb2.set_ticks([])
ax.set(xticks=x[:-1] + 0.5, yticks=y[-1:0:-1] + 0.5, 
       xticklabels=region_list, yticklabels=region_list[::-1]) # mind rx by rows
ax.set_xlabel(TYPEY, fontsize="large", fontweight='bold')
ax.set_ylabel(TYPEX, fontsize="large", fontweight='bold')
title = ax.set_title(TITLES[direc])
ax.set_xticklabels(region_list, rotation=45, ha="right", rotation_mode="anchor")
ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
ax.spines[['top', 'right']].set_visible(False)
if direc == "01":
    cax.set_visible(False)
    cax2.set_visible(False)
fig.patches.extend([Rectangle(RECTS[direc], 1, 1, fill=False, edgecolor='k', lw=3, zorder=15, transform=ax.transData)])
# ax.add_patch(Rectangle((4.5, -0.5), 1, 1, fill=False, edgecolor='k', lw=3, zorder=15)) # This cannot add on top of axes
ax.text(-0.39, 1.0, LABELS[direc], transform=title.get_transform(), fontweight='bold',
        fontsize='large', va='baseline', ha="left")
fig.savefig("pics/pairwise_tele/areafraction_{}_event{}_sig{}.pdf".format(datanm, direc, sig), bbox_inches='tight')
fig.show()
print()

# %%
