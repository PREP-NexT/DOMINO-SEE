#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
"""
@Create   : 30/3/23 9:09 PM
@Author   : WANG HUI-MIN
@Update   : This is to separate the bundle calculation from the bundle plots
"""
import numpy as np
from utils.get_index_from_coord import get_index_for_square
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from util.plot import truncate_colormap
mpl.rcParams['font.family'] = 'Myriad Pro'
mpl.rcParams['font.size'] = 9
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["axes.titlesize"] = "large"
mpl.rcParams["xtick.labelsize"] = "medium"
mpl.rcParams["ytick.labelsize"] = "medium"

datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_temp'

th = 1.5
sig = 0.005

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
indices = {i: get_index_for_square(lat, lon, *c) for (i, c) in regions.items()}

# %% Plot functions
def plot_region(regions, rx, ry, figsize, colorbar_position="right"):
    # fig = pplt.figure(figsize=figsize)
    centx = np.mean(regions[rx][1]) if regions[rx][1][1] > regions[rx][1][0] else (np.mean(regions[rx][1]) - 180)
    centy = np.mean(regions[ry][1]) if regions[ry][1][1] > regions[ry][1][0] else (np.mean(regions[ry][1]) - 180)
    cent = np.mean([centx, centy]) if centy - centx <= 180 else (np.mean([centx, centy]) + 180)
    fig = plt.figure(figsize=figsize, tight_layout=True)
    if colorbar_position == "right":
        gs = fig.add_gridspec(1, 4, left=0.05, right=0.925, top=0.95, bottom=0.05, wspace=0.0, width_ratios=[1, 0.08, 0.045, 0.045])
    elif colorbar_position == "bottom":
        gs = fig.add_gridspec(4, 1, left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.0, height_ratios=[1, 0.05, 0.05, 0.05])
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=cent))
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    for ax0 in [ax2, ax3]:
        pos = ax0.get_position()  # Get the current position as a Bbox
        if colorbar_position == "right":
            ax0.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        elif colorbar_position == "bottom":
            ax0.set_position([pos.x0, pos.y0, pos.width * 0.85, pos.height])

    # ax = fig.subplot(projection=ccrs.PlateCarree(central_longitude=cent))
    # ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cent))
    ax.add_feature(cfeature.COASTLINE, lw=0.5, facecolor="none", edgecolor="0.5")
    ax.add_feature(cfeature.LAND, color="0.82")
    ax.set_facecolor("0.96")
    # ax.add_feature(cfeature.BORDERS, ls=":", lw=0.5, facecolor='None', edgecolor='black')
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.3, color='k')
    gl.right_labels = False
    gl.top_labels = False
    return fig, ax


def add_region_box(ax, region, edgecolor="tab:blue"):
    c = regions[region]
    ax.add_patch(mpatches.Rectangle((c[1][0], c[0][0]),
                                  width=(c[1][1] - c[1][0]) if (c[1][1] - c[1][0]) > 0 else (c[1][1] - c[1][0] + 360),
                                  height=c[0][1] - c[0][0],
                                  ec=edgecolor, fc='none', lw=2, transform=ccrs.PlateCarree(), zorder=10))


def plot_contour(lon, lat, data, axfunc, **kwargs):
    lon0 = lon.copy()
    data0 = data.copy()
    lon0delta = np.diff(lon0)
    if not np.allclose(lon0delta, lon0delta[0]):  # cross meridian
        id = np.where(~np.isclose(lon0delta, lon0delta[0]))[0]
        lon0 = np.roll(lon0, -id-1)
        data0 = np.roll(data0, -id-1, axis=1)
    cs = axfunc(np.mod(lon0 - 180.0, 360.0) - 180.0, lat, data0, **kwargs)
    return cs


def plot_contour_pad(lon, lat, data, axfunc, pad=1, **kwargs):
    lon0 = lon.copy()
    data0 = data.copy()
    lon0delta = np.diff(lon0)
    if not np.allclose(lon0delta, lon0delta[0]):  # cross meridian
        id = np.where(~np.isclose(lon0delta, lon0delta[0]))[0]
        lon0 = np.roll(lon0, -id-1)
        data0 = np.roll(data0, -id-1, axis=1)
    lon0 = np.mod(lon0 - 180.0, 360.0) - 180.0
    lat0 = lat.copy()
    # pad the data to make boundary part of the countour
    lon0 = np.concatenate(([lon0[0] - (lon0[1] - lon0[0])], lon0, [lon0[-1] + (lon0[1] - lon0[0])]))
    lat0 = np.concatenate(([lat0[0] - (lat0[1] - lat0[0])], lat0, [lat0[-1] + (lat0[1] - lat0[0])]))
    data0 = np.pad(data0, pad, mode='constant', constant_values=np.False_)
    cs = axfunc(lon0, lat0, data0, **kwargs)
    return cs

CMRED = truncate_colormap(mpl.cm.Reds, 0.1)
CMRED.set_under('none')
CMBLUE = truncate_colormap(mpl.cm.Blues, 0.1)
CMBLUE.set_under('none')
LC = {"00": "tab:red", "01": "grey", "10": "grey", "11": "tab:blue"}

# %% Region 1 and 2
figcase = 1
cblc = "right"
ax_kw = {"xmargin": 0.05}
line_kw = {"lw": 0.04}
label, offset = "", 0
if figcase == 0:
    rps = [{"rx": "East Africa", "ry": "India", "direction": "00"},
           {"rx": "Australia", "ry": "South Africa", "direction": "11"},
           ]
    fs = (4.5, 3.9)
    rframe = "South Africa", "India" 
    extent = (11.25, 159.75, -62.93, 33.37)
    title = "Intra-layer synchronized hotspots"
    savenm = "within"
    label, offset = "E", -0.08
    cblc = "bottom"
    cblabel = None
elif figcase == 1:
    rps = [{"rx": "Canada", "ry": "Argentina", "direction": "01"},
           {"rx": "Mexico", "ry": "West US", "direction": "10"},
           ]
    fs = (4.5, 3.75)
    rframe = "Canada", "Argentina"
    extent = (-133.62, -39.38, -40.88, 56.88)
    title = "Inter-layer synchronized hotspots"
    savenm = "cross"
    label, offset = "C", -0.1
    ax_kw = {"xmargin": 0.12, "ymargin": 0.02}
    line_kw = {"lw": 0.12}
    cblabel = None
else:
    raise Exception("Wrong figcase")


fig, ax = plot_region(regions, rframe[0], rframe[1], figsize=fs, colorbar_position=cblc)
normmax = 0
for rp in rps:
    rx, ry, direc = rp["rx"], rp["ry"], rp["direction"]
    fbundle = np.load("4bundle/bundle_{}-{}_{}_event{}_{}.npz".format(rx, ry, datanm, direc, th))
    print("Link Density is {:.4f}%".format(fbundle["link_bundle"].sum() / (indices[rx][2].size * indices[ry][2].size) * 100))
    bundlex, bundley = fbundle["bundlex"], fbundle["bundley"]
    link_bundle = fbundle["link_bundle"]
    lnk_frcx = (link_bundle.sum(axis=1) / bundley.sum()).reshape(bundlex.shape)
    lnk_frcx[~bundlex] = np.nan
    lnk_frcy = (link_bundle.sum(axis=0) / bundlex.sum()).reshape(bundley.shape)
    lnk_frcy[~bundley] = np.nan
    # temp plot of lnk_frc density
    # fig1, ax1 = plt.subplots(1, 2, figsize=(6, 2.5))
    # ax1[0].hist(lnk_frcx[~np.isnan(lnk_frcx)], bins=20)
    # ax1[1].hist(lnk_frcy[~np.isnan(lnk_frcy)], bins=20)

    normmax_temp = np.fmax(np.nanmax(lnk_frcx), np.nanmax(lnk_frcy))
    print("{}-{} fraction max = {} from {} {}".format(rx, ry, normmax_temp, np.nanmax(lnk_frcx), np.nanmax(lnk_frcy)))
    normmax = normmax_temp if normmax_temp > normmax else normmax

for rp in rps:
    rx, ry, direc = rp["rx"], rp["ry"], rp["direction"]
    fbundle = np.load("4bundle/bundle_{}-{}_{}_event{}_{}.npz".format(rx, ry, datanm, direc, th))
    print("Link Density is {:.4f}%".format(fbundle["link_bundle"].sum() / (indices[rx][2].size * indices[ry][2].size) * 100))

    cm0 = CMRED if direc[0] == "0" else CMBLUE
    cm1 = CMRED if direc[1] == "0" else CMBLUE
    c0 = "salmon" if direc[0] == "0" else "tab:cyan"
    c1 = "salmon" if direc[1] == "0" else "tab:cyan"

    bundlex, bundley = fbundle["bundlex"], fbundle["bundley"]
    add_region_box(ax, rx, edgecolor=c0)
    add_region_box(ax, ry, edgecolor=c1)

    plot_contour(indices[rx][1], indices[rx][0], bundlex, ax.contour, transform=ccrs.PlateCarree(), colors=["black"],
                 zorder=15, levels=[.5, 1.5], linewidths=1)
    plot_contour(indices[ry][1], indices[ry][0], bundley, ax.contour, transform=ccrs.PlateCarree(), colors=["black"],
                 zorder=15, levels=[.5, 1.5], linewidths=1)


    link_bundle = fbundle["link_bundle"]
    print("Link number is {}".format(link_bundle.sum()))
    lnk_frcx = (link_bundle.sum(axis=1) / bundley.sum()).reshape(bundlex.shape)
    lnk_frcx[~bundlex] = np.nan
    lnk_frcy = (link_bundle.sum(axis=0) / bundlex.sum()).reshape(bundley.shape)
    lnk_frcy[~bundley] = np.nan

    norm = mpl.colors.BoundaryNorm(mpl.ticker.MaxNLocator(8).tick_values(0, normmax/3.5), extend="max", ncolors=256)
    cs0 = plot_contour(indices[rx][1], indices[rx][0], lnk_frcx, ax.pcolormesh, transform=ccrs.PlateCarree(), cmap=cm0,
                       norm=norm)  # , shading="nearest"
    cs1 = plot_contour(indices[ry][1], indices[ry][0], lnk_frcy, ax.pcolormesh, transform=ccrs.PlateCarree(), cmap=cm1,
                       norm=norm)  # , shading="nearest"

    edgex = indices[rx][2][np.where(link_bundle)[0]]
    edgey = indices[ry][2][np.where(link_bundle)[1]]
    for e in range(0, link_bundle.sum(), 25):
        ax.plot(latlon[[edgex[e], edgey[e]], 1], latlon[[edgex[e], edgey[e]], 0],
                LC[direc], alpha=0.075, **line_kw, transform=ccrs.Geodetic(), rasterized=True) #'darkgrey'

ttl = ax.set_title(title)
# ax.set(**ax_kw)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.text(offset, 1, label, transform=ttl.get_transform(), fontweight='bold',
        fontsize='large', va='baseline', ha="left")
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMRED),
             cax=fig.axes[1], label=None, ticks=mpl.ticker.NullLocator(), orientation='vertical' if cblc == "right" else 'horizontal')
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMBLUE),
             cax=fig.axes[2], orientation='vertical' if cblc == "right" else 'horizontal', label=cblabel) 

fig.savefig("pics/pairwise_tele/bundlekde_{}_{}_areafrac.pdf".format(datanm, savenm), bbox_inches='tight')
fig.savefig("pics/pairwise_tele/bundlekde_{}_{}_areafrac.png".format(datanm, savenm), bbox_inches='tight', dpi=330)

print()

# %%
