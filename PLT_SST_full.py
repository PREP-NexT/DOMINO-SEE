# %%
import numpy as np
from pandas import to_datetime
import pandas as pd
import time
from get_index_from_coord import get_index_for_square
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from matplotlib.font_manager import font_scalings
# font_scalings["larger"] = 1.3
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
pplt.rc["font.family"] = "Myriad Pro"
pplt.rc["font.largesize"] = "larger"
pplt.rc["figure.dpi"] = 200


datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
vp = np.load("0data/prcp_validpoint_annual_100.npy")
vp = vp.reshape(vp.size)
path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_temp'

sst = xr.open_dataset("0data/sst.mnmean.nc")
sst = sst.sel(time=slice("1950-01-01", "2016-12-31"))
# calculate sst anomaly by subtracting mean corresponding to that month
sstA = sst.groupby("time.month") - sst.groupby("time.month").mean(dim="time")

th = 1.5
sig = 0.005

tic = time.time()

# %% Regions
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


# Calculate P*P convolution of densx
def convolve_PxP(x, pad=1):  # Pad x Pad
    x = np.pad(x, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
    x = np.array([[x[i-pad:i+pad+1, j-pad:j+pad+1].sum() for j in range(pad, x.shape[1]-pad)] for i in range(pad, x.shape[0]-pad)]) / ((pad*2+1)**2)
    return x

# Find the maximum of convolved densx
def convolve_PxP_max(x, pad=1):
    x = convolve_PxP(x, pad)
    x = np.unravel_index(np.argmax(x), x.shape)
    p = np.meshgrid(np.arange(x[0]-pad, x[0]+pad+1), np.arange(x[1]-pad, x[1]+pad+1))
    return p

def eca_window(b, symdelt=2):
    window = np.ones(2 * symdelt + 1)
    if symdelt == 0:
        bw = b  #.copy()
    else:
        bw = np.apply_along_axis(lambda x: np.convolve(x, window)[symdelt:-symdelt] >= 0.5, 1, b)
    return bw

def eca_eventfinder(b1, b2, b1w, b2w):
    EVprec = b1 & b2w
    EVtrig = b2 & b1w
    return EVprec, EVtrig

def add_region_box(ax, region, edgecolor="tab:blue", facecolor="none", linewidth=2):
    c = regions[region]
    ax.add_patch(mpatches.Rectangle((c[1][0], c[0][0]),
                                  width=(c[1][1] - c[1][0]) if (c[1][1] - c[1][0]) > 0 else (c[1][1] - c[1][0] + 360),
                                  height=c[0][1] - c[0][0],
                                  ec=edgecolor, fc=facecolor, lw=linewidth, transform=ccrs.PlateCarree(), zorder=10))

def add_box(ax, lon, lat, edgecolor="tab:blue", facecolor="none", linewidth=1):
    ax.add_patch(mpatches.Rectangle((lon[0], lat[0]),
                                  width=lon[1] - lon[0],
                                  height=lat[1] - lat[0],
                                  ec=edgecolor, fc=facecolor, lw=linewidth, transform=ccrs.PlateCarree(), zorder=20))

# %%
C0 = {"0": "salmon", "1": "tab:cyan"}
RPS = {0: [{"rx": "East Africa", "ry": "India", "direction": "00", "clvl": (-0.5, 0.55, 0.05)},
           {"rx": "Australia", "ry": "South Africa", "direction": "11", "clvl": (-0.5, 0.55, 0.05)}], 
       1: [{"rx": "Mexico", "ry": "West US", "direction": "10", "clvl": (-0.5, 0.55, 0.05)},
           {"rx": "Canada", "ry": "Argentina", "direction": "01", "clvl": (-0.5, 0.55, 0.05)},]}
LABELS = {0: ["F", "G"], 1: ["D", "E"]}
SAVENMS = {0: "intra", 1: "inter"}

figcase = 1
fig, axs = pplt.subplots(ncols=1, nrows=2, figwidth=4.5, share=0, tight=True, projection=ccrs.PlateCarree(central_longitude=180))

rps = RPS[figcase]
for i, rp in enumerate(rps):
    ax = axs[i]
    rp = rps[i]
    clvl = rp["clvl"]
    rx, ry, direc = rp["rx"], rp["ry"], rp["direction"]
    fbundle = np.load("4bundle/bundle_{}-{}_{}_event{}_{}.npz".format(rx, ry, datanm, direc, th))
    print("Link Density is {:.4f}%".format(fbundle["link_bundle"].sum() / (indices[rx][2].size * indices[ry][2].size) * 100))
    lonx = indices[rx][1]
    latx = indices[rx][0]
    lony = indices[ry][1]
    laty = indices[ry][0]
    # fbundle.files
    densx, densy = fbundle["densx"], fbundle["densy"]

    # find the maximum P*P square in densx
    px = convolve_PxP_max(densx, pad=2)
    pxlatlon = np.array((latx[px[0]].ravel(), lonx[px[1]].ravel())).T
    # find the row numbers in latlon that are in pxlatlon
    idx = np.nonzero(np.all(latlon == pxlatlon[:,np.newaxis], axis=2))[1]

    # find the maximum P*P square in densy
    py = convolve_PxP_max(densy, pad=2)
    pylatlon = np.array((laty[py[0]].ravel(), lony[py[1]].ravel())).T
    # find the row numbers in latlon that are in pylatlon
    idy = np.nonzero(np.all(latlon == pylatlon[:,np.newaxis], axis=2))[1]

    infileX = "{}_glb_spi1_event_{}.npz".format(datanm, "drt{}".format(-th) if direc[0] == "0" else "fld{}".format(th))
    infileY = "{}_glb_spi1_event_{}.npz".format(datanm, "drt{}".format(-th) if direc[1] == "0" else "fld{}".format(th))
    evX = np.load("1event/{}".format(infileX))["ev"]
    evY = np.load("1event/{}".format(infileY))["ev"]
    evx = evX[idx, :]
    evy = evY[idy, :]

    # synthetic time series
    th_nev = 1
    evx0 = (evx.sum(axis=0) >= th_nev)[np.newaxis, :]
    evy0 = (evy.sum(axis=0) >= th_nev)[np.newaxis, :]

    evx0w = eca_window(evx0, symdelt=1)
    evy0w = eca_window(evy0, symdelt=1)

    ecax0, ecay0 = eca_eventfinder(evx0, evy0, evx0w, evy0w)
    ecat = ecax0 | ecay0
    evort = (evx0 | evy0) & ~ecat

    lag = 0
    sstandm = sstA.sel(time=ddate[np.where(ecat)[1] - lag]).mean(dim="time")
    sstan3 = sstandm.sst
    ms = ax.pcolormesh(sstan3.lon, sstan3.lat, sstan3, transform=ccrs.PlateCarree(), 
                       levels=np.arange(*clvl), extend="both", zorder=0, rasterized=True)

    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ttl = ax.set_title("Both {} {} and {} {}".format(rx, "drought" if direc[0] == "0" else "pluvial", 
                                                    ry, "drought" if direc[1] == "0" else "pluvial", lag))
    _ = ax.add_feature(cfeature.LAND, facecolor=[0.8, 0.8, 0.8], edgecolor="black", zorder=5, lw=0.5)
    add_region_box(ax, rx, edgecolor=C0[direc[0]])
    add_region_box(ax, ry, edgecolor=C0[direc[1]])
    gl = ax.gridlines(draw_labels=["left", "bottom"], linestyle=":", linewidth=0.3, color='k', zorder=10)
    add_box(ax, (pxlatlon[:, 1].min(), pxlatlon[:, 1].max()), (pxlatlon[:, 0].min(), pxlatlon[:, 0].max()), edgecolor="r", facecolor="r")
    add_box(ax, (pylatlon[:, 1].min(), pylatlon[:, 1].max()), (pylatlon[:, 0].min(), pylatlon[:, 0].max()), edgecolor="r", facecolor="r")
    fig.text(-0.09, 1, LABELS[figcase][i], va="baseline", ha="left", fontsize="large", fontweight="bold", transform=ttl.get_transform())

fig.colorbar(ms, label="SST anomaly [K]", loc="r", width=0.1, rows=(1, 2), ticks=np.arange(*clvl)[::2], extend="both")
fig.savefig("pics/sst/sstAand_{}_lag{}.pdf".format(SAVENMS[figcase], lag), dpi=300, bbox_inches="tight")
