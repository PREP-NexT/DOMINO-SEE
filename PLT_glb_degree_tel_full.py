"""
This script is used to plot network degrees for links > 2500km
"""
# %%
import numpy as np
from pandas import to_datetime
import matplotlib as mpl
import matplotlib.colors as mcolors
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as ticker  # Format Cartesian to Geo
import colormaps as cmaps
pplt.rc["font.sans-serif"] = "Myriad Pro"
pplt.rc["font.largesize"] = "large"
pplt.rc["axes.labelsize"] = "med-large"
pplt.rc["savefig.dpi"] = 300
# pplt.rc["font.size"] = 12

datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
cos_lat = np.cos(latlon[:, 0] * np.pi / 180).reshape(lat.size, lon.size)
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
vp = np.load("0data/prcp_validpoint_annual_100.npy")
path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_temp'

cmdry = mcolors.ListedColormap(cmaps.wind_17lev([1, 2, 7, 8, 10, 12, 13, 14]))
cmdry.set_under(cmaps.wind_17lev(0))
cmdry.set_over(cmaps.wind_17lev(17))
cmwet = mcolors.ListedColormap(cmaps.precip_11lev([1, 3, 4, 5, 7, 8, 9, 10]))
cmwet.set_under(cmaps.precip_11lev(0))
cmwet.set_over(cmaps.precip_11lev(11))

titles = {"00": "(drought–drought network)", "11": "(pluvial–pluvial network)", "01": "(drought–pluvial network)"}

th = 1.5
sig = 0.005

# Version 3
def plot_joint_degree(ax, degree, cmap=mpl.cm.YlOrRd):
    ax.format(latlim=(lat.min(), lat.max()))
    ax.coastlines()
    # ax.add_feature(cfeature.BORDERS, ls=":", lw=0.5, facecolor='None', edgecolor='black')
    gl = ax.gridlines(draw_labels=False, linestyle=":", linewidth=0.3, color='k')

    bounds = np.array([150, 300, 500, 800, 1300, 2200, 3700, 6000, 10000]).astype('int')

    ax.set_facecolor("lightgray")
    degree = np.ma.masked_array(degree, ~vp)
    cs = ax.pcolormesh(lon, lat, degree, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N), extend="both", rasterized=True)

    pxr = ax.panel('r')
    deglat = np.mean(degree, axis=1)  # / latlon.shape[0]
    pxr.plot(deglat, lat, 'k')
    pxr.format(ylim=(lat.min(), lat.max()), xlim=(6000, 0),  # set xlim to fix axis direction
               xticks=[0, 2000, 4000, 6000],
               yticks=gl.ylocator, yticklabels=ticker.LatitudeFormatter(),
               xrotation=90)

    return {'r': pxr, }, cs  #'b': pxb 

DIREC = {0: "00", 1: "11", 2: "01", 3: "01"}
CMAP = {0: cmdry, 1: cmwet, 2: cmdry, 3: cmwet}
TITLE_PRE = {0: "Drought layer ", 1: "Pluvial layer ", 2: "Drought layer ", 3: "Pluvial layer "}

# %% Teleconnection
fig = pplt.figure(figwidth=9, dpi=330)
ax = fig.subplots(ncols=2, nrows=2, projection=ccrs.PlateCarree())
ax.format(abc="A", abcloc="l")
for nax in range(len(ax)):
    direc = DIREC[nax]
    degree_tel0 = np.load("{}/3link/linkdegtel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))["degree0"]
    degree_tel1 = np.load("{}/3link/linkdegtel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))["degree1"]
    # np.savez("{}/3link/linkdeg{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th),
    #          degree0=degree0, degree1=degree1)
    # np.savez("{}/3link/linkdegtel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th),
    #          degree0=degree_tel0, degree1=degree_tel1)
    cmap = CMAP[nax]
    if nax == 0:
        px, cs = plot_joint_degree(ax[nax], degree_tel0, cmap=cmap)
    elif nax == 1:
        px, cs = plot_joint_degree(ax[nax], degree_tel1, cmap=cmap)
    elif nax == 2:
        px, cs = plot_joint_degree(ax[nax], degree_tel0, cmap=cmap)
        cbar = ax[nax].colorbar(cs, location='bottom', orientation='horizontal', width=0.15, 
        label="No. of teleconnection links [-]", aspect=40)
    elif nax == 3:
        px, cs = plot_joint_degree(ax[nax], degree_tel1, cmap=cmap)
        cbar = ax[nax].colorbar(cs, location='bottom', orientation='horizontal', width=0.15, 
        label="No. of teleconnection links [-]", aspect=40)
    ax[nax].format(title=TITLE_PRE[nax] + titles[direc])

# %%
if True:
    fig.savefig("pics/dist/glbdegree_{}_{}_stat-tel.png".format(datanm, th), bbox_inches='tight')
    fig.savefig("pics/dist/glbdegree_{}_{}_stat-tel.pdf".format(datanm, th), bbox_inches='tight')
