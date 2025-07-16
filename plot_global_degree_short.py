"""
This script is used to plot network degrees for links <= 2500km
"""
# %%
import numpy as np
import scipy.sparse as sp
from pandas import to_datetime
import matplotlib as mpl
import matplotlib.colors as mcolors
import ultraplot as pplt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as ticker  # Format Cartesian to Geo
import colormaps as cmaps
pplt.rc["font.sans-serif"] = "Myriad Pro"
pplt.rc["savefig.dpi"] = 300
pplt.rc["font.largesize"] = "large"
# pplt.rc["font.size"] = 12

datanm = "spimv2"
lon = np.load('0data/{}_lon.npy'.format(datanm))
lat = np.load('0data/{}_lat.npy'.format(datanm))
latlon = np.load('0data/{}_latlon.npy'.format(datanm))
cos_lat = np.cos(latlon[:, 0] * np.pi / 180).reshape(lat.size, lon.size)
ddate = to_datetime(np.load('0data/{}_date.npy'.format(datanm)))
vp = np.load("0data/prcp_validpoint_annual_100.npy")
path = ''

distth = 2500

cmdry = mcolors.ListedColormap(cmaps.wind_17lev([1, 2, 7, 8, 10, 12, 13, 14]))
cmdry.set_under(cmaps.wind_17lev(0))
cmdry.set_over(cmaps.wind_17lev(17))
cmwet = mcolors.ListedColormap(cmaps.precip_11lev([1, 3, 4, 5, 7, 8, 9, 10]))
cmwet.set_under(cmaps.precip_11lev(0))
cmwet.set_over(cmaps.precip_11lev(11))
# nmclr0 = "tab:red" if direc[0] == "0" else "tab:blue"
# nmclr1 = "tab:red" if direc[1] == "0" else "tab:blue"
# color0 = cmdry if direc[0] == "0" else cmwet
# color1 = cmdry if direc[1] == "0" else cmwet

titles = {"00": "(drought–drought network)", "11": "(pluvial–pluvial network)", "01": "(drought–pluvial network)"}

th = 1.5
# direc = "11"
sig = 0.005

# %% Version 3
def plot_joint_degree(ax, degree, cmap=mpl.cm.YlOrRd):
    # fig = pplt.figure(figwidth=7)
    # ax = fig.subplot(projection=ccrs.PlateCarree())
    ax.format(latlim=(lat.min(), lat.max()))
    ax.coastlines()
    # ax.add_feature(cfeature.BORDERS, ls=":", lw=0.5, facecolor='None', edgecolor='black')
    gl = ax.gridlines(draw_labels=False, linestyle=":", linewidth=0.3, color='k')

    bounds = np.logspace(np.log10(np.percentile(degree[degree > 0], 0.1)), 4, num=cmap.N-1)
    if degree.max() > 10000:
        bounds = np.append(np.insert(bounds, 0, 0), 20000).astype('int')
    else:
        bounds = np.append(np.insert(bounds, 0, 0), 20000).astype('int')
    ax.set_facecolor("lightgray")
    degree = np.ma.masked_array(degree, ~vp)
    cs = ax.pcolormesh(lon, lat, degree, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N), extend="both", rasterized=True)
    # cbar = fig.colorbar(cs, label="Number of Links", location='left')
    # cbar.set_ticklabels(list(bounds[:-1]) + [" "])
    # ax.contourf(lon, lat, (~vp).astype('int'), transform=ccrs.PlateCarree(),
    #             colors='none', levels=[.5, 1.5], hatches=['//////'])
    pxr = ax.panel('r', share=False)
    deglat = np.mean(degree, axis=1)  # / latlon.shape[0]
    pxr.plot(deglat, lat, 'k')
    pxr.format(ylim=(lat.min(), lat.max()), xlim=(deglat.max(), deglat.min()), # set xlim to fix axis direction
               yticks=gl.ylocator, yticklabels=ticker.LatitudeFormatter(),
               xrotation=90)
    # pxr.format(ylim=(lat.min(), lat.max()), xlim=(6000, 0),  # set xlim to fix axis direction
    #            xticks=[0, 2000, 4000, 6000],
    #            yticks=gl.ylocator, yticklabels=ticker.LatitudeFormatter(),
    #            xrotation=90)
    # pxb = ax.panel('b')
    # lon_new = lon.copy()
    # lon_new[lon_new >= 180] -= 360
    # lon_new = np.roll(lon_new, (lon_new < 0).sum())
    # deglon = np.roll(np.mean(degree, axis=0), (lon_new < 0).sum()) #/ latlon.shape[0]
    # pxb.plot(lon_new, deglon, 'k')
    # pxb.format(xlim=(lon_new.min(), 180), ylim=(0, 4500),# fix 180 in x axis
    #            xticks=gl.xlocator, xticklabels=ticker.LongitudeFormatter(),
    #            yticks=[0, 2000, 4000])
    return {'r': pxr, }, cs  #'b': pxb 


DIREC = {0: "00", 1: "11", 2: "01", 3: "01"}
CMAP = {0: cmdry, 1: cmwet, 2: cmdry, 3: cmwet}
TITLE_PRE = {0: "Drought layer ", 1: "Pluvial layer ", 2: "Drought layer ", 3: "Pluvial layer "}

# %% Short-distance
fig = pplt.figure(figwidth=9, dpi=330, share=False)
ax = fig.subplots(ncols=2, nrows=2, projection=ccrs.PlateCarree())
ax.format(abc="A", abcloc="l")
for nax in range(len(ax)):
    direc = DIREC[nax]
    lnk_shr = sp.load_npz("{}3link/linkshr{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))
    print("Short Link Fraction: {:.2f}%".format(lnk_shr.size / (vp.size ** 2) * 100))

    degree_shr0 = np.array(lnk_shr.sum(axis=0).reshape(lat.size, lon.size))
    degree_shr1 = np.array(lnk_shr.sum(axis=1).reshape(lat.size, lon.size))

    cmap = CMAP[nax]
    if nax == 0:
        px, cs = plot_joint_degree(ax[nax], degree_shr0, cmap=cmap)
        cbar = ax[nax].colorbar(cs, location='bottom', orientation='horizontal', width=0.15, 
                                label="No. of short-distance links [-]", aspect=40)
    elif nax == 1:
        px, cs = plot_joint_degree(ax[nax], degree_shr1, cmap=cmap)
        cbar = ax[nax].colorbar(cs, location='bottom', orientation='horizontal', width=0.15,
                                label="No. of short-distance links [-]", aspect=40)
    elif nax == 2:
        px, cs = plot_joint_degree(ax[nax], degree_shr0, cmap=cmap)
        cbar = ax[nax].colorbar(cs, location='bottom', orientation='horizontal', width=0.15, 
                                label="No. of short-distance links [-]", aspect=40)
    elif nax == 3:
        px, cs = plot_joint_degree(ax[nax], degree_shr1, cmap=cmap)
        cbar = ax[nax].colorbar(cs, location='bottom', orientation='horizontal', width=0.15, 
                                label="No. of short-distance links [-]", aspect=40)
    ax[nax].format(title=TITLE_PRE[nax] + titles[direc])
    # if direc[0] == "0":
    #     fig, ax, px = plot_joint_degree(degree_shr0, cmap=color0)
    #     ax.format(title="Drought Layer, " + titles[direc])
    #     fig.show()
    #     fig.savefig("pics/dist/glbdegree{}_{}_event{}_{}_stat-shr.png".format(0, datanm, direc, th), dpi=300, bbox_inches='tight')
    # if direc[1] == "1":
    #     fig, ax, px = plot_joint_degree(degree_shr1, cmap=color1)
    #     ax.format(title="Pluvial Layer, " + titles[direc])
    #     fig.show()
    #     fig.savefig("pics/dist/glbdegree{}_{}_event{}_{}_stat-shr.png".format(1, datanm, direc, th), dpi=300, bbox_inches='tight')
fig.savefig("pics/dist/glbdegree_{}_{}_stat-shr.png".format(datanm, th), bbox_inches='tight')
fig.savefig("pics/dist/glbdegree_{}_{}_stat-shr.pdf".format(datanm, th), bbox_inches='tight')
print()

