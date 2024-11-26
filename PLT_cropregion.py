#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
"""
@Create   : 3/12/22 9:26 pm
@Author   : WANG HUI-MIN
"""
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray as rxr
pplt.rc["font.family"] = "Myriad Pro"
pplt.rc["font.largesize"] = "large"
pplt.rc["figure.dpi"] = 300

def plot_base(ax, draw_labels=True, **kwargs):
    # fig = plt.figure(figsize=figsize, tight_layout=True)
    # ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines(zorder=5)
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor="none", zorder=-1)
    gl = ax.gridlines(draw_labels=draw_labels, linestyle=":", linewidth=0.3, color='k', **kwargs)
    gl.right_labels = False
    gl.top_labels = False
    # if draw_labels is True: # Only maintain left and bottom
    #     for geolabel in gl.geo_label_artists:
    #         if geolabel._x > 0:
    #             geolabel.set_visible(False)
    # ax.add_feature(cfeature.BORDERS, facecolor='None', edgecolor='black', ls='--')
    # ax.set_extent([-180, 180, lat.min(), lat.max()], crs=ccrs.PlateCarree())
    return fig, ax


crop = rxr.open_rasterio("0data/Crop/gl-croplands-geotif/cropland.tif")
past = rxr.open_rasterio("0data/Crop/gl-pastures-geotif/pasture.tif")
regions = {"China_North": ([33.25, 43.0], [102.5, 121.0]),
           "China_South": ([20.0, 33.0], [102.0, 122.5]),
           "Europe_East": ([46.25, 55.5], [18.25, 58.0]),
           "Europe_West": ([40.0, 56.0], [355.0, 18.0]),
           "Mediterranean": ([32.0, 46.0], [20.25, 44.0]),
           "Africa_East": ([-3.0, 16.5], [23.5, 41.5]),
           "Africa_South": ([-35.0, -15.0], [18.0, 35.0]),
           "Argentina": ([-39.0, -24.5], [295.0, 311.5]),
           "Canada": ([46.25, 55.0], [246.5, 264.0]),
           "US_East": ([30.25, 46.0], [261.0, 275.0]),
           "US_West": ([35.0, 48.0], [235.5, 246.0]),
           "Mexico": ([16.0, 30.0], [255.0, 266.0]),
           "Australia": ([-38.0, -26.0], [137.5, 153.0]),
           "India": ([8.0, 29.0], [70.0, 88.0])}
region_list = list(regions.keys())

cropland = crop.data[0].copy()
cropland[cropland <= 0.0] = np.nan
cropland *= 100.0

fig, axs = pplt.subplots(ncols=2, nrows=1, share=0, figwidth=9, abc="A", projection=ccrs.Robinson())
ax = axs[0]
plot_base(ax, draw_labels=False, ylocs=np.arange(-60, 90, 20))
ax.set_extent([-180, 180, -60, 90], ccrs.PlateCarree())
ax.set_xlim(ax.projection.x_limits)
img = ax.pcolormesh(crop.x.data, crop.y.data, cropland, cmap=mpl.cm.Oranges, norm=mpl.colors.Normalize(vmin=0, vmax=100),
                    transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
ax.set(title="Cropland fraction", facecolor="0.9")
Rect = {i: mpatches.Rectangle((c[1][0], c[0][0]),
                              width=(c[1][1] - c[1][0]) if (c[1][1] - c[1][0]) > 0 else (c[1][1] - c[1][0] + 360),
                              height=c[0][1] - c[0][0],
                              ec='magenta', fc='none', lw=2, transform=ccrs.PlateCarree())
        for (i, c) in regions.items()}
for (i, rec) in Rect.items():
    ax.add_patch(rec)
ax.colorbar(img, loc="b", format="%.0f%%", length=0.65, width=0.1)

# %%
pastland = past.data[0].copy()
pastland[pastland <= 0.0] = np.nan
pastland *= 100.0

ax = axs[1]
plot_base(ax, draw_labels=False, ylocs=np.arange(-60, 90, 20))
# fig, ax = plot_base(figsize=(7, 4), draw_labels=False, ylocs=np.arange(-60, 90, 20))
ax.set_extent([-180, 180, -60, 90], ccrs.PlateCarree())
ax.set_xlim(ax.projection.x_limits)
img = ax.pcolormesh(past.x.data, past.y.data, pastland, cmap=mpl.cm.Greens, norm=mpl.colors.Normalize(vmin=0),
                    transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
ax.set(title="Pastureland fraction", facecolor="0.9")
Rect = {i: mpatches.Rectangle((c[1][0], c[0][0]),
                              width=(c[1][1] - c[1][0]) if (c[1][1] - c[1][0]) > 0 else (c[1][1] - c[1][0] + 360),
                              height=c[0][1] - c[0][0],
                              ec='magenta', fc='none', lw=2, transform=ccrs.PlateCarree())
        for (i, c) in regions.items()}
for (i, rec) in Rect.items():
    ax.add_patch(rec)

ax.colorbar(img, loc="b", format="%.0f%%", length=0.65, width=0.1)
fig.savefig("pics/agriregion.pdf", dpi=300, bbox_inches='tight')

print()
# %%
