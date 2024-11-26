import numpy as np
from itertools import product


def get_index_from_coord(lats, lons, lat_t, lon_t):  # 找出离target最近点的序号
    lat = lats[np.argmin(np.abs(lats - lat_t))]
    lon = lons[np.argmin(np.abs(lons - lon_t))]
    coords = np.array(list(product(lats, lons)))
    idx = np.intersect1d(np.where(coords[:, 0] == lat), np.where(coords[:, 1] == lon))
    return idx


class IndexOfRegion:
    def __init__(self, lat, lon):
        index = []
        for lat_t in np.linspace(1.375 - 0.5, 1.375 + 0.5, 5):  # [27.5, 27.75, 28, 28.25, 28.5]:
            for lon_t in np.linspace(103.875 - 0.5, 103.875 + 0.5, 5):  # [78.5, 78.75, 79., 79.25, 79.5]:
                index = np.concatenate((index, get_index_from_coord(lat, lon, lat_t, lon_t)))
        index = np.unique(index).astype('int32')
        self.index = {'SG': index}

        index = []
        for lat_t in [27.5, 27.75, 28., 28.25, 28.5]:
            for lon_t in [78.5, 78.75, 79., 79.25, 79.5]:
                index = np.concatenate((index, get_index_from_coord(lat, lon, lat_t, lon_t)))
        index = np.unique(index).astype('int32')
        self.index['NISM'] = index


def get_index_for_point(lats, lons, lat_r, lon_r, width):  # point 周围的点
    coords = np.array(list(product(lats, lons)))
    idx0 = get_index_from_coord(lats, lons, lat_r, lon_r)
    idxlat = np.where(lats == coords[idx0, 0])[0]
    idxlon = np.where(lons == coords[idx0, 1])[0]
    index = np.where((coords[:, 0] >= lats[idxlat - width]) & (coords[:, 0] <= lats[idxlat + width]) &
                     (coords[:, 1] >= lons[idxlon - width]) & (coords[:, 1] <= lons[idxlon + width]))[0]
    return index


def get_index_for_square(lats, lons, lat_r, lon_r):
    coords = np.array(list(product(lats, lons)))
    lat = lats[(lats >= lat_r[0]) & (lats <= lat_r[1])]
    if lon_r[0] >= lon_r[1]:  # Cross the meridian
        lon = lons[(lons >= lon_r[0]) | (lons <= lon_r[1])]
        index = np.where(((coords[:, 0] >= lat_r[0]) & (coords[:, 0] <= lat_r[1])) &
                         ((coords[:, 1] >= lon_r[0]) | (coords[:, 1] <= lon_r[1])))[0]
    else:
        lon = lons[(lons >= lon_r[0]) & (lons <= lon_r[1])]
        index = np.where((coords[:, 0] >= lat_r[0]) & (coords[:, 0] <= lat_r[1]) &
                         (coords[:, 1] >= lon_r[0]) & (coords[:, 1] <= lon_r[1]))[0]
    coord = coords[index]
    return lat, lon, index, coord


class IndexOfSquare:
    def __init__(self, lat, lon, lat_r, lon_r, name):
        self.lat, self.lon, self.index, self.coord = get_index_for_square(lat, lon, lat_r, lon_r)
        self.name = name


class AsiaSquare:
    def __init__(self, lat, lon):
        self.lat, self.lon, self.index, self.coord = get_index_for_square(lat, lon, [-10, 50], [70, 140])
        self.name = "Asia"


class ChinaSquare:
    def __init__(self, lat, lon):
        self.lat, self.lon, self.index, self.coord = get_index_for_square(lat, lon, [15, 50], [95, 135])
        self.name = "China"


class IndiaSquare:
    def __init__(self, lat, lon):
        self.lat, self.lon, self.index, self.coord = get_index_for_square(lat, lon, [5, 35], [65, 95])
        self.name = "India"


class EquatorSquare:
    def __init__(self, lat, lon):
        self.lat, self.lon, self.index, self.coord = get_index_for_square(lat, lon, [-20, 20], [90, 120])
        self.name = "Equator"



    # def __init__(self, lat, lon):
    #     coords = np.array(list(product(lat, lon)))
    #     self.lat = lat[(lat >= -10) & (lat <= 50)]
    #     self.lon = lon[(lon >= 70) & (lon <= 140)]
    #     self.index = np.where((coords[:, 0] >= -10) & (coords[:, 0] <= 50) &
    #                           (coords[:, 1] >= 70) & (coords[:, 1] <= 140))[0]


