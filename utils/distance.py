from math import radians, sin, cos, acos
from numba import vectorize

@vectorize(['float64(float64, float64, float64, float64)'], target='parallel')
def haversine(lon1, lat1, lon2, lat2):
    lon1 = radians(lon1)
    lat1 = radians(lat1)
    lon2 = radians(lon2)
    lat2 = radians(lat2)
    # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  # 不要使用map function，内存不会释放
    dist = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)
    dist = 1 if dist > 1 else (-1 if dist < -1 else dist)
    return acos(dist)

def phaversine(lon0, lat0):
    import numpy as np
    dist_matrix = haversine(lon0, lat0, lon0.reshape(-1, 1), lat0.reshape(-1, 1))
    return dist_matrix[np.tril_indices_from(dist_matrix, k=-1)]