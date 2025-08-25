"""Equidistant points on a sphere.

Fibbonachi Spiral:
https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html

Fekete points:
https://arxiv.org/pdf/0808.1202.pdf

Geodesic grid: (sec. 3.2)
https://arxiv.org/pdf/1711.05618.pdf

Review on geodesic grids:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.997&rep=rep1&type=pdf

"""
# Original work Copyright (C) 2022 Felix Strnad <felix.strnad@uni-tuebingen.de>
# From: https://github.com/mlcs/climnet/blob/main/climnet/grid/grid.py
# Modifications Copyright (C) 2025 Hui-Min Wang <wanghuimin@u.nus.edu>
# 
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import math
import os
import pickle
import sys
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.transform import Rotation as Rot

from .fekete import bendito, points_on_sphere

class BaseGrid(ABC):
    """Abstract base class for geographical grids.
    
    This class provides the common interface and functionality for different
    types of geographical grids.
    
    Attributes:
        grid: Dictionary containing grid coordinates with 'lat' and 'lon' keys
    """
    def __init__(self) -> None:
        self.grid: Optional[Dict[str, np.ndarray]] = None

    @abstractmethod
    def get_distance_equator(self) -> float:
        """Return distance between points at the equator in km."""
        pass

    @abstractmethod
    def create_grid(self) -> Dict[str, np.ndarray]:
        """Create and return the grid as a dictionary with 'lat' and 'lon' keys."""
        pass

    def to_pickle(self, filepath: str) -> None:
        """Save grid to pickle file.
        
        Args:
            filepath: Path where to save the pickled grid
        """
        with open(filepath, 'wb') as fp:
            pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def from_pickle(cls, filepath: str):
        """Load grid from pickle file.
        
        Args:
            filepath: Path to the pickled grid file
            
        Returns:
            Grid instance loaded from file
        """
        with open(filepath, 'rb') as fp:
            return pickle.load(fp)

    def cut_grid(self, lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """Extract grid points within specified latitude and longitude ranges.

        TODO: allow taking regions around the date line (longitude wrap-around)
        Note: Ranges crossing the date line (longitude wrap-around) are not yet supported.

        Args:
            lat_range: (min_lat, max_lat) in degrees
            lon_range: (min_lon, max_lon) in degrees
            
        Returns:
            Dictionary with 'lat' and 'lon' keys containing filtered coordinates
            
        Raises:
            ValueError: If longitude range crosses the date line
            RuntimeError: If no grid has been created yet
        """
        if self.grid is None:
            raise RuntimeError("No grid exists. Call create_grid() first.")
            
        if lon_range[0] > lon_range[1]:
            raise ValueError("Ranges around the date line are not yet supported.")
            
        print(f"Cutting grid to lat: {lat_range}, lon: {lon_range}")

        mask = ((self.grid['lat'] >= lat_range[0]) & 
                (self.grid['lat'] <= lat_range[1]) & 
                (self.grid['lon'] >= lon_range[0]) & 
                (self.grid['lon'] <= lon_range[1]))
        cut_grid = {'lat': self.grid['lat'][mask], 'lon': self.grid['lon'][mask]}
        
        return cut_grid


class RegularGrid(BaseGrid):
    """Gaussian Grid of the earth which is the classical grid type.

    [WIP] This class is under development and not fully tested.

    Args:
    ----
    grid_step_lon: float 
        Grid step in longitudinal direction in degree
    grid_step_lat: float
        Grid step in longitudinal direction in degree

    """

    def __init__(self, grid_step_lon, grid_step_lat, grid=None):
        self.grid_step_lon = grid_step_lon
        self.grid_step_lat = grid_step_lat
        self.grid = grid
        if grid is None:
            self.create_grid() 

    def create_grid(self) -> Dict[str, np.ndarray]:
        """Create regular grid with specified longitude and latitude steps.
        
        Returns:
            Dictionary with 'lat' and 'lon' keys containing grid coordinates
        """
        init_lon, init_lat = self._regular_lon_lat_step(self.grid_step_lon, self.grid_step_lat)

        lon_mesh, lat_mesh = np.meshgrid(init_lon, init_lat)
        
        self.grid = {'lat': lat_mesh.flatten(), 'lon': lon_mesh.flatten()}

        return self.grid
        
    def get_distance_equator(self):
        """Return distance between points at the equator."""
        d_lon = deg_to_equatorial_distance(self.grid_step_lon, radius=6371)
        return d_lon
    
    @staticmethod
    def _regular_lon_lat_step(lon_step: float, lat_step: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create longitude and latitude arrays with specified steps, centered within bounds.
        
        Args:
            lon_step: Longitude step in degrees
            lat_step: Latitude step in degrees
            
        Returns:
            Tuple of (longitude_array, latitude_array)
        """
        num_lon = int(np.ceil(360/lon_step))
        lon_border = (360 - (num_lon-1) * lon_step)/2
        num_lat = int(np.ceil(180/lat_step))
        lat_border = (180 - (num_lat-1) * lat_step)/2
        lon = np.linspace(-180 + lon_border, 180 - lon_border, num_lon)
        lat = np.linspace(-90 + lat_border, 90 - lat_border, num_lat)
        return lon, lat


class GaussianGrid(BaseGrid):
    """Gaussian Grid of the earth which is the classical grid type.

    [WIP] This class is under development and not fully tested.

    Args:
    ----
    grid_step_lon: float 
        Grid step in longitudinal direction in degree
    grid_step_lat: float
        Grid step in longitudinal direction in degree

    """

    def __init__(self, grid_step_lon, grid_step_lat, grid=None):
        self.grid_step_lon = grid_step_lon
        self.grid_step_lat = grid_step_lat
        self.grid = grid
        if grid is None:
            self.create_grid()



    def create_grid(self): # TODO: 确认高斯网格的起止点
        init_lat = np.arange(-89.5, 90.5, self.grid_step_lat)
        init_lon = np.arange(-179.5, 180.5, self.grid_step_lon)

        lon_mesh, lat_mesh = np.meshgrid(init_lon, init_lat) # lats as repeat, lons as tile
        
        self.grid = {'lat': lat_mesh.flatten(), 'lon': lon_mesh.flatten()}

        return self.grid
        


    def get_distance_equator(self):
        """Return distance between points at the equator."""
        d_lon = deg_to_equatorial_distance(self.grid_step_lon, radius=6371)
        return d_lon


class FibonacciGrid(BaseGrid):
    """Fibonacci sphere creates a equidistance grid on a sphere.
    
    [WIP] This class is under development and not fully tested.

    Parameters:
    -----------
    distance_between_points: float
        Distance between the equidistance grid points in km.
    grid: dict (or 'old' makes old version of fib grid, 'maxmin' to maximize min. min-distance)

        If grid is already computed, e.g. {'lon': [], 'lat': []}. Default: None
    """

    def __init__(self, distance_between_points, grid=None, epsilon = 0.36, save = True):
        self.distance = distance_between_points
        self.num_points = distance_to_grid_num(self.distance)
        self.epsilon = None
        self.grid = None
        if grid is None: # maxavg is standard
            self.create_grid(self.num_points, epsilon, save = save)
            self.epsilon = epsilon
        elif grid == 'old':
            self.create_grid('old')
        elif grid == 'maxmin':
            eps = self._maxmin_epsilon(self.num_points)
            self.create_grid(self.num_points, eps, save = save)
            self.epsilon = eps
        else:
            self.grid = grid
        self.reduced_grid = None
    

    def create_grid(self, num_points = 1, epsilon = 0.36, save =True):
        if num_points == 'old':
            """Create Fibonacci grid."""
            print(f'Create fibonacci grid with {self.num_points} points.')
            cartesian_grid = self.fibonacci_sphere(self.num_points)
            lon, lat = cart_to_geo(cartesian_grid[:,0],
                                        cartesian_grid[:,1],
                                        cartesian_grid[:,2])
        else:
            filepath = f'fibonaccigrid_{num_points}_{epsilon}.p'
            if os.path.exists(filepath):
                print(f'\nLoad Fibonacci grid with {self.num_points} points and epsilon = {epsilon}.')
                with open(filepath, 'rb') as fp:
                    self.grid = pickle.load(fp)
                    return self.grid
            else:
                print(f'\nCreate refined fibonacci grid with {num_points} points and epsilon = {epsilon}.')
                goldenRatio = (1 + 5**0.5)/2
                i = np.arange(0, num_points) 
                theta = 2 *np.pi * i / goldenRatio
                phi = np.arccos(1 - 2*(i+epsilon)/(num_points-1+2*epsilon))
                x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
                lon, lat = cart_to_geo(x,y,z)
                self.grid = {'lon': lon, 'lat': lat}
                if save:
                    with open(filepath, 'wb') as fp:
                        pickle.dump(self.grid, fp, protocol=pickle.HIGHEST_PROTOCOL)
        self.grid = {'lon': lon, 'lat': lat}
        return self.grid

    def fibonacci_sphere(self, num_points=1):
        """Creates the fibonacci sphere points on a unit sphere.
        Code inspired by:
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        """
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)


    def fibonacci_refined(self, num_points = 1, epsilon = 0.36): # epsilon = 0.36 for optimal average distance
            
        print(f'Create refined fibonacci grid with {self.num_points} points and epsilon={epsilon}.')

        goldenRatio = (1 + 5**0.5)/2
        i = np.arange(0, num_points) 
        theta = 2 *np.pi * i / goldenRatio
        phi = np.arccos(1 - 2*(i+epsilon)/(num_points-1+2*epsilon))
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        lon, lat = cartesian2spherical(x,y,z)
        self.grid = {'lon': lon, 'lat': lat}
        return self.grid

    def get_num_points(self):
        """Get number of points for this grid's distance setting.
        
        Returns:
            Number of points required for the current distance
        """
        return distance_to_grid_num(self.distance)


    def fit_numPoints_distance(self):
        """Fit functional relationship between next-nearest-neighbor distance of
           fibonacci points and number of points."""
        num_points = np.linspace(200, 10000, 20, dtype=int)
        main_distance = []
        for n in num_points:
            points = self.fibonacci_sphere(n)
            lon, lat = cart_to_geo(points[:,0], points[:,1], points[:,2])

            distance = neighbor_distance(lon, lat)
            hist, bin_edge = np.histogram(distance.flatten(), bins=100)

            main_distance.append(bin_edge[np.argmax(hist)])

        # fit function
        logx = np.log(main_distance)
        logy = np.log(num_points)
        coeffs = np.polyfit(logx, logy, deg=1)

        def func(x):
            return np.exp(coeffs[1]) * x**(coeffs[0])

        dist_array = np.linspace(100, 1400, 20)
        y_fit = func(dist_array)

        # # Plot fit and data
        # fig, ax = plt.subplots()
        # ax.plot(main_distance, num_points)
        # ax.plot(dist_array, y_fit )
        # ax.set_xscale('linear')
        # ax.set_yscale('linear')

        return coeffs

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        return  self.distance
    
    @staticmethod
    def _maxmin_epsilon(num_points):
        """Determine optimal epsilon value for maximizing minimum distance between points.
        
        Args:
            num_points: Number of points in the grid
            
        Returns:
            float: Optimal epsilon value for the given number of points
        """
        if num_points >= 600000:
            epsilon = 214
        elif num_points>= 400000:
            epsilon = 75
        elif num_points>= 11000:
            epsilon = 27
        elif num_points>= 890:
            epsilon = 10
        elif num_points>= 177:
            epsilon = 3.33
        elif num_points>= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
        return epsilon
    
    def keep_original_points(self, orig_grid, regular = True):
        if self.grid is None:
            self.create_grid()
        if regular:
            new_lon, new_lat, dists = [], [], []
            lons = np.sort(np.unique(orig_grid['lon']))
            lats = np.sort(np.unique(orig_grid['lat']))
            for i in range(len(self.grid['lat'])):
                lo = self.grid['lon'][i]
                la = self.grid['lat'][i]
                pm_lon = np.array([(lons[j]-lo)*(lons[j+1]-lo) for j in range(len(lons)-1)])
                pm_lat = np.array([(lats[j]-la)*(lats[j+1]-la) for j in range(len(lats)-1)])
                if np.where(pm_lon<0)[0].shape[0] == 0:
                    rel_lon = [lons[0], lons[-1]]
                else:
                    lon_idx = np.where(pm_lon<0)[0][0]
                    rel_lon = [lons[lon_idx], lons[lon_idx+1]]
                if np.where(pm_lat<0)[0].shape[0] == 0:
                    rel_lat = [lats[0], lats[-1]]
                else:
                    lat_idx = np.where(pm_lat<0)[0][0]
                    rel_lat = [lats[lat_idx], lats[lat_idx+1]]
                min_dist = 99999
                for l1 in rel_lon:
                    for l2 in rel_lat:
                        this_dist = geo_distance(l1, l2, lo, la)
                        if this_dist < min_dist:
                            min_lon, min_lat = l1, l2
                            min_dist = this_dist
                new_lon.append(min_lon)
                new_lat.append(min_lat)
                dists.append(min_dist)
            self.reduced_grid = {'lon': np.array(new_lon), 'lat': np.array(new_lat)}
            return np.array(dists)
        else:
            raise KeyError('Only regular grids!')

    def min_dists(self, grid2 = None):
        if grid2 is None:
            lon1, lon2 = self.grid['lon'], self.grid['lon']
            lat1, lat2 = self.grid['lat'], self.grid['lat']
            d = 9999 * np.ones((len(lon1),len(lon2)))
            for i in range(len(lon1)):
                for j in range((len(lon1))):
                    if i < j:
                        d[i,j] = geo_distance(lon1[i], lat1[i], lon2[j], lat2[j])
                    elif i>j:
                        d[i,j] = d[j,i]
            return d.min(axis=1)
        else: # min dist from self.grid point to other grid
            lon1, lon2 = self.grid['lon'], grid2['lon']
            lat1, lat2 = self.grid['lat'], grid2['lat']
            d = 9999 * np.ones((len(lon1),len(lon2)))
            for i in range(len(lon1)):
                for j in range(len(lon2)):
                    d[i,j] = geo_distance(lon1[i], lat1[i], lon2[j], lat2[j])
            return d.min(axis=1)


class FeketeGrid(BaseGrid):
    """FeketeGrid creates a equidistance grid on a sphere.

    Parameters:
    -----------
    num_points: int
        Number of points to generate on the sphere
    num_iter: int, optional
        Number of iterations for grid optimization. Default: 1000
    grid: None, dict, or tuple, optional
        Grid initialization method:
        - None: Create new grid from scratch (default)
        - dict: Initialize from {'lat': [...], 'lon': [...]} dictionary
        - tuple: Initialize from fekete.bendito output (X, dq)
    parallel: bool or None, optional
        Whether to use parallel execution. None for auto-detect based on memory
    """
    def __init__(self, num_points: int, num_iter: int = 0, 
                 grid: Optional[Union[Dict[str, np.ndarray], Tuple]] = None, 
                 parallel: Optional[bool] = None) -> None:
        # Validate inputs
        if not isinstance(num_points, int) or num_points <= 0:
            raise ValueError("num_points must be a positive integer")
        if not isinstance(num_iter, int) or num_iter < 0:
            raise ValueError("num_iter must be a non-negative integer")
            
        # Initialize basic parameters
        self.num_points = num_points
        self.num_iter = num_iter
        self.grid: Optional[Dict[str, np.ndarray]] = None
        self.dq: List[float] = []
        self._plys_coords: Optional[np.ndarray] = None
        
        # Create or load grid based on input type
        if grid is None:
            # Case 1: Create new grid from scratch
            self.create_grid(num_iter=num_iter, parallel=parallel)
        elif isinstance(grid, dict):
            # Case 2: Initialize from lat/lon dictionary
            self._create_from_dict(grid)
        elif isinstance(grid, tuple):
            # Case 3: Initialize from fekete.py output tuple
            self.grid = self.create_grid_from_tuple(grid)
        else:
            raise TypeError("Grid must be None, dict with 'lat'/'lon' keys, or tuple from fekete.py output")
    
    """
    properties and parameters
    """
    @property
    def distance(self) -> float:
        """Get the average distance between points in km."""
        return grid_num_to_distance(self.num_points)

    def get_distance_equator(self) -> float:
        """Return distance between points at the equator in km."""
        return self.distance
    
    def _determine_parallel_mode(self, parallel: Optional[bool], num_points: int) -> bool:
        """Determine whether to use parallel execution based on memory constraints.
        
        Args:
            parallel: User preference for parallel execution (None for auto-detect)
            num_points: Number of points in the grid
            
        Returns:
            True if parallel execution should be used, False otherwise
        """
        if parallel is None:
            try:
                # Test if we can allocate the required memory for serial execution
                test_array = np.zeros((num_points, num_points), dtype=np.float64)
                del test_array  # Clean up immediately
                return False
            except MemoryError:
                print("Not enough memory to run serially. Running in parallel ...")
                return True
        return parallel
    
    """
    different ways to create FeketeGrid
    """
    def _create_from_dict(self, grid_dict: Dict[str, np.ndarray]) -> None:
        """Create grid from lat/lon dictionary.
        
        Args:
            grid_dict: Dictionary with 'lat' and 'lon' keys containing coordinate arrays
            
        Raises:
            ValueError: If dictionary format is invalid or arrays have different lengths
        """
        if not isinstance(grid_dict, dict) or 'lat' not in grid_dict or 'lon' not in grid_dict:
            raise ValueError("Grid dict must contain 'lat' and 'lon' keys")
        
        # Validate that lat and lon arrays have the same length
        if len(grid_dict['lat']) != len(grid_dict['lon']):
            raise ValueError("Latitude and longitude arrays must have the same length")
        
        # Check if grid length matches num_points
        actual_points = len(grid_dict['lat'])
        if actual_points != self.num_points:
            warnings.warn(f"Grid has {actual_points} points but num_points was set to {self.num_points}. "
                            f"Updating num_points to {actual_points}.")
            self.num_points = actual_points
        
        # Store the grid and initialize dq as empty list
        self.grid = {'lat': np.array(grid_dict['lat']), 'lon': np.array(grid_dict['lon'])}
        self.dq = []
    
    def create_grid_from_tuple(self, grid_tuple: Tuple[np.ndarray, List[float]]) -> Dict[str, np.ndarray]:
        """Create grid from fekete.bendito output tuple.
        
        Args:
            grid_tuple: Tuple containing (X, dq) where X is 3D Cartesian coordinates
                       and dq is the optimization history
                       
        Returns:
            Dictionary with 'lat' and 'lon' keys containing grid coordinates
            
        Raises:
            ValueError: If tuple format is invalid
        """
        try:
            X, self.dq = grid_tuple
        except (ValueError, TypeError):
            raise ValueError('Grid tuple should contain (coordinates_array, optimization_history)')
            
        if not isinstance(X, np.ndarray) or X.shape[1] != 3:
            raise ValueError('Coordinates array should be Nx3 numpy array')
            
        lon, lat = cart_to_geo(X[:,0], X[:,1], X[:,2])
        grid = {'lon': lon, 'lat': lat}
        
        # Check if grid length matches num_points
        actual_points = len(grid['lat'])
        if actual_points != self.num_points:
            warnings.warn(f"Grid has {actual_points} points but num_points was set to {self.num_points}. "
                            f"Updating num_points to {actual_points}.")
            self.num_points = actual_points
        
        return grid
    
    def create_grid(self, num_iter: int = 1000, parallel: Optional[bool] = None) -> None:
        """Create new Fekete grid from scratch.
        
        Args:
            num_iter: Number of iterations for grid optimization
            parallel: Whether to use parallel execution (None for auto-detect)
        """
        # Initialize with random configuration
        self.initialize_grid(save=False)  # TODO: save is kept for future development of `save_epoch`
        
        # Improve the grid if iterations are requested
        if num_iter > 0:
            self.improve_grid(num_iter=num_iter, save=False, parallel=parallel)
        
    
    """
    save and load
    """
    def to_pickle(self, filepath: Optional[str] = None) -> None:
        """Save FeketeGrid to pickle file.
        
        Args:
            filepath: Path to save file. If None, generates default name.
        """
        if filepath is None:
            filepath = f'feketegrid_n{self.num_points}_i{self.num_iter}.p'
        return super().to_pickle(filepath)

    def save_grid(self, filepath: Optional[str] = None) -> None:
        """Save grid to pickle file.
        
        .. deprecated:: 
            Use to_pickle() instead. This method will be removed in a future version.
        """
        warnings.warn(
            "save_grid is deprecated and will be removed in a future version. Use to_pickle instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if filepath is None:
            filepath = f'feketegrid_{self.num_points}_{self.num_iter}.p'
        with open(filepath, 'wb') as fp:
            pickle.dump((self.grid, self.dq), fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load_grid(cls, filepath: Optional[str] = None):
        """Load grid from pickle file.
        
        .. deprecated:: 
            Use from_pickle() instead. This method will be removed in a future version.
        """
        warnings.warn(
            "load_grid is deprecated and will be removed in a future version. Use from_pickle instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if filepath is None:
            filepath = f'feketegrid_{self.num_points}_{self.num_iter}.p'
        with open(filepath, 'rb') as fp:
            self.grid, self.dq = pickle.load(fp)


    """
    Core: Initialize => Improve logic
    """
    def initialize_grid(self, save: bool = True) -> Dict[str, np.ndarray]:
        print("Epochwise generation launched. Generating random initial configuration ...")
        X = points_on_sphere(self.num_points)         # initial random configuration
        lon, lat = cart_to_geo(X[:,0], X[:,1], X[:,2])
        self.num_iter = 0
        self.grid = {'lon': lon, 'lat': lat}
        self.dq = []
        if save:
            filepath = f'feketegrid_{self.num_points}_{0}.p'
            self.save_grid(filepath)
        return self.grid
    

    def improve_grid(self, num_iter: int = 1000, save: bool = True, parallel: Optional[bool] = None) -> Dict[str, np.ndarray]:
        """Improve existing grid through optimization iterations.
        
        Args:
            num_iter: Number of optimization iterations to perform
            save: Whether to save the improved grid to file
            parallel: Whether to use parallel execution (None for auto-detect)
            
        Returns:
            Updated grid dictionary with 'lat' and 'lon' keys
            
        Raises:
            RuntimeError: If no grid exists to improve
        """
        if self.grid is None:
            raise RuntimeError("No grid exists to improve. Call initialize_grid() first.")
            
        X0 = np.array(geo_to_cart(self.grid['lon'], self.grid['lat'])).T
        parallel_mode = self._determine_parallel_mode(parallel, self.num_points) if parallel is not None else False
        X, dq = bendito(N=self.num_points, maxiter=num_iter, X=X0, parallel=parallel_mode)
        self.dq.append(dq)
        lon, lat = cart_to_geo(X[:,0], X[:,1], X[:,2])
        self.grid = {'lon': lon, 'lat': lat}
        self.num_iter = self.num_iter + num_iter

        if save:
            filepath = f'feketegrid_{self.num_points}_{self.num_iter}.p'
            self.save_grid(filepath)
        return self.grid
        

    """
    Boundaries calculation
    """
    def create_SphericalVoronoi(self):
        import scipy.spatial as sp
        spVoronoi = sp.SphericalVoronoi(np.array(geo_to_cart(self.grid["lon"], self.grid["lat"])).T, threshold=1e-8)
        spVoronoi.sort_vertices_of_regions()
        self.regions_vertidx = spVoronoi.regions
        self.vertices = np.array(cart_to_geo(*spVoronoi.vertices.T)).T
        self.regions_vert = [self.vertices[idx, :] for idx in self.regions_vertidx]
        # self.spVoronoi = spVoronoi  # TODO: remove after testing
        return self.regions_vert
    

    def _harmonize_SphericalVoronoi(self):
        import shapely as sh
        # 先harmonize个数
        harmonTarget = np.array([len(x) for x in self.regions_vert]).max()
        harmonized_regions_vertidx = self.regions_vertidx
        harmonized_regions_vertidx = [np.insert(np.array(r), np.zeros(harmonTarget-len(r), dtype="int"), r[0]) 
                                      for r in harmonized_regions_vertidx]
        # 再harmonize方向(根据ccw)
        harmonized_regions_vertidx = [r if sh.LinearRing(self.vertices[r, :]).is_ccw else r[::-1]
                                      for r in harmonized_regions_vertidx]
        # np.array([sh.LinearRing(self.vertices[r, :]).is_ccw for r in harmonized_regions_vertidx]).all() # TEST for ccw
        # 再生成_plys_coords
        self._plys_coords = np.array([self.vertices[r, :] for r in harmonized_regions_vertidx])
        return self._plys_coords

    """
    API for output
    """
    def to_cdo_gridfile(self, filepath = None, out_vertex = False):
        self.to_gridfile(filepath, out_vertex)
    
    def to_gridfile(self, filepath = None, out_vertex = False):
        if filepath is None:
            filepath = f'feketegrid_n{self.num_points}_it{self.num_iter}.txt'
        with open(filepath, 'w') as fp:
            fp.write("gridtype\t= unstructured\n")
            fp.write(f"gridsize\t= {self.num_points}\n")
            fp.write(f"xsize\t= {self.num_points}\n")
            fp.write(f"ysize\t= {self.num_points}\n")
            xvals = str.join("   ", self.grid["lon"].astype(str))
            fp.write(f"xvals\t= {xvals}\n")
            yvals = str.join("   ", self.grid["lat"].astype(str))
            fp.write(f"yvals\t= {yvals}\n")
            if out_vertex:
                if self._plys_coords is None:
                    self.create_SphericalVoronoi()
                # _ = self.harmonize_nvertex()
                _ = self._harmonize_SphericalVoronoi()
                fp.write(f"nvertex\t= {self._plys_coords.shape[1]}\n")
                xbounds = str.join("   ", self._plys_coords[:, :, 0].ravel().astype(str))
                fp.write(f"xbounds\t= {xbounds}\n")
                ybounds = str.join("   ", self._plys_coords[:, :, 1].ravel().astype(str))
                fp.write(f"ybounds\t= {ybounds}\n")
            fp.write("xunits\t= degrees\n")
            fp.write("yunits\t= degrees\n")
        return filepath
    
    # TODO: API to uxarray here!

    """
    Utilities
    """
    def align_to_target_grid(self, target_coords, method='L-BFGS-B', initial_guess=None):
        """
        Align Fekete grid to target coordinates by minimizing spherical distances.
        
        This function uses numerical optimization to find the rotation that minimizes the sum
        of great circle distances between grid points and their targets. Unlike the Wahba
        solution which minimizes Euclidean distances, this directly optimizes spherical distances.
        
        Parameters
        ----------
        target_coords : dict
            Target coordinates with 'lon' and 'lat' keys containing arrays of target positions.
            Must have same length as current grid.
        method : str, default='L-BFGS-B'
            Optimization method. Options: 'L-BFGS-B', 'BFGS', 'Powell', 'Nelder-Mead'
        initial_guess : array-like, optional
            Initial rotation vector (3,) for optimization. If None, uses Wahba solution as starting point.
            
        Returns
        -------
        dict
            New grid coordinates {'lon': array, 'lat': array} after optimal spherical alignment.
            Also contains optimization info: 'success', 'message', 'nfev', 'fun'
            
        Notes
        -----
        Optimization method: **Numerical optimization** (not theoretical solution)
        - Uses rotation vector parameterization to avoid matrix constraints
        - Minimizes: Σ arccos(clamp((R @ current_i) · target_i, -1, 1))
        - No closed-form solution exists for this objective function
        - Convergence depends on initial guess and grid configuration
        
        The rotation vector r represents rotation by ||r|| radians around axis r/||r||.
        """
        from scipy.optimize import minimize
        
        # Validate inputs (same as previous function)
        if self.grid is None:
            raise ValueError("No Fekete grid exists. Call create_grid() first.")
        
        if not isinstance(target_coords, dict) or 'lat' not in target_coords or 'lon' not in target_coords:
            raise ValueError("target_coords must be dict with 'lat' and 'lon' keys")
            
        if len(target_coords['lat']) != len(self.grid['lat']):
            raise ValueError("target_coords must have same length as current grid")
        
        # Convert to 3D Cartesian coordinates
        curr_x, curr_y, curr_z = geo_to_cart(self.grid['lon'], self.grid['lat'])
        current_points = np.array([curr_x, curr_y, curr_z]).T
        current_points = current_points / np.linalg.norm(current_points, axis=1, keepdims=True)
        
        target_x, target_y, target_z = geo_to_cart(target_coords['lon'], target_coords['lat'])
        target_points = np.array([target_x, target_y, target_z]).T
        target_points = target_points / np.linalg.norm(target_points, axis=1, keepdims=True)
        
        def spherical_distance_objective(rotation_vector):
            """Objective function: sum of spherical distances after rotation."""
            # Convert rotation vector to rotation matrix
            if np.linalg.norm(rotation_vector) < 1e-10:
                R = np.eye(3)
            else:
                R = Rot.from_rotvec(rotation_vector).as_matrix()
            
            # Apply rotation
            rotated_points = (R @ current_points.T).T
            rotated_points = rotated_points / np.linalg.norm(rotated_points, axis=1, keepdims=True)
            
            # Compute spherical distances
            dot_products = np.sum(rotated_points * target_points, axis=1)
            # Clamp to [-1, 1] to handle numerical errors
            dot_products = np.clip(dot_products, -1.0, 1.0)
            spherical_distances = np.arccos(dot_products)
            
            return np.sum(spherical_distances)
        
        # Get initial guess
        if initial_guess is None:
            # Use Wahba solution as starting point
            wahba_result = self.align_to_target_grid(target_coords)
            # Convert to rotation vector by comparing original and Wahba result
            wahba_x, wahba_y, wahba_z = geo_to_cart(wahba_result['lon'], wahba_result['lat'])
            wahba_points = np.array([wahba_x, wahba_y, wahba_z]).T
            wahba_points = wahba_points / np.linalg.norm(wahba_points, axis=1, keepdims=True)
            
            # Find rotation from current to Wahba result
            H = current_points.T @ wahba_points
            U, S, Vt = np.linalg.svd(H)
            R_init = Vt.T @ U.T
            if np.linalg.det(R_init) < 0:
                Vt_corrected = Vt.copy()
                Vt_corrected[-1, :] *= -1
                R_init = Vt_corrected.T @ U.T
            
            initial_guess = Rot.from_matrix(R_init).as_rotvec()
        
        # Optimize using numerical methods
        result = minimize(
            spherical_distance_objective,
            initial_guess,
            method=method,
            options={'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        # Apply optimal rotation
        if np.linalg.norm(result.x) < 1e-10:
            optimal_R = np.eye(3)
        else:
            optimal_R = Rot.from_rotvec(result.x).as_matrix()
        
        final_points = (optimal_R @ current_points.T).T
        final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
        
        # Convert back to spherical coordinates
        new_lon, new_lat = cart_to_geo(final_points[:, 0], final_points[:, 1], final_points[:, 2])
        
        return {
            'lon': new_lon, 
            'lat': new_lat,
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'fun': result.fun  # Final spherical distance sum
        }
    
    def map_to_regular_grid(self, target_grid):
        """
        Creates a one-to-one mapping between Fekete grid points and regular grid points.
        
        Each Fekete point is mapped to its nearest available regular grid point,
        ensuring no regular grid point is used twice. Results maintain original Fekete grid order.
        
        Parameters
        ----------
        target_grid : dict
            Dictionary with 'lon' and 'lat' keys containing arrays of regular grid coordinates
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'regular_coords': corresponding regular grid coordinates (lon, lat pairs)
            - 'distances': distances between mapped points
            - 'unmapped_indices': indices of Fekete points that couldn't be mapped
            
        Raises
        ------
        ValueError
            If no grid has been created yet
        """
        if self.grid is None:
            raise ValueError("No Fekete grid exists. Call create_grid() first.")
        
        # Extract and sort unique coordinates from target regular grid
        target_lons = np.sort(np.unique(target_grid['lon']))
        target_lats = np.sort(np.unique(target_grid['lat']))
        
        num_fekete_points = len(self.grid['lat'])
        candidate_distances = []
        candidate_coordinates = []
        
        # PHASE 1: Find candidate regular grid points for each Fekete point
        for fekete_idx in range(num_fekete_points):
            fekete_lon = self.grid['lon'][fekete_idx]
            fekete_lat = self.grid['lat'][fekete_idx]
            
            # Find longitude/latitude intervals (same logic as original)
            lon_products = (target_lons[:-1] - fekete_lon) * (target_lons[1:] - fekete_lon)
            lat_products = (target_lats[:-1] - fekete_lat) * (target_lats[1:] - fekete_lat)
            
            # Determine bounds
            lon_negative_indices = np.where(lon_products < 0)[0]
            if len(lon_negative_indices) == 0:
                bounding_lons = [target_lons[0], target_lons[-1]]
            else:
                lon_idx = lon_negative_indices[0]
                bounding_lons = [target_lons[lon_idx], target_lons[lon_idx + 1]]
            
            lat_negative_indices = np.where(lat_products < 0)[0]
            if len(lat_negative_indices) == 0:
                bounding_lats = [target_lats[0], target_lats[-1]]
            else:
                lat_idx = lat_negative_indices[0]
                bounding_lats = [target_lats[lat_idx], target_lats[lat_idx + 1]]
            
            # Calculate distances (same as original)
            distances = []
            coordinates = []
            for lon in bounding_lons:
                for lat in bounding_lats:
                    distance = geo_distance(lon, lat, fekete_lon, fekete_lat)
                    distances.append(distance)
                    coordinates.append((lon, lat))
            
            # Sort by distance
            distance_order = np.argsort(distances)
            candidate_distances.append(np.array(distances)[distance_order])
            candidate_coordinates.append(np.array(coordinates)[distance_order])
        
        # PHASE 2: Assignment - using original algorithm but with pre-allocated arrays
        # Pre-allocate result arrays for better performance
        assigned_coords = []
        assigned_distances = []
        unmapped_indices = []
        used_coord_set = set()  # Fast O(1) lookup instead of O(n) list search
        
        # Process in order of nearest distance (like original)
        nearest_distances = np.array([distances[0] for distances in candidate_distances])
        processing_order = np.argsort(nearest_distances)
        
        for fekete_idx in processing_order:
            assigned = False
            
            # Try each candidate coordinate
            for candidate_idx in range(len(candidate_coordinates[fekete_idx])):
                coord = tuple(candidate_coordinates[fekete_idx][candidate_idx])  # Convert to tuple for set
                
                if coord not in used_coord_set:
                    # Assign this coordinate
                    assigned_coords.append(coord)
                    assigned_distances.append(candidate_distances[fekete_idx][candidate_idx])
                    used_coord_set.add(coord)
                    assigned = True
                    break
            
            if not assigned:
                unmapped_indices.append(fekete_idx)
                warnings.warn(
                    f'No available regular grid points for Fekete point at '
                    f'({self.grid["lon"][fekete_idx]:.3f}, {self.grid["lat"][fekete_idx]:.3f}). '
                    f'Point will be unmapped.'
                )
        
        # PHASE 3: Restore original Fekete grid order (SIMPLIFIED)
        mapped_fekete_indices = [idx for idx in processing_order if idx not in unmapped_indices]
        
        if len(mapped_fekete_indices) > 0:
            # Create inverse mapping to restore original order
            order_map = {fekete_idx: i for i, fekete_idx in enumerate(mapped_fekete_indices)}
            restore_order = [order_map[idx] for idx in sorted(mapped_fekete_indices)]
            
            final_regular_coords = np.array(assigned_coords)[restore_order]
            final_distances = np.array(assigned_distances)[restore_order]
            mapped_fekete_indices = np.array(sorted(mapped_fekete_indices))
        else:
            final_regular_coords = np.array([]).reshape(0, 2)
            final_distances = np.array([])
            mapped_fekete_indices = np.array([])
        
        return {
            # 'fekete_indices': mapped_fekete_indices,
            'regular_coords': final_regular_coords,
            'distances': final_distances,
            'unmapped_indices': np.array(unmapped_indices)
        }


def grid_num_to_distance(num_points: int) -> float:
    """Convert number of grid points to distance between points.
    
    Calculates the corresponding distance between points (in km) for a given
    number of grid points on an equidistant Earth grid.
    
    Args:
        num_points: Number of points on the grid
        
    Returns:
        Distance between adjacent points in km
        
    Raises:
        ValueError: If num_points is not positive
    
    Example:
        >>> distance = grid_num_to_distance(1000)
        >>> print(f"Distance: {distance:.2f} km")
    """
    if hasattr(num_points, 'item'):  # Handle numpy array elements
        num_points = num_points.item()
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points must be a positive integer")
    k = -2.01155176  # Same k as in distance_to_grid_num
    a = np.exp(20.0165958)  # Same a as in distance_to_grid_num
    return (num_points/a)**(1/k)  # Inverse formula: distance = (num_points/a)^(1/k)

def distance_to_grid_num(distance: float) -> int:
    """Convert distance between points to required number of grid points.
    
    Calculates how many grid points are needed to achieve a specific
    distance between adjacent points (in km) on an equidistant Earth grid.
    
    Args:
        distance: Target distance between grid points in km
        
    Returns:
        Number of grid points required
        
    Raises:
        ValueError: If distance is not positive
    
    Example:
        >>> # For 0.25 degree spacing (≈ 27.75 km at equator)
        >>> num_points = distance_to_grid_num(111*0.25)  # 111 km per degree at equator
        >>> print(f"Points needed: {num_points}")
    """
    if hasattr(distance, 'item'):  # Handle numpy array elements
        distance = distance.item()
    if not isinstance(distance, (int, float)) or distance <= 0:
        raise ValueError("distance must be a positive number")
    
    # Constants from log-log fit of grid data
    k = -2.01155176
    a = np.exp(20.0165958)
    return int(a * distance**k)


# def regular_lon_lat(num_lon, num_lat): # creates regular grid with borders half the distance of one step at each border
#     lon = np.linspace(-180+360/(2*num_lon),180-360/(2*num_lon),num_lon)
#     lat = np.linspace(-90 + 180/(2*num_lat), 90 - 180/(2*num_lat), num_lat)
#     return lon, lat


def cart_to_geo(x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert 3D Cartesian coordinates to geographic coordinates (longitude and latitude).
    
    Args:
        x: X coordinate in 3D Cartesian space (unit sphere)
        y: Y coordinate in 3D Cartesian space (unit sphere)
        z: Z coordinate in 3D Cartesian space (unit sphere)
        
    Returns:
        Tuple of (longitude, latitude) in degrees
        - longitude: [-180, 180] degrees
        - latitude: [-90, 90] degrees
        
    Note:
        Input coordinates should be on the unit sphere (x² + y² + z² = 1)
    """
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    return lon, lat

def geo_to_cart(lon: Union[float, np.ndarray], lat: Union[float, np.ndarray], radius: float = 1) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert geographic coordinates (longitude and latitude) to 3D Cartesian coordinates.
    
    Args:
        lon: Longitude in degrees [-180, 180]
        lat: Latitude in degrees [-90, 90]
        radius: Radius of the sphere, default is 1 (unit sphere)
        
    Returns:
        Tuple of (x, y, z) coordinates in 3D Cartesian space
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
        
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = radius * np.cos(lon_rad) * np.cos(lat_rad)
    y = radius * np.sin(lon_rad) * np.cos(lat_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

@np.vectorize
def geo_distance(lon1: Union[float, np.ndarray], lat1: Union[float, np.ndarray], 
                lon2: Union[float, np.ndarray], lat2: Union[float, np.ndarray], 
                radius: float = 6371) -> Union[float, np.ndarray]:
    """Calculate the great-circle distance between two points on a sphere.
    
    Uses the haversine formula to compute the shortest distance over the earth's surface.
    This implementation uses arctan2 instead of arcsin for better numerical stability,
    especially for very small distances.
    
    Args:
        lon1: Longitude of first point in degrees
        lat1: Latitude of first point in degrees
        lon2: Longitude of second point in degrees
        lat2: Latitude of second point in degrees
        radius: Radius of the sphere in km, default is Earth's radius (6371 km)
        
    Returns:
        Distance between points in the same units as radius
        
    Raises:
        ValueError: If radius is not positive
        
    Note:
        This function is automatically vectorized for numpy arrays.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
        
    # Convert to radians
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


def deg_to_equatorial_distance(grid_step: float, radius: float = 6371) -> float:
    """Convert angular grid resolution (degrees) to equivalent distance at the equator (km).
    
    Args:
        grid_step: Angular resolution/spacing in degrees
        radius: Radius of the sphere in km, default is Earth's radius (6371 km)
        
    Returns:
        Equivalent spatial distance/spacing in kilometers at the equator
        
    Raises:
        ValueError: If grid_step is not positive
    """
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
        
    return geo_distance(0, 0, grid_step, 0, radius=radius)


def equatorial_distance_to_deg(distance: float, radius: float = 6371) -> float:
    """Convert equatorial distance (km) to equivalent angular grid resolution (degrees).
    
    Args:
        distance: Spatial distance/spacing in kilometers at the equator
        radius: Radius of the sphere in km, default is Earth's radius (6371 km)
        
    Returns:
        Equivalent angular resolution/spacing in degrees
        
    Raises:
        ValueError: If distance is not positive
    """
    if distance <= 0:
        raise ValueError("distance must be positive")
        
    return distance * 360 / (2 * np.pi * radius)

# def min_dists(grid1, grid2 = None): TODO: implement grid2 but I dont know why need it
#     if grid2 is None:
#         lon1, lon2 = grid1['lon'], grid1['lon']
#         lat1, lat2 = grid1['lat'], grid1['lat']
#         d = 9999 * np.ones((len(lon1),len(lon2)))
#         for i in range(len(lon1)):
#             for j in range((len(lon1))):
#                 if i < j:
#                     d[i,j] = geo_distance(lon1[i], lat1[i], lon2[j], lat2[j])
#                 elif i>j:
#                     d[i,j] = d[j,i]
#         return d.min(axis=1)
#     else: # min dist from self.grid point to other grid
#         lon1, lon2 = grid1['lon'], grid2['lon']
#         lat1, lat2 = grid1['lat'], grid2['lat']
#         d = 9999 * np.ones((len(lon1),len(lon2)))
#         for i in range(len(lon1)):
#             for j in range(len(lon2)):
#                 d[i,j] = geo_distance(lon1[i], lat1[i], lon2[j], lat2[j])
#         return d.min(axis=1)

def neighbour_distance(grid: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate the distance to the nearest neighboring point for each point in the grid.
    
    This function computes the nearest neighbor distance for each point, which is useful
    for analyzing grid uniformity and identifying spatial outliers.
    
    Args:
        grid: Grid dictionary with 'lon' and 'lat' keys containing coordinates in degrees
        
    Returns:
        Array of minimum distances (in km) to the nearest neighbor for each point
    """
    lon = np.asarray(grid['lon'])
    lat = np.asarray(grid['lat'])
    
    n_points = len(lon)
    
    # Initialize distance matrix with NaN values
    d = np.full((n_points, n_points), np.nan)
    
    # Calculate distances between all pairs of points (only upper triangle for efficiency)
    for i in range(n_points):
        for j in range(i+1, n_points):
            distance = geo_distance(lon[i], lat[i], lon[j], lat[j])
            d[i, j] = distance
            d[j, i] = distance  # Mirror the distance matrix
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(d, np.inf)
    
    # Find minimum distance for each point
    return np.nanmin(d, axis=1)

