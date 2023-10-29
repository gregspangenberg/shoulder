# from shoulder.humerus import surgical_neck #ideally imported but will infitinte loop
from shoulder.humerus import mesh

from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np


class Slices(ABC):
    def __init__(
        self, obb: mesh.Obb, zslice_num: int, interp_num: int, return_odd=False
    ):
        self._mesh_oriented_uobb = obb.mesh
        self.obb = obb
        self.return_odd = return_odd
        self._zslice_num = zslice_num
        self._interp_num = interp_num
        self._z_orig = np.mean(self._zs)
        self._z_incrs = self._zs - self._z_orig

    @cached_property
    def _slices(self):
        # grab the polygon of the slice
        origin = [0, 0, self._z_orig]
        normal = [0, 0, 1]
        slices = self.obb.mesh.section_multiplane(
            plane_origin=origin, plane_normal=normal, heights=self._z_incrs
        )
        return slices

    def slices(self, cutoff: tuple):
        return self._cutoff(self._slices, cutoff)

    @cached_property
    def _centroids(self):
        cents = np.zeros((len(self._z_incrs), 2))
        for i, s in enumerate(self._slices):
            cents[i] = s.centroid
        return cents

    @cached_property
    def _centroids_repeated(self):
        cents = np.repeat(self._centroids.reshape(-1, 2, 1), self._interp_num, axis=2)
        return cents

    def centroids(self, cutoff: tuple):
        return self._cutoff(self._centroids, cutoff)

    @cached_property
    def _areas1(self):
        area1 = np.zeros(len(self._z_incrs))
        for i, slice in enumerate(self._slices):
            if len(slice.entities) > 1:
                # keep only largest polygon if more than 1
                area1[i] = slice.polygons_closed[
                    np.argmax([p.area for p in slice.polygons_closed])
                ].area
            else:
                area1[i] = slice.area
        return area1

    def areas1(self, cutoff: tuple):
        return self._cutoff(self._areas1, cutoff)

    @cached_property
    def _ixy(self) -> np.ndarray:
        # preallocate variables
        cart = np.zeros((len(self._z_incrs), 2, self._interp_num))
        for i, slice in enumerate(self._slices):
            if len(slice.entities) > 1:
                # keep only largest polygon if more than 1
                slice = slice.discrete[
                    np.argmax([p.area for p in slice.polygons_closed])
                ]
            else:
                slice = slice.discrete[0]
            # resample cartesion coordinates to create evenly spaced points
            cart[i] = self._resample_polygon(slice, self._interp_num).T

        return cart

    def ixy(self, cutoff: tuple):
        return self._cutoff(self._ixy, cutoff)

    @cached_property
    def _ixy_centered(self):
        return self._ixy - self._centroids_repeated

    def ixy_centered(self, cutoff: tuple):
        return self._cutoff(self._ixy_centered, cutoff)

    @cached_property
    def _itr(self):
        polar = np.zeros(self._ixy.shape)
        for i, p in enumerate(polar):
            polar[i] = self._cart2pol(self._ixy[i][0, :], self._ixy[i][1, :])
        return polar

    def itr(self, cutoff: tuple) -> np.ndarray:
        return self._cutoff(self._ixy, cutoff)

    @cached_property
    def _itr_start(self):
        polar = np.zeros(self._ixy.shape)
        for i, p in enumerate(polar):
            pol = self._cart2pol_no_sort(self._ixy[i][0, :], self._ixy[i][1, :])
            polar[i] = np.c_[pol[:, np.argmin(pol[0]) :], pol[:, : np.argmin(pol[0])]]
        return polar

    def itr_start(self, cutoff: tuple):
        return self._cutoff(self._itr_start, cutoff)

    @cached_property
    def _itr_start_even_theta(self):
        polar = np.zeros(self._ixy.shape)
        for i, p in enumerate(polar):
            pol = self._cart2pol_no_sort(self._ixy[i][0, :], self._ixy[i][1, :])
            polar[i] = np.c_[pol[:, np.argmin(pol[0]) :], pol[:, : np.argmin(pol[0])]]
        return polar

    def itr_start_even_theta(self, cutoff: tuple):
        return self._cutoff(self._itr_start, cutoff)

    @cached_property
    def _itr_centered(self):
        polar = np.zeros(self._ixy.shape)
        for i, p in enumerate(polar):
            polar[i] = self._cart2pol(
                self._ixy_centered[i][0, :], self._ixy_centered[i][1, :]
            )
        return polar

    def itr_centered(self, cutoff: tuple):
        return self._cutoff(self._itr_centered, cutoff)

    @cached_property
    def _itr_centered_start(self):
        polar = np.zeros(self._ixy.shape)
        for i, p in enumerate(polar):
            pol = self._cart2pol_no_sort(
                self._ixy_centered[i][0, :], self._ixy_centered[i][1, :]
            )
            polar[i] = np.c_[pol[:, np.argmin(pol[0]) :], pol[:, : np.argmin(pol[0])]]
        return polar

    def itr_centered_start(self, cutoff: tuple):
        return self._cutoff(self._itr_centered_start, cutoff)

    @cached_property
    @abstractmethod
    def _zs(self) -> np.ndarray:
        """returns the z's over whole interval"""

    def zs(self, cutoff) -> np.ndarray:
        return self._cutoff(self._zs, cutoff)

    def _cutoff(self, entity, cutoff: tuple):
        start_i = int((1 - cutoff[1]) * len(entity))
        end_i = int((1 - cutoff[0]) * len(entity))
        if self.return_odd:
            if (len(entity[start_i:end_i]) % 2) == 0:
                end_i -= 1

        return entity[start_i:end_i]

    def _resample_polygon(self, xy: np.ndarray, interp_num: int) -> np.ndarray:
        """interpolate between points in array to ensure even spacing between all points

        Args:
            xy (np.ndarray): array with columns for x and y coordinates arranged by order in
                which they occur while tracing along edge of polygon.
            n_points (int, optional): number of evenly spaced points to return. Defaults to 100.

        Returns:
            np.ndarray: evenly spaced points
        """
        # Cumulative Euclidean distance between successive polygon points.
        # This will be the "x" for interpolation
        d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

        # get linearly spaced points along the cumulative Euclidean distance
        d_sampled = np.linspace(0, d.max(), interp_num)

        # interpolate x and y coordinates
        xy_interp = np.c_[
            np.interp(d_sampled, d, xy[:, 0]), np.interp(d_sampled, d, xy[:, 1])
        ]

        return xy_interp

    def _cart2pol(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """convert from cartesian coordinates to radial"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        sorter = np.argsort(theta)
        r_arr = np.vstack((theta[sorter], r[sorter]))

        return r_arr

    def _cart2pol_no_sort(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """convert from cartesian coordinates to radial"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        r_arr = np.vstack((theta, r))

        return r_arr


class FullSlices(Slices):
    def __init__(
        self,
        obb,
        zslice_num=200,
        interp_num=100,
        return_odd=False,
    ):
        super().__init__(obb, zslice_num, interp_num, return_odd)

    @cached_property
    def _zs(self) -> np.ndarray:
        z_max = 0.99 * np.max(self.obb.mesh.bounds[:, -1])
        z_min = 0.99 * np.min(self.obb.mesh.bounds[:, -1])

        return np.linspace(z_max, z_min, self._zslice_num)


class ProximalSlices(Slices):
    def __init__(
        self,
        obb,
        surgical_neck,
        zslice_num=600,  # must not change needed for anp cnn
        interp_num=512,  # must not change needed for anp cnn
        return_odd=False,
    ):
        self.surgical_neck = surgical_neck
        super().__init__(
            obb,
            zslice_num,
            interp_num,
            return_odd,
        )

    @cached_property
    def _zs(self) -> np.ndarray:
        z_max = 0.99 * np.max(self.obb.mesh.bounds[:, -1])
        z_min = self.surgical_neck.neck_z

        return np.linspace(z_max, z_min, self._zslice_num)


class DistalSlices(Slices):
    def __init__(
        self,
        obb,
        zslice_num=200,
        interp_num=500,
        return_odd=False,
    ):
        super().__init__(
            obb,
            zslice_num,
            interp_num,
            return_odd,
        )

    @cached_property
    def _zs(self) -> np.ndarray:
        z_max = 0.99 * np.min(self.obb.mesh.bounds[:, -1])
        z_min = 0

        return np.linspace(z_max, z_min, self._zslice_num)
