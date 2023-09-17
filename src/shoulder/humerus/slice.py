# from shoulder.humerus import surgical_neck #ideally imported but will infitinte loop
from shoulder.humerus import mesh

from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np


class Slices(ABC):
    def __init__(
        self, obb: mesh.Obb, cutoff_pcts: list, zslice_num: int, interp_num: int
    ):
        self._mesh_oriented_uobb = obb.mesh
        self.obb = obb
        self._cutoff_pcts = cutoff_pcts
        self._zslice_num = zslice_num
        self._interp_num = interp_num
        self._z_orig = np.mean(self.zs)
        self._z_incrs = self.zs - self._z_orig

    @cached_property
    def slices(self):
        # grab the polygon of the slice
        origin = [0, 0, self._z_orig]
        normal = [0, 0, 1]
        slices = self.obb.mesh.section_multiplane(
            plane_origin=origin, plane_normal=normal, heights=self._z_incrs
        )
        return slices

    @cached_property
    def areas1(self):
        area1 = np.zeros(len(self._z_incrs))
        for i, slice in enumerate(self.slices):
            if len(slice.entities) > 1:
                # keep only largest polygon if more than 1
                area1[i] = slice.polygons_closed[
                    np.argmax([p.area for p in slice.polygons_closed])
                ].area
            else:
                area1[i] = slice.area
        return area1

    @cached_property
    def ixy(self):
        # preallocate variables
        cart = np.zeros((len(self._z_incrs), 2, self._interp_num))
        for i, slice in enumerate(self.slices):
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

    @cached_property
    def irt(self):
        polar = np.zeros(self.ixy.shape)
        for i, p in enumerate(polar):
            polar[i] = self._cart2pol(self.ixy[i][0, :], self.ixy[i][1, :])
        return polar

    @cached_property
    @abstractmethod
    def zs(self) -> np.ndarray:
        """returns the z's over the cutoff interval"""

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


class FullSlices(Slices):
    def __init__(
        self,
        obb,
        cutoff_pcts=[0.35, 0.85],
        zslice_num=100,
        interp_num=100,
    ):
        super().__init__(obb, cutoff_pcts, zslice_num, interp_num)

    @cached_property
    def zs(self) -> np.ndarray:
        z_max = np.max(self.obb.mesh.bounds[:, -1])
        z_min = np.min(self.obb.mesh.bounds[:, -1])
        z_length = abs(z_max) + abs(z_min)  # goes across centerline
        low, high = self._cutoff_pcts
        low_z = z_min + low * z_length
        high_z = z_min + high * z_length

        # return np.linspace(low_z, high_z, self._zslice_num)
        return np.linspace(high_z, low_z, self._zslice_num)


class ProximalSlices(Slices):
    def __init__(
        self,
        obb,
        surgical_neck,
        cutoff_pcts=[0.35, 0.75],
        zslice_num=300,
        interp_num=1000,
    ):
        self.surgical_neck = surgical_neck
        super().__init__(obb, cutoff_pcts, zslice_num, interp_num)

    @cached_property
    def zs(self) -> np.ndarray:
        z_max = np.max(self.obb.mesh.bounds[:, -1])
        z_min = self.surgical_neck.neck_z
        z_length = abs(z_max) + abs(z_min)  # goes across centerline
        low, high = self._cutoff_pcts
        low_z = z_min + low * z_length
        high_z = z_min + high * z_length

        # return np.linspace(low_z, high_z, self._zslice_num)
        return np.linspace(high_z, low_z, self._zslice_num)
