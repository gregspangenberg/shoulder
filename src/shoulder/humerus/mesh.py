from functools import cached_property
import warnings
from pathlib import Path
import trimesh
import circle_fit
import numpy as np
import scipy.signal
from abc import ABC, abstractmethod


import matplotlib.pyplot as plt


class MeshLoader:
    def __init__(self, stl_file) -> None:
        if not isinstance(stl_file, Path):
            stl_file = Path(stl_file)

        self.file = stl_file
        self.name = stl_file.stem

    @cached_property
    def _mesh_ct(self) -> trimesh.Trimesh:
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f"{self.name} is not watertight!")
        return m

    @property
    def mesh_ct(self):
        # immutable
        m = self._mesh_ct.copy()

        return m

    @cached_property
    def mesh(self):
        # mutable
        m = self._mesh_ct.copy()

        return m


class Obb(ABC, MeshLoader):
    # oriented bounding box transformation matrix (4x4)
    transform: np.ndarray
    # cutoff of bounding box for uneven cut of proximal humerus
    cutoff_pcts: list
    z_bounds: tuple
    z_length: float

    @abstractmethod
    def _obb(self) -> list:
        """calculates the oriented bouding box returns _mesh, _transform, _cutoff_pcts(optional)"""


class FullObb(Obb):
    def __init__(self, stl_file):
        super().__init__(stl_file)
        self.transform = self._obb()
        self.cutoff_pcts = [0.5, 0.8]

    def _obb(self):
        """rotates the humerus so the humeral head faces up (+ y-axis)

        Args:
            mesh (trimesh.mesh): mesh to rotate up

        Returns:
            mesh: rotated mesh
            ct_T: transform back to CT space
        """

        """ The center of volume is now at (0,0,0) with the y axis of the CSYS being the long axis of the humerus.
        The z being left-right and the x being up-down when viewed along the y-axis (humeral-axis)
        Whether the humeral head lies in +y space or -y space is unkown. The approach to discover which end is which
        is to take a slice on each end and see which shape is more circular. The more circular end is obviously the
        humeral head.
        """

        # apply oriented bounding box
        _transform_obb = self.mesh.apply_obb()  # modify in place returns transform

        # Get z bounds of box
        self.z_bounds = (self.mesh.bounds[0][-1], self.mesh.bounds[1][-1])
        self.z_length = abs(self.z_bounds[0]) + abs(self.z_bounds[1])

        # look at slice shape on each end
        humeral_end = 0  # y-coordinate default value
        residu_init = np.inf  # residual of circle fit default value
        for z_limit in self.z_bounds:
            # make the slice
            # move 5% inwards on the half, so 2.5% of total length
            z_slice = 0.95 * z_limit
            slice = self.mesh.section(
                plane_origin=[0, 0, z_slice], plane_normal=[0, 0, 1]
            )
            # returns the 2d view at plane and the transformation back to 3d space for each point
            slice, to_3d = slice.to_planar()

            # pull out the points along the shapes edge
            _, _, _, residu = circle_fit.least_squares_circle(np.array(slice.vertices))

            # 1st pass, less than inf record, 2nd pass if less than 1st
            if residu < residu_init:
                residu_init = residu
                humeral_end = z_limit

        # if the y-coordinate of the humeral head is in negative space then
        # we are looking to see if a flip was performed and if it was needed
        # humeral_end is a set containing (y-coordinate, residual from circle fit)
        if humeral_end < 0:
            # flip was reversed so update the ct_transform to refelct that
            transform_flip = np.array(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            self.mesh.apply_transform(transform_flip)
        else:
            transform_flip = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

        # add in flip that perhaps occured
        _transform = np.matmul(transform_flip, _transform_obb)
        return _transform


class ProxObb(Obb):
    def __init__(self, stl_file):
        super().__init__(stl_file)
        self.transform, self.cutoff_pcts = self._obb()

    def _obb(self):
        """to determine which side is the humeral head and which side is the cut shaft is a
        simple comparison of area at each end. The cut is not always clean and is sometimes angled
        which presents some difficulties.
        To overcome this get the area of 100 points along the z axis of the obb. If the area becomes smaller
        towards the end remove if from consideration."""

        def consecutive(arr):
            return max(np.split(arr, (np.where(np.diff(arr) != 1)[0] + 1)), key=len)

        # apply oriented bounding box
        _transform_obb = self.mesh.apply_obb()  # modify in place returns transform

        # Get z bounds of box
        self.z_bounds = (self.mesh.bounds[0][-1], self.mesh.bounds[1][-1])
        self.z_length = abs(self.z_bounds[0]) + abs(self.z_bounds[1])

        # find largest area along z axis
        inset_factor = 0.99  # percent shrink z of first slice
        # evenly space z intervals
        num_zs = 100
        z_intervals = np.linspace(
            self.z_bounds[0] * inset_factor, self.z_bounds[1] * inset_factor, num_zs
        ).flatten()
        z_area = []
        for z in z_intervals:
            slice = self.mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            slice, to_3d = slice.to_planar()
            z_area.append(slice.area)

        # the middle of humeral head has the largest area for proximal humerus
        humeral_head_z = z_intervals[np.argmax(z_area)]

        # flip so the humeral head is up
        if humeral_head_z < 0:
            transform_flip = np.array(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            self.mesh.apply_transform(transform_flip)
            # reverse the order now that it has been flipped
            z_area = z_area[::-1]
        else:
            transform_flip = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
        # add in flip that perhaps occured
        _transform = np.matmul(transform_flip, _transform_obb)

        # calculate the difference between one slice and the next but smoothed
        grad_z_area = np.gradient(scipy.signal.savgol_filter(z_area, 3, 1))

        # keep gradients smaller than a diff of 5, these must be the canal as it changes little in area
        # this will also remove the improperly cut portion
        canal_zs = consecutive(np.where(grad_z_area < 10)[0])
        self.cutoff_bot = canal_zs[0]

        # cutoff percentages for when canal needs to be found
        cutoff_pcts = [canal_zs[0] / num_zs, canal_zs[-1] / num_zs]

        return _transform, cutoff_pcts
