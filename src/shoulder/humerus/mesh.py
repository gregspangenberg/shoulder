from functools import cached_property, lru_cache
import warnings
from pathlib import Path
import trimesh
import circle_fit
import numpy as np


class MeshLoader:
    def __init__(self, stl_file) -> None:
        self.file = Path(stl_file)
        self.name = Path(stl_file).stem

    @cached_property
    def mesh_ct(self):
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f"{self.name} is not watertight!")
        return m


class MeshObb(MeshLoader):
    def __init__(self, stl_file):
        super().__init__(stl_file)

        # self.mesh, self.transform = self.transform_obb(self.mesh_ct)

    @property
    def mesh(self) -> trimesh.Trimesh:
        return self._transform_obb[0]

    @property
    def transform(self) -> np.ndarray:
        return self._transform_obb[1]

    @cached_property
    def _transform_obb(self):
        print("did")
        """rotates the humerus so the humeral head faces up (+ y-axis)

        Args:
            mesh (trimesh.mesh): mesh to rotate up

        Returns:
            mesh: rotated mesh
            flip_y_T: transform to flip axis if it was performed
            ct_T: transform back to CT space
        """

        """ The center of volume is now at (0,0,0) with the y axis of the CSYS being the long axis of the humerus.
        The z being left-right and the x being up-down when viewed along the y-axis (humeral-axis)
        Whether the humeral head lies in +y space or -y space is unkown. The approach to discover which end is which
        is to take a slice on each end and see which shape is more circular. The more circular end is obviously the
        humeral head.
        """

        # apply oriented bounding box
        _mesh = self.mesh_ct.copy()
        _transform_obb = _mesh.apply_obb()  # modify in place returns transform

        # Get z bounds of box
        y_limits = (_mesh.bounds[0][-1], _mesh.bounds[1][-1])

        # look at slice shape on each end

        humeral_end = 0  # y-coordinate default value
        residu_init = np.inf  # residual of circle fit default value
        for y_limit in y_limits:
            # make the slice
            # move 5% inwards on the half, so 2.5% of total length
            y_slice = 0.95 * y_limit
            slice = _mesh.section(plane_origin=[0, 0, y_slice], plane_normal=[0, 0, 1])
            # returns the 2d view at plane and the transformation back to 3d space for each point
            slice, to_3d = slice.to_planar()

            # pull out the points along the shapes edge
            _, _, _, residu = circle_fit.least_squares_circle(np.array(slice.vertices))

            # 1st pass, less than inf record, 2nd pass if less than 1st
            if residu < residu_init:
                humeral_end = y_limit

        # if the y-coordinate of the humeral head is in negative space then
        # we are looking to see if a flip was performed and if it was needed
        # humeral_end is a set containing (y-coordinate, residual from circle fit)
        if humeral_end < 0:
            # print("flipped")
            # flip was reversed so update the ct_transform to refelct that
            transform_flip_y = np.array(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            _mesh.apply_transform(transform_flip_y)
        else:
            # print("not flipped")
            transform_flip_y = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

        # add in flip that perhaps occured
        _transform = np.matmul(transform_flip_y, _transform_obb)

        return _mesh, _transform
