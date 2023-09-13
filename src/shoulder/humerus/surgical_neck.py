from shoulder import utils
from shoulder.base import Landmark

import plotly.graph_objects as go
from functools import cached_property
import numpy as np
import ruptures


class SurgicalNeck(Landmark):
    def __init__(self, obb) -> None:
        self._mesh_oriented_uobb = obb.mesh
        self._transform_uobb = obb.transform
        self._obb_cutoff_pcts = obb.cutoff_pcts
        self.points_ct = self.points

    @cached_property
    def points(self):
        # this basically calcuates where the surgical neck is
        z_max = np.max(self._mesh_oriented_uobb.bounds[:, -1])
        z_min = np.min(self._mesh_oriented_uobb.bounds[:, -1])
        z_length = abs(z_max) + abs(z_min)

        z_low_pct = self._obb_cutoff_pcts[0]
        z_high_pct = self._obb_cutoff_pcts[1]
        obb_distal_cutoff = z_low_pct * z_length + z_min
        obb_proximal_cutoff = z_high_pct * z_length + z_min

        z_intervals = np.linspace(obb_distal_cutoff, 0.99 * z_max, 100)

        z_area = np.zeros(len(z_intervals))
        for i, z in enumerate(z_intervals):
            slice = self._mesh_oriented_uobb.section(
                plane_origin=[0, 0, z], plane_normal=[0, 0, 1]
            )
            slice, to_3d = slice.to_planar()
            # big_poly = slice.polygons_closed[
            #     np.argmax([p.area for p in slice.polygons_closed])
            # ]
            z_area[i,] = slice.area

        algo = ruptures.KernelCPD(kernel="rbf")
        algo.fit(z_area)
        bkp = algo.predict(n_bkps=1)
        self.surgical_neck_z = z_intervals[bkp[0]]

        surgical_neck = self._mesh_oriented_uobb.section(
            plane_origin=[0, 0, self.surgical_neck_z], plane_normal=[0, 0, 1]
        ).discrete[0]
        surgical_neck_ct = utils.transform_pts(
            surgical_neck, utils.inv_transform(self._transform_uobb)
        )

        return surgical_neck_ct

    def cutoff_zs(self, bottom_pct=0.35, top_pct=0.85):
        """given cutoff perccentages with 0 being the surgical neck and 1 being the
        top of the head return the z coordaintes
        """
        z_max = np.max(self._mesh_oriented_uobb.bounds[:, -1])

        surgical_neck_top_head = z_max - self.surgical_neck_z
        bottom = self.surgical_neck_z + (surgical_neck_top_head * bottom_pct)
        top = self.surgical_neck_z + (surgical_neck_top_head * top_pct)
        return [bottom, top]

    def transform_landmark(self, transform) -> None:
        if self.points is not None:
            self.points = utils.transform_pts(self.points_ct, transform)

    def _graph_obj(self):
        if self.points is None:
            return None

        else:
            plot = go.Scatter3d(
                x=self.points[:, 0],
                y=self.points[:, 1],
                z=self.points[:, 2],
                name="Bicipital Groove",
            )
            return plot
