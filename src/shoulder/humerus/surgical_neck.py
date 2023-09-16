from shoulder import utils
from shoulder.humerus import mesh
from shoulder.base import Landmark
from shoulder.humerus import slice

import plotly.graph_objects as go
from functools import cached_property
import numpy as np
import ruptures


class SurgicalNeck(Landmark):
    def __init__(self, slc: slice.FullSlices) -> None:
        self._slc = slc
        self.points_ct = self.points.copy()
        self.neck_z: float

    @cached_property
    def points(self):
        # predict location
        algo = ruptures.KernelCPD(kernel="rbf")
        algo.fit(self._slc.areas1)
        bkp = algo.predict(n_bkps=1)
        self.neck_z = self._slc.zs[bkp[0]]

        surgical_neck = self._slc.obb.mesh.section(
            plane_origin=[0, 0, self.neck_z], plane_normal=[0, 0, 1]
        )  # .discrete[0]
        if len(surgical_neck.entities) > 1:
            surgical_neck = surgical_neck.discrete[
                np.argmin(
                    [
                        np.sum(np.abs(np.mean(s[:, :2], axis=0)))
                        for s in surgical_neck.discrete
                    ]
                )
            ]
        else:
            surgical_neck = surgical_neck.discrete[0]

        surgical_neck_ct = utils.transform_pts(
            surgical_neck, utils.inv_transform(self._slc.obb.transform)
        )

        return surgical_neck_ct

    # keep for now
    def cutoff_zs(self, bottom_pct=0.35, top_pct=0.85):
        """given cutoff perccentages with 0 being the surgical neck and 1 being the
        top of the head return the z coordaintes
        """
        z_max = np.max(self._slc.obb.mesh.bounds[:, -1])

        surgical_neck_top_head = z_max - self.neck_z
        bottom = self.neck_z + (surgical_neck_top_head * bottom_pct)
        top = self.neck_z + (surgical_neck_top_head * top_pct)
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
                name="Surgical Neck",
            )
            return plot
