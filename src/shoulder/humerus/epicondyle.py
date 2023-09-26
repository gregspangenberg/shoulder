from shoulder.humerus import slice
from shoulder.base import Landmark
from shoulder import utils

import plotly.graph_objects as go
import shapely.affinity
import numpy as np
import itertools


class TransEpicondylar(Landmark):
    def __init__(self, slc: slice.Slices):
        self._slc = slc
        self._axis_ct = None
        self._axis = None

    def axis(self, num_slices: int = 50):
        if self._axis is None:
            # find z distance where medial lateral distance is longest

            dist = []
            cutoff = (0.8, 0.99)
            for s in self._slc.slices(cutoff):
                mrr = s.polygons_closed[0].minimum_rotated_rectangle
                dist.append(utils.major_axis_dist(mrr))
            idx_max_dist = dist.index(max(dist))
            slice_mrr_max = self._slc.slices(cutoff)[idx_max_dist]
            z_mrr_max = self._slc.zs(cutoff)[idx_max_dist]

            # get shapely object from path
            polygon = slice_mrr_max.polygons_closed[0]

            # create rotated bounding box
            bound = polygon.minimum_rotated_rectangle
            bound_angle = utils.azimuth(bound)

            # cut ends off at edge of bounding box that align with major axis
            bound_scale = shapely.affinity.rotate(bound, bound_angle)
            bound_scale = shapely.affinity.scale(bound_scale, xfact=1.5, yfact=0.999)
            bound_scale = shapely.affinity.rotate(bound_scale, -bound_angle)
            ends = polygon.difference(bound_scale)

            # now we have the most medial and lateral points
            # sometimes one of the end sections can be split in two leaving more than 2 total ends
            if len(list(ends.geoms)) > 2:
                ab_dists = []
                # iterate through all distance combos
                for a, b in itertools.combinations(list(ends.geoms), 2):
                    ab_dists.append(
                        [
                            a,
                            b,
                            utils._dist(
                                np.array(a.centroid.xy).flatten(),
                                np.array(b.centroid.xy).flatten(),
                            ),
                        ]
                    )  # [obj,obj,distance]
                # find location of max distance return shapely objs
                end_geoms = list(
                    np.array(ab_dists)[np.argmax(np.array(ab_dists)[:, 2]), :2]
                )
                end_pts = np.array(
                    [end_geoms[0].centroid.xy, end_geoms[1].centroid.xy]
                ).reshape(2, 2)
            else:
                end_pts = np.array(
                    [ends.geoms[0].centroid.xy, ends.geoms[1].centroid.xy]
                ).reshape(2, 2)

            # transform back
            end_pts = np.c_[end_pts, np.repeat(z_mrr_max, 2)]
            end_pts_ct = utils.transform_pts(
                end_pts, utils.inv_transform(self._slc.obb.transform)
            )

            self._axis_ct = end_pts_ct
            self._axis = end_pts_ct
        return self._axis

    def transform_landmark(self, transform) -> None:
        if self._axis is not None:
            self._axis = utils.transform_pts(self._axis_ct, transform)

    def _graph_obj(self):
        if self._axis is None:
            return None
        else:
            plot = go.Scatter3d(
                x=self._axis[:, 0],
                y=self._axis[:, 1],
                z=self._axis[:, 2],
                name="Transverse Epicondylar Axis",
            )
            return plot
