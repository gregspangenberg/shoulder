from shoulder.humerus import mesh
from shoulder.humerus import canal
from shoulder.base import Landmark
from shoulder import utils

import plotly.graph_objects as go
import shapely.affinity
import numpy as np
import itertools


class TransEpicondylar(Landmark):
    def __init__(self, obb: mesh.FullObb):
        self._mesh_oriented = obb.mesh
        self._transform = obb.transform
        self._axis_ct = None
        self._axis = None

    def axis(self, num_slices: int = 50):
        def medial_lateral_dist_multislice(msh_o, num_slices: int):
            """slices along distal humerus and computes the medial lateral distance with a rotated bounding box

            Args:
                msh_o (trimesh.mesh): rotated trimesh mesh object
                num_slices (int): number of slices to make between 10% and 0%

            """
            # get length of the tranformed bone
            total_length = np.sum(abs(msh_o.bounds[:, -1]))  # entire length of bone
            pos_length = msh_o.bounds[
                msh_o.bounds[:, -1] >= 0, -1
            ]  # bone in positive space, bone before the centerline midpoint

            proximal_cutoff = -0.8 * total_length + pos_length
            distal_cutoff = -0.99 * total_length + pos_length

            # spacing of cuts
            cuts = np.linspace(proximal_cutoff, distal_cutoff, num=num_slices).flatten()

            dist = []

            for cut in cuts:
                try:
                    path = msh_o.section(
                        plane_origin=[0, 0, cut], plane_normal=[0, 0, 1]
                    )
                    slice, to_3d = path.to_planar()
                except:
                    break

                # get shapely object from path
                polygon = slice.polygons_closed[0]

                # create rotated bounding box
                majr_dist = utils.major_axis_dist(
                    polygon.minimum_rotated_rectangle
                )  # maximize this distance
                dist.append(majr_dist)
            idx_max_dist = dist.index(max(dist))
            max_dist_cut = cuts[idx_max_dist]

            return max_dist_cut

        if self._axis is None:
            # find z distance where medial lateral distance is longest
            z_dist = medial_lateral_dist_multislice(self._mesh_oriented, num_slices)

            # slice at location of max medial-lateral distance
            path = self._mesh_oriented.section(
                plane_normal=[0, 0, 1], plane_origin=[0, 0, z_dist]
            )
            slice, to_3d = path.to_planar()

            # get shapely object from path
            polygon = slice.polygons_closed[0]

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
                end_geoms = list(
                    np.array(ab_dists)[np.argmax(np.array(ab_dists)[:, 2]), :2]
                )  # find location of max distance return shapely objs
                end_pts = np.array(
                    [end_geoms[0].centroid.xy, end_geoms[1].centroid.xy]
                ).reshape(2, 2)
            else:
                end_pts = np.array(
                    [ends.geoms[0].centroid.xy, ends.geoms[1].centroid.xy]
                ).reshape(2, 2)

            # add z distance back ins
            end_pts = utils.z_zero_col(end_pts)

            # transform back
            end_pts = utils.transform_pts(end_pts, to_3d)
            end_pts_ct = utils.transform_pts(
                end_pts, utils.inv_transform(self._transform)
            )

            self._axis_ct = end_pts_ct
            self._axis = end_pts_ct
        return self._axis

    def transform_landmark(self, transform) -> None:
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
