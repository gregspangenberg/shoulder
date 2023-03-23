from shoulder import utils

import numpy as np
import circle_fit
from skspatial.objects import Line, Points
import shapely
import itertools
from functools import cached_property


class MeshObb:
    def __init__(self, mesh):
        self.mesh = mesh
        self.mesh_oriented = self.mesh.copy()
        # apply_obb will alter self.mesh_oriented
        self._transform_obb = self.mesh_oriented.apply_obb()

    @cached_property
    def transform(self):
        """rotates the humerus so the humeral head faces up (+ y-axis)

        Args:
            mesh (trimesh.mesh: mesh to rotate up

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
        # Get z bounds of box
        y_limits = (self.mesh_oriented.bounds[0][-1], self.mesh_oriented.bounds[1][-1])

        # look at slice shape on each end
        humeral_end = (
            0,
            np.inf,
        )  # (y_coordinate, residual_of_circle_fit), there is perhaps a better way of recording data
        for y_limit in y_limits:
            # make the slice
            y_slice = (
                0.95 * y_limit
            )  # move 5% inwards on the half, so 2.5% of total length
            slice = self.mesh_oriented.section(
                plane_origin=[0, 0, y_slice], plane_normal=[0, 0, 1]
            )
            # returns the 2d view at plane and the transformation back to 3d space for each point
            slice, to_3d = slice.to_planar()

            # pull out the points along the shapes edge
            slice_pts = np.array(slice.vertices)
            xc, yc, r, residu = circle_fit.least_squares_circle(slice_pts)

            # 1st pass, less than inf record, 2nd pass if less than 1st
            if residu < humeral_end[1]:
                humeral_end = (y_limit, residu)

        # if the y-coordinate of the humeral head is in negative space then
        # we are looking to see if a flip was performed and if it was needed
        # humeral_end is a set containing (y-coordinate, residual from circle fit)
        if humeral_end[0] < 0:
            # print("flipped")
            # flip was reversed so update the ct_transform to refelct that
            transform_flip_y = np.array(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            self.mesh_oriented.apply_transform(transform_flip_y)
        else:
            # print("not flipped")
            transform_flip_y = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

        # add in flip that perhaps occured
        transform = np.matmul(transform_flip_y, self._transform_obb)

        return transform


class Canal:
    def __init__(self, mesh: MeshObb):
        self._mesh_oriented_uobb = mesh.mesh_oriented
        self._transform_uobb = mesh.transform

    def axis(self, cutoff_pcts: list, num_slices: int):
        """calculates the centerline in region of humerus

        Args:
            cutoff_pcts (list): cutoff for where centerline is to be fit between i.e [0.2,0.8] -> middle 60% of the bone
            num_slices (int): number of slices to generate between cutoff points for which centroids will be calculated

        Returns:
            canal_pts_ct: 2x3 matrix of xyz points at ends of centerline
        """

        def axial_centroids(msh_o, cutoff_pcts, num_centroids):
            """Slices bone along long axis between the specified cutoff points and returns
            the centroids along the length

            Args:
                mesh (trimesh.mesh): trimesh mesh object to find centroids along length
                cutoff_pcts (list): list of two cutoff percentages i.e [0.2,0.8] would remove the upper 20% and lower 20%
                num_centroids (int): number of slices beween cutoff points to calculate centroids for

            Returns:
                centroids (np.array): array of xyz points for centroids along length
                cutoff_length (float): length between the cutoff percentages on the bone
            """

            # get length of the bone
            y_length = 2 * (
                abs(msh_o.bounds[0][-1])
            )  # mesh centered at 0, multiply by 2 to get full length along humeral canal

            # find distance that the cutoff percentages are at
            cutoff_pcts.sort()  # ensure bottom slice pct is first
            distal_cutoff = cutoff_pcts[0] * y_length - (
                y_length / 2
            )  # pct of total y-length then subtract to return center to 0
            proximal_cutoff = cutoff_pcts[1] * y_length - (y_length / 2)
            # length between cutoff pts
            cutoff_length = abs(proximal_cutoff - distal_cutoff)

            # spacing of cuts
            cuts = np.linspace(distal_cutoff, proximal_cutoff, num=num_centroids)

            centroids = []  # record data
            for cut in cuts:
                slice = msh_o.section(plane_origin=[0, 0, cut], plane_normal=[0, 0, 1])
                centroids.append(np.array(slice.centroid).reshape(1, 3))

            centroids = np.concatenate(centroids, axis=0)

            return centroids, cutoff_length

        # slice it !
        centroids, cutoff_length = axial_centroids(
            self._mesh_oriented_uobb, cutoff_pcts, num_slices
        )

        # calculate centerline
        canal_fit = Line.best_fit(Points(centroids))
        canal_direction = canal_fit.direction
        canal_mdpt = canal_fit.point

        # ensure that the vector is pointed proximally
        if canal_fit.direction[-1] < 0:
            canal_direction = canal_direction * -1

        # repersent centerline as two points at the extents of the cutoff
        canal_prox = canal_mdpt + (canal_direction * (cutoff_length / 2))
        canal_dstl = canal_mdpt - (canal_direction * (cutoff_length / 2))
        canal_pts = np.array([canal_prox, canal_dstl])
        canal_pts_ct = utils.transform_pts(
            canal_pts, utils.inv_transform(self._transform_uobb)
        )

        return canal_pts_ct


class MeshCanal:
    def __init__(self, mesh: MeshObb, canal_axis):
        self._mesh = mesh.mesh
        self._mesh_uobb = mesh.mesh_oriented
        self._transform_uobb = mesh.transform
        self.canal_axis = canal_axis

    @cached_property
    def transform(self):
        """Construct a coordinate system for the canal. Take the x axis of the OBB csys
        and project it onto a plane orthgonal to the canal axis, this will make the x axis
        orthogonal to the canal axis. Then take the cross product to find the last axis.
        This creates a transform from the canal csys to the ct csys but we would like the opposite
        so invert it before returning the transform"""
        # canal axis
        z_hat = utils.unit_vector(self.canal_axis[0], self.canal_axis[1])
        # grab x axis from OBB csys
        x_hat = self._transform_uobb[:3, :1].flatten()

        # project to be orthogonal
        x_hat -= z_hat * np.dot(x_hat, z_hat) / np.dot(z_hat, z_hat)
        x_hat /= np.linalg.norm(x_hat)

        # find last axis
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)

        # assemble
        pos = np.average(self.canal_axis, axis=0)
        transform = np.c_[x_hat, y_hat, z_hat, pos]
        transform = np.r_[transform, np.array([0, 0, 0, 1]).reshape(1, 4)]

        # return a transform that goes form CT_csys -> Canal_csys
        transform = utils.inv_transform(transform)
        return transform

    @property
    def mesh_oriented(self):
        mesh_orient = self._mesh.copy()
        mesh_orient = mesh_orient.apply_transform(self.transform)
        return mesh_orient


class TransEpicondylar:
    def __init__(self, mesh: MeshCanal):
        self._mesh_oriented = mesh.mesh_oriented
        self._transform = mesh.transform

    def axis(self, num_slices: int):
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
            cuts = np.linspace(proximal_cutoff, distal_cutoff, num=num_slices)

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
        end_pts_ct = utils.transform_pts(end_pts, utils.inv_transform(self._transform))

        return end_pts_ct
