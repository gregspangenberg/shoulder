from shoulder import utils
from shoulder.humerus import mesh
import numpy as np
import circle_fit
from skspatial.objects import Line, Points
import shapely
import itertools
from functools import cached_property


class Canal:
    def __init__(self, mesh: mesh.Obb):
        """Calculates the centerline of the humeral canal"""

        self._mesh_oriented_uobb = mesh.mesh
        self._transform_uobb = mesh.transform
        self._axis = None

    def axis(self, cutoff_pcts: list = [0.2, 0.8], num_slices: int = 50) -> np.ndarray:
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

        if self._axis is None:
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

            self._axis = canal_pts_ct
        return self._axis

    # get_transform method only needed for the first axis of a csys i.e. the independent axis
    def get_transform(self) -> np.ndarray:
        """Get transform from CT csys to a csys with z-axis as the canal.

        Args:
            canal_axis (np.ndarray): 2x3 matrix of xyz points at ends of centerline
            _transform_uobb (np.ndarray): transform matrix from the CT csys to the OBB csys

        Returns:
            np.ndarray: 4x4 transform matrix from the CT csys to the canal csys

        Take the canal as the z axis x axis of the OBB csys and project it onto a plane
        orthgonal to the canal axis, this will make the x axis othogonal to the canal axis.
        Then take the cross product to find the last axis. This creates a transform from the
        canal csys to the ct csys but we would like the opposite so invert it before returning the transform
        """
        # canal axis
        z_hat = utils.unit_vector(self.axis()[0], self.axis()[1])
        # grab x axis from OBB csys
        x_hat = self._transform_uobb[:3, :1].flatten()

        # project to be orthogonal
        x_hat -= z_hat * np.dot(x_hat, z_hat) / np.dot(z_hat, z_hat)
        x_hat /= np.linalg.norm(x_hat)

        # find last axis
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)

        # assemble
        pos = np.average(self._axis, axis=0)
        transform = np.c_[x_hat, y_hat, z_hat, pos]
        transform = np.r_[transform, np.array([0, 0, 0, 1]).reshape(1, 4)]

        # return a transform that goes form CT_csys -> Canal_csys
        transform = utils.inv_transform(transform)
        return transform