from shoulder import utils
from shoulder.humerus import slice
from shoulder.base import Landmark, Transform

import plotly.graph_objects as go
import numpy as np
from skspatial.objects import Line, Points


class Canal(Landmark):
    def __init__(self, slc: slice.FullSlices, tfrm: Transform, proximal=False):
        """Calculates the centerline of the humeral canal"""
        self._slc = slc
        self._tfrm = tfrm
        self._proximal = proximal
        self._points_ct = None
        self._axis_ct = None

    def points(self, cutoff_pcts=(0.35, 0.75)) -> np.ndarray:
        """calculates all the centroids along the canal

        Args:
            cutoff_pcts (tuple): cutoff for where centerline is to be fit between i.e (0.2,0.8) -> middle 60% of the bone

            num_slices (int): number of slices to generate between cutoff points for which centroids will be calculated

        Returns:
            canal_pts_ct: 2x3 matrix of xyz points at ends of centerline
        """

        if self._points_ct is None:
            if self._proximal:
                if cutoff_pcts == (0.35, 0.75):  # if unchanged
                    cutoff_pcts = (
                        self._slc.obb.cutoff_pcts[0],
                        self._slc.obb.cutoff_pcts[1],
                    )
            self._cutoff_pcts = cutoff_pcts
            # centroids
            centroids = np.zeros((len(self._slc.zs(self._cutoff_pcts)), 3))
            for i, (s, z) in enumerate(
                zip(
                    self._slc.slices(self._cutoff_pcts), self._slc.zs(self._cutoff_pcts)
                )
            ):
                centroids[i] = np.r_[s.centroid, z]

            # transform back then record the centroids
            centroids_ct = utils.transform_pts(
                centroids, utils.inv_transform(self._slc.obb.transform)
            )
            self._points_ct = centroids_ct
            self._points_obb = centroids

        self._points = utils.transform_pts(self._points_ct, self._tfrm.matrix)
        return self._points

    def axis(self, cutoff_pcts=(0.35, 0.75)) -> np.ndarray:
        """calculates all the centroids along the canal and returns the first and last points of a line fit to the centroids
        cutoff_pcts will only update once future calculations are cached.
        """
        if self._axis_ct is None:
            if self._points_ct is None:
                self.points(cutoff_pcts)
            # calculate centerline
            canal_fit = Line.best_fit(Points(self._points_obb))
            canal_direction = canal_fit.direction
            canal_mdpt = canal_fit.point

            # ensure that the vector is pointed proximally
            if canal_fit.direction[-1] < 0:
                canal_direction = canal_direction * -1

            # repersent centerline as two points at the extents of the cutoff
            z_length_cutoff = self._slc.obb.z_length * np.mean(self._cutoff_pcts)
            canal_prox = canal_mdpt + (canal_direction * (z_length_cutoff / 2))
            canal_dstl = canal_mdpt - (canal_direction * (z_length_cutoff / 2))
            canal_pts = np.array([canal_prox, canal_dstl])
            canal_pts_ct = utils.transform_pts(
                canal_pts, utils.inv_transform(self._slc.obb.transform)
            )
            self._axis_ct = canal_pts_ct

        self._axis = utils.transform_pts(self._axis_ct, self._tfrm.matrix)
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
        z_hat = utils.unit_vector(self._axis[0], self._axis[1])
        # grab x axis from OBB csys
        x_hat = self._slc.obb.transform[:3, :1].flatten()

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

    def transform_landmark(self) -> None:
        if self._axis_ct is not None:
            self.axis()
        if self._points_ct is not None:
            self.points()

    def _graph_obj(self):
        if self._points_ct is None:
            return None
        else:
            plot = go.Scatter3d(
                x=self._points[:, 0],
                y=self._points[:, 1],
                z=self._points[:, 2],
                name="Canal Axis",
            )
            return plot
