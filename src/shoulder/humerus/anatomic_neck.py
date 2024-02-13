from shoulder.humerus import slice
from shoulder.base import Landmark, Transform
from shoulder import utils
from shoulder.humerus import bicipital_groove

import plotly.graph_objects as go
import numpy as np
import skspatial.objects
from ellipse import LsqEllipse
import trimesh.geometry


from sklearn.preprocessing import MinMaxScaler
import importlib.resources
import onnxruntime as rt


class AnatomicNeck(Landmark):
    def __init__(
        self, slc: slice.Slices, bcptl: bicipital_groove.DeepGroove, tfrm: Transform
    ):
        self._slc = slc
        self._bcptl = bcptl
        self._tfrm = tfrm
        self._points_ct = None
        self._plane_ct = None
        self._plane_points_ct = None
        self._central_axis_ct = None
        self._normal_axis_ct = None

    def points(self) -> np.ndarray:
        """calculate all the points along the anatomic neck"""
        if self._points_ct is None:
            cutoff = (0.0, 0.852)  # not changeable
            itr = self._slc.itr_start(cutoff)
            zs = self._slc.zs(cutoff)

            image = np.zeros((itr.shape[0], itr.shape[2]))
            itr_shft = np.zeros(itr.shape)
            for i, (z, tr) in enumerate(zip(zs, itr)):
                # interpolate on even theta
                # sometimes the last is also the first which creates artifacts
                t_sampling = np.linspace(tr[0][0], tr[0][-2], tr.shape[1])
                tr = np.c_[t_sampling, np.interp(t_sampling, tr[0, :-1], tr[1, :-1])].T

                # shift to starting at bicipital groove
                self._bcptl.axis()  # force the calculation of the bicipital groove if not done yet
                closest_bg_idx = np.argmin(np.abs(tr[0] - self._bcptl.bg_theta))
                tr = np.c_[tr[:, closest_bg_idx:], tr[:, :closest_bg_idx]]

                # add to image
                image[i] = tr[1]
                # add to itr shifted
                itr_shft[i] = tr

            image_shape = image.shape
            image = MinMaxScaler().fit_transform(image.reshape(-1, 1))
            image = image.reshape(image_shape)

            # open random forest saved in onnx
            # Unet_CRF_fil9 better in arthritic worse in non-arthritic
            with open(
                importlib.resources.files("shoulder")
                / "humerus/models/unetcrf_anp.onnx",
                "rb",
            ) as file:
                unet = rt.InferenceSession(
                    file.read(), providers=["CPUExecutionProvider"]
                )

            # get mask prediction
            input_name = unet.get_inputs()[0].name
            input_image = image.astype(np.float32).reshape(
                1, 1, image_shape[0], image_shape[1]
            )
            mask = unet.run(None, {input_name: input_image})[0]

            # extract mask edge
            mask = np.squeeze(mask)  # transform  (1,1,512,512) -> (512,512)
            # mask = mask[1, :, :]  # for b loss models
            # mask = (mask > 0.5).astype(int)  # for b loss models
            mask = (mask > 0).astype(int)  # for h loss models
            mask_edge = np.abs(np.diff(mask, prepend=0))
            mask_edge = mask_edge.astype(bool)
            mask = mask.astype(bool)

            # pull out theta and radius
            t = itr_shft[:, 0, :]
            r = itr_shft[:, 1, :]
            # setup zs to be same shape as t and r
            zs = np.repeat(zs.reshape(-1, 1), t.shape[1], axis=1)

            # grab the radial values that correspond to the edge of the mask
            zs_e = zs[mask_edge]
            t_e = t[mask_edge]
            r_e = r[mask_edge]
            x_e = r_e * np.cos(t_e)
            y_e = r_e * np.sin(t_e)
            # create array of points in obb space
            anp_points = np.c_[x_e, y_e, zs_e]
            self._points_obb = anp_points  # needed to calc axis, plane etc.

            # grab all values on the segmented articular surface
            zs_a = zs[mask]
            t_a = t[mask]
            r_a = r[mask]
            x_a = r_a * np.cos(t_a)
            y_a = r_a * np.sin(t_a)
            articular_all_points = np.c_[
                x_a, y_a, zs_a
            ]  # needed to calc radius of curva
            self._points_all_articular_obb = articular_all_points

            # transform from OBB to CT space
            anp_points = utils.transform_pts(
                anp_points, utils.inv_transform(self._slc.obb.transform)
            )
            self._points_ct = anp_points

        self._points = utils.transform_pts(self._points_ct, self._tfrm.matrix)
        return self._points

    def plane(self) -> skspatial.objects.Plane:
        """calculate the anatomic neck plane"""
        if self._plane_ct is None:
            self.points()  # calculate landmark if not yet calculated

            plane = skspatial.objects.Plane.best_fit(self._points_obb)
            normal = plane.normal.copy()
            # ensure normal vector is pointed up
            if normal[-1] < 0:
                normal *= -1

            # the centroid of the anp is too low as the CNN output doesn't included the top 20%
            # of the humerus. To correct for that an ellipse should be fit to the points and then
            # the center of the ellipse will become the new plane centroid

            # find transform to plane csys
            to_2D = trimesh.geometry.plane_transform(plane.point, normal)
            pts_2d = utils.transform_pts(self._points_obb, to_2D)
            center, _, _, _ = LsqEllipse().fit(pts_2d[:, :-1]).as_parameters()
            # add 0 and repeat so there are multiple points which is needed for transformation function
            center = np.repeat(np.r_[center, 0].reshape(1, 3), 2, axis=0)
            center = utils.transform_pts(center, np.linalg.inv(to_2D))[0]
            # construct new plane
            self._plane_sk_obb = skspatial.objects.Plane(center, normal)

            self._plane_ct = utils.transform_plane(
                self._plane_sk_obb, utils.inv_transform(self._slc.obb.transform)
            )

        self._plane = utils.transform_plane(self._plane_ct, self._tfrm.matrix)
        return self._plane

    def plane_points(self) -> np.ndarray:
        """calculate the anatomic neck plane and return the points which intersect the bone"""
        if self._plane_points_ct is None:
            self.plane()  # calculate landmark if not yet calculated

            plane_pts = np.array(
                self._slc.obb.mesh_ct.section(
                    plane_origin=self._plane_ct.point,
                    plane_normal=self._plane_ct.normal,
                ).vertices
            )

            self._plane_points_ct = plane_pts

        self._plane_points = utils.transform_pts(
            self._plane_points_ct, self._tfrm.matrix
        )
        return self._plane_points

    def axis_normal(self) -> np.ndarray:
        """calculate the anatomic neck plane normal and return the upper and lower points which intersects the bone"""
        if self._normal_axis_ct is None:
            if self._plane_ct is None:
                self.plane()  # calculate landmark if not yet calculated

            nrml = self._plane_sk_obb.normal.copy()
            if nrml[2] < 0:
                nrml *= -1

            upper_loc, _, _ = self._slc._mesh_oriented_uobb.ray.intersects_location(
                ray_origins=self._plane_sk_obb.point.reshape(-1, 3),
                ray_directions=nrml.reshape(-1, 3),
            )
            bottom_loc, _, _ = self._slc._mesh_oriented_uobb.ray.intersects_location(
                ray_origins=self._plane_sk_obb.point.reshape(-1, 3),
                ray_directions=-1 * nrml.reshape(-1, 3),
            )
            nrml_endpts = np.r_[upper_loc, bottom_loc]

            nrml_endpts = utils.transform_pts(
                nrml_endpts, utils.inv_transform(self._slc.obb.transform)
            )
            self._normal_axis_ct = nrml_endpts

        self._normal_axis = utils.transform_pts(self._normal_axis_ct, self._tfrm.matrix)
        return self._normal_axis

    def axis_central(self) -> np.ndarray:
        """calculate the head central axis from the anatomic neck normal and return the upper and lower points which intersects the bone"""
        if self._central_axis_ct is None:
            if self._plane_ct is None:
                self.plane()  # calculate landmark if not yet calculated

            # ensure normal is pointed upright
            nrml = self._plane_sk_obb.normal.copy()
            if nrml[2] < 0:
                nrml *= -1

            # remove z component and return to unit vecotr
            nrml[2] = 0
            nrml = nrml / np.linalg.norm(nrml)

            upper_loc, _, _ = self._slc._mesh_oriented_uobb.ray.intersects_location(
                ray_origins=self._plane_sk_obb.point.reshape(-1, 3),
                ray_directions=nrml.reshape(-1, 3),
            )
            bottom_loc, _, _ = self._slc._mesh_oriented_uobb.ray.intersects_location(
                ray_origins=self._plane_sk_obb.point.reshape(-1, 3),
                ray_directions=-1 * nrml.reshape(-1, 3),
            )
            nrml_endpts = np.r_[
                upper_loc, bottom_loc
            ]  # must return upper first transepi relies upon this
            cntrl = utils.transform_pts(
                nrml_endpts, utils.inv_transform(self._slc.obb.transform)
            )
            self._central_axis_ct = cntrl

        self._central_axis = utils.transform_pts(
            self._central_axis_ct, self._tfrm.matrix
        )
        return self._central_axis

    def transform_landmark(self) -> None:
        if self._points_ct is not None:
            self.points()
        if self._plane_ct is not None:
            self.plane()
        if self._plane_points_ct is not None:
            self.plane_points()
        if self._normal_axis_ct is not None:
            self.axis_normal()
        if self._central_axis_ct is not None:
            self.axis_central()

    def _graph_obj(self):
        if self._points_ct is None:
            return None
        else:
            plot = [
                go.Scatter3d(
                    x=self._points[:, 0],
                    y=self._points[:, 1],
                    z=self._points[:, 2],
                    mode="markers",
                    showlegend=True,
                    name="Anatomic Neck",
                ),
                go.Scatter3d(
                    x=self.plane_points()[:, 0],
                    y=self.plane_points()[:, 1],
                    z=self.plane_points()[:, 2],
                    mode="markers",
                    showlegend=True,
                    name="Anatomic Neck Plane",
                ),
            ]

            return plot
