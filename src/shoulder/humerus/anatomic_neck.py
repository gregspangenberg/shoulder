from shoulder.humerus import slice
from shoulder.base import Landmark
from shoulder import utils
from shoulder.humerus import bicipital_groove

import plotly.graph_objects as go
import numpy as np
import skspatial.objects

from sklearn.preprocessing import MinMaxScaler
import importlib.resources
import onnxruntime as rt


class AnatomicNeck(Landmark):
    def __init__(self, slc: slice.Slices, bcptl: bicipital_groove.DeepGroove):
        self._slc = slc
        self._bcptl = bcptl
        self._points_ct = None
        self._points = None

    def points(self):
        if self._points is None:
            cutoff = (0.161, 0.8)  # not changeable
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
            with open(
                importlib.resources.files("shoulder") / "humerus/models/anp.onnx", "rb"
            ) as file:
                unet = rt.InferenceSession(
                    file.read(), providers=["CPUExecutionProvider"]
                )

            # get mask prediction
            input_name = unet.get_inputs()[0].name
            input_image = image.astype(np.float32).reshape(1, 1, 384, 512)
            mask = unet.run(None, {input_name: input_image})[0]

            # extract mask edge
            mask = np.squeeze(mask)
            mask = (mask > 0).astype(int)
            mask_edge = np.abs(np.diff(mask, prepend=0))
            mask_edge = mask_edge.astype(bool)

            # pull out theta and radius
            t = itr_shft[:, 0, :]
            r = itr_shft[:, 1, :]
            # setup zs to be same shape as t and r
            zs = np.repeat(zs.reshape(-1, 1), t.shape[1], axis=1)
            zs = zs[mask_edge]
            # grab the radial values that correspond to the edge of the mask
            t = t[mask_edge]
            r = r[mask_edge]
            x = r * np.cos(t)
            y = r * np.sin(t)

            # create array of points in obb space and transform to CT
            anp_points = np.c_[x, y, zs]
            anp_points = utils.transform_pts(
                anp_points, utils.inv_transform(self._slc.obb.transform)
            )

            self._points = anp_points
            self._points_ct = anp_points
        return self._points

    def transform_landmark(self, transform) -> None:
        if self._points is not None:
            self._points = utils.transform_pts(self._points_ct, transform)

    def _graph_obj(self):
        if self._points is None:
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
            ]

            return plot
