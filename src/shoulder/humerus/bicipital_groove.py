from shoulder import utils
from shoulder.base import Landmark, Transform
from shoulder.humerus import slice

import numpy as np
import math
import scipy.signal
import skspatial.objects
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.neighbors

import importlib.resources
import onnxruntime as rt


class DeepGroove(Landmark):
    def __init__(self, slc: slice.Slices, canal, tfrm: Transform):
        self._slc = slc
        self._canal_axis = canal.axis()
        self._tfrm = tfrm
        self._points_ct = None
        self._axis_ct = None

    def points(self, cutoff_pcts=(0.2, 0.75), deg_window=7) -> np.ndarray:
        """calculate the points that lie along the bicipital groove"""

        def _X_process(polar, polar_0, zs):
            """creation of input array for random forest classifier"""

            def closest_angles(array, v):
                """create feature"""
                angs = []
                for a in array:
                    angs.append(math.atan2(math.sin(v - a), math.cos(v - a)))
                return np.abs(angs)

            def peak_nearest(all_peaks_theta):
                """create feature"""
                angles = []
                if len(all_peaks_theta) == 1:
                    return np.array([0])
                for p in all_peaks_theta:
                    angs = closest_angles(all_peaks_theta, p)
                    angs = angs[np.round(angs, 2) != 0]
                    angs.sort()
                    angles.append(angs[0])

                return np.array(angles)

            def peak_next_nearest(all_peaks_theta):
                """create feature"""
                angles = []
                if len(all_peaks_theta) == 1:
                    return np.array([0])
                if len(all_peaks_theta) == 2:
                    return np.array([0, 0])
                for p in all_peaks_theta:
                    angs = closest_angles(all_peaks_theta, p)
                    angs = angs[np.round(angs, 2) != 0]
                    angs.sort()
                    angles.append(angs[1])

                return np.array(angles)

            def canal_dist(self, theta_peaks, radius_peaks, z_peaks):
                """create feature"""
                # repeat z_peaks for the length of the peaks
                z_peaks = np.repeat(z_peaks, len(theta_peaks))

                canal_u = utils.unit_vector(self._canal_axis[0], self._canal_axis[1])
                # get canal points at the z heights
                canal_pts = canal_u.reshape(-1, 1) @ z_peaks.reshape(1, -1)
                canal_pts = canal_pts[:2, :]  # remove z
                peak_pts = _pol2cart(np.c_[theta_peaks, radius_peaks]).T

                dist = peak_pts - canal_pts
                dist = np.sqrt(np.sum(dist**2, axis=0))  # get distance

                return dist

            def theta_zstd(polar, peak):
                """create feature"""
                allradi_atpeak = polar[:, 1, peak]

                return allradi_atpeak.flatten().std()

            z_scale = MinMaxScaler().fit_transform(zs.reshape(-1, 1)).flatten()
            peak_zs = []
            peak_theta = []
            peak_radius = []
            peak_near = []
            peak_next_near = []
            peak_prom = []
            peak_width = []
            peak_widthheight = []
            peak_canal_dist = []
            peak_num = []
            peak_zstd = []

            for i, (rpol, rpol0) in enumerate(zip(polar, polar_0)):
                theta = rpol0[0]
                radius = rpol0[1]
                radius_og = rpol[1]
                radius = -1 * radius
                radius = scipy.signal.savgol_filter(radius, 10, 1)

                # sometimes the start or end contains a peak we need to shift
                rmin = -1 * np.argmin(radius)
                radius_roll = np.roll(radius, rmin)
                # now find peaks
                peaks, _prop = scipy.signal.find_peaks(
                    radius_roll,
                    height=-10,
                    prominence=0.6,
                    width=0.1,
                )
                peaks = (peaks - rmin) % self._slc._interp_num

                # if there are more than 10 peaks discard the lowest prominence one
                n = 7
                if len(peaks) > n:
                    part = np.argpartition(_prop["prominences"], -n)[-n:]
                    peaks = peaks[part]  # top n largest

                    for k, v in _prop.items():
                        _prop[k] = [v[i] for i in part]

                peak_theta.extend(theta[peaks])
                peak_radius.extend(radius_og[peaks])
                peak_near.extend(peak_nearest(theta[peaks]))
                peak_next_near.extend(peak_next_nearest(theta[peaks]))
                peak_zs.extend([z_scale[i]] * len(peaks))
                peak_prom.extend(_prop["prominences"])
                peak_width.extend(_prop["widths"])
                peak_widthheight.extend(_prop["width_heights"])
                peak_canal_dist.extend(
                    canal_dist(self, theta[peaks], radius_og[peaks], zs[i])
                )
                peak_num.extend(np.repeat((len(peaks) / n), len(peaks)))
                peak_zstd.extend([theta_zstd(polar, p) for p in peaks])

            X = np.c_[
                peak_radius,
                peak_near,
                peak_next_near,
                peak_zs,
                peak_prom,
                peak_width,
                peak_widthheight,
                peak_canal_dist,
                peak_num,
            ]

            X = StandardScaler().fit_transform(X)

            return X, np.array(peak_theta), np.array(peak_zs), np.array(peak_num)

        if self._points_ct is None:
            polar = self._slc.itr_centered_start(cutoff_pcts)
            zs = self._slc.zs(cutoff_pcts)

            # make each radial slice stationary by subtracting the mean
            polar_0 = polar.copy()
            polar_0[:, 1, :] = np.apply_along_axis(
                lambda x: x - np.mean(x), axis=1, arr=polar[:, 1, :]
            )

            # preprocess the data to get in the correct format
            self._X, self._peak_theta, _, _ = _X_process(polar, polar_0, zs)

            # open random forest saved in onnx
            with open(
                importlib.resources.files("shoulder") / "humerus/models/rfc_bg3.onnx",
                "rb",
            ) as file:
                clf = rt.InferenceSession(
                    file.read(), providers=["CPUExecutionProvider"]
                )
            pred_proba = clf.run(None, {"X": self._X})[1]

            # apply activation kernel
            kde = sklearn.neighbors.KernelDensity(kernel="linear")
            kde.fit(self._peak_theta[pred_proba[:, 1] > 0.4].reshape(-1, 1))
            tlin = np.linspace(-1 * np.pi, np.pi, 1024).reshape(-1, 1)
            bg_prob = np.exp(kde.score_samples(tlin))
            self.bg_theta = tlin[np.argmax(bg_prob)][0]

            # get local minima by specifying serach window for
            # search up to 15 degrees away on each side
            ivar = int(round(deg_window / (360 / self._slc._interp_num)))

            if ivar < 1:
                ivar = 1
            bg_xyz = np.zeros((len(zs), 3))
            bg_xy = np.zeros((len(zs), 2))
            bg_local_theta = np.zeros((len(zs), 1))
            for i, z in enumerate(zs):
                bg_i_esti = _find_nearest_idx(polar_0[i, 0, :].flatten(), self.bg_theta)

                # sometimes the degree variance will be higher than than the index bg is found at
                # when this occurs the indexing will start with a negative numebr causing it to fail
                # basically a wrap around problem
                if ivar > bg_i_esti:
                    bg_range = np.concatenate(
                        (
                            polar_0[i, :, (bg_i_esti - ivar) :],
                            polar_0[i, :, : (bg_i_esti + ivar)],
                        ),
                        axis=1,
                    )
                else:
                    bg_range = polar_0[
                        i,
                        :,
                        (bg_i_esti - ivar) : (bg_i_esti + ivar),
                    ]
                bg_i_local = np.argmin(bg_range[1, :])
                # transform back to radial coordinates
                bg_i_local = bg_i_local + (bg_i_esti - ivar)  # put back in context
                bg_local_theta[i, :] = polar[i, 0, bg_i_local]
                bg_xy[i, :] = _pol2cart(
                    polar[
                        i,
                        :,
                        bg_i_local,
                    ].reshape(1, 2)
                )
            bg_xyz = np.c_[bg_xy, zs]
            bg_xyz = bg_xyz + utils.z_zero_col(self._slc.centroids(cutoff_pcts))
            self._points_obb = bg_xyz

            # transform back
            bg_xyz = utils.transform_pts(
                bg_xyz,
                utils.inv_transform(self._slc.obb.transform),
            )
            self._points_ct = bg_xyz

        self._points = utils.transform_pts(self._points_ct, self._tfrm.matrix)
        return self._points

    def axis(self) -> np.ndarray:
        """calculate the axis that fits the bicipital groove and return the most extreme points"""
        if self._axis_ct is None:
            if self._points_ct is None:
                self.points()

            x, y, z = self._points_obb.T
            z_dist = np.max(z) - np.min(z)
            line_fit = skspatial.objects.Line.best_fit(self._points_obb)
            ends = np.array(
                [
                    line_fit.point + (line_fit.direction * (z_dist / 2)),
                    line_fit.point - (line_fit.direction * (z_dist / 2)),
                ]
            )
            ends = utils.transform_pts(
                ends, utils.inv_transform(self._slc.obb.transform)
            )
            self._axis_ct = ends

        self._axis = utils.transform_pts(self._axis_ct, self._tfrm.matrix)
        return self._axis

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
                name="Bicipital Groove",
            )
            return plot


def _find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")

    if idx == len(array):
        return idx - 1
    else:
        return idx


def _cart2pol(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """convert from cartesian coordinates to radial"""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    sorter = np.argsort(theta)
    r_arr = np.vstack((theta[sorter], r[sorter]))

    return r_arr


def _pol2cart(arr):
    r = arr[:, 1]
    theta = arr[:, 0]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.c_[x, y]
