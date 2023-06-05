from shoulder import utils
from shoulder.base import Landmark

import numpy as np
import math
import scipy.signal
import skspatial.objects
import plotly.graph_objects as go
import pandas as pd

import ruptures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.neighbors
import pickle

import matplotlib.pyplot as plt


class DeepGroove(Landmark):
    def __init__(self, obb):
        self._mesh_oriented_uobb = obb.mesh
        self._transform_uobb = obb.transform
        self._obb_cutoff_pcts = obb.cutoff_pcts
        self._points_ct = None
        self._points = None
        self._axis_ct = None
        self._axis = None
        self._data = None
        self._data_z = None
        self._pred = None

    def axis(
        self, cutoff_pcts=[0.35, 0.85], zslice_num=30, interp_num=1000, deg_window=6
    ):
        def _multislice(mesh, zs, interp_num, zslice_num):
            # preallocate variables
            polar = np.zeros(
                (
                    zslice_num,
                    2,
                    interp_num,
                )
            )
            weights = np.zeros((zslice_num, 2, interp_num))
            to_3Ds = np.zeros((zslice_num, 4, 4))

            for i, z in enumerate(zs):
                # grab the polygon of the slice
                origin = [0, 0, z]
                normal = [0, 0, 1]
                path = mesh.section(plane_origin=origin, plane_normal=normal)
                slice, to_3D = path.to_planar(normal=normal)
                # keep only largest polygon
                big_poly = slice.polygons_closed[
                    np.argmax([p.area for p in slice.polygons_closed])
                ]
                # resample cartesion coordinates to create evenly spaced points
                _pts = np.asarray(big_poly.exterior.xy).T
                _pts = _resample_polygon(_pts, interp_num)

                # convert to polar and ensure even degree spacing
                _pol = _cart2pol(_pts)

                # if a cavity is present do not count that as a weight
                # theta_diff = np.diff(_pol[:, 0], prepend=-10) < 0
                # print(_pol.shape)
                # _pol = _remove_cavitites(_pol)
                # if _pol.shape[0] != interp_num:
                #     print(z)
                #     print(_pol.shape)

                # _pts = _pol2cart(_pol)
                # _pts = _resample_polygon(_pts, interp_num)
                # _pol = _cart2pol(_pts)

                polar[i, :, :] = _pol.T
                # weights[i, :, :] = cav_weight.T
                to_3Ds[i, :, :] = to_3D

            return polar, to_3Ds

        def _X_process(polar_0, zs):
            def closest_angles(array, v):
                angs = []
                for a in array:
                    angs.append(math.atan2(math.sin(v - a), math.cos(v - a)))
                return np.abs(angs)

            def peak_nearest(all_peaks_theta):
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

            z_scale = MinMaxScaler().fit_transform(zs.reshape(-1, 1)).flatten()
            peak_zs = []
            peak_theta = []
            peak_radius = []
            peak_near = []
            peak_next_near = []
            peak_prom = []
            peak_width = []
            peak_widthheight = []
            for i, row in enumerate(polar_0):
                theta = row[0]
                radius = row[1]
                radius_og = radius.copy()
                radius = -1 * radius
                radius = scipy.signal.savgol_filter(radius, 10, 1)

                # sometimes the start or end contains a peak we need to shift
                rmin = -1 * np.argmin(radius)
                radius_roll = np.roll(radius, rmin)
                # now find peaks
                peaks, _prop = scipy.signal.find_peaks(
                    radius_roll,
                    height=0.1,
                    prominence=0.6,
                    width=0.1,
                )
                peaks = (peaks - rmin) % interp_num

                # if there are more than 3 peaks discard the lowest prominence one
                if len(peaks) > 3:
                    part = np.argpartition(_prop["prominences"], -3)[-3:]
                    peaks = peaks[part]  # top 3 largest

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

            X = pd.DataFrame(
                {
                    # "peak_theta": peak_theta,
                    "peak_radius": peak_radius,
                    "peak_near": peak_near,
                    "peak_next_near": peak_next_near,
                    "peak_z": peak_zs,
                    "peak_prom": peak_prom,
                    "peak_width": peak_width,
                    "peak_widthheight": peak_widthheight,
                }
            )
            scaler = StandardScaler()
            for col in [
                "peak_radius",
                "peak_near",
                "peak_next_near",
                "peak_prom",
                "peak_width",
                "peak_widthheight",
            ]:
                X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1)).flatten()

            return X, np.array(peak_theta), np.array(peak_zs)

        if self._axis is None:
            proximal_cutoff, distal_cutoff = self._surgical_neck_cutoff_zs(*cutoff_pcts)
            # slice_num  must use odd soo 1 if even
            if (zslice_num % 2) == 0:
                zslice_num += 1

            zs = np.linspace(distal_cutoff, proximal_cutoff, num=zslice_num).flatten()

            polar, to_3Ds = _multislice(
                self._mesh_oriented_uobb, zs, interp_num, zslice_num
            )
            # make each radial slice stationary by subtracting the mean
            polar_0 = polar.copy()
            polar_0[:, 1, :] = np.apply_along_axis(
                lambda x: x - np.mean(x), axis=1, arr=polar[:, 1, :]
            )

            # preprocess the data to get in the correct format
            X, peak_theta, peak_zs = _X_process(polar_0, zs)

            with open("src/shoulder/humerus/models/RFC_bg.pkl", "rb") as file:
                clf = pickle.load(file)

            kde = sklearn.neighbors.KernelDensity(kernel="linear")
            kde.fit(peak_theta[clf.predict(X).astype(bool)].reshape(-1, 1))
            tlin = np.linspace(-1 * np.pi, np.pi, 1000).reshape(-1, 1)
            bg_prob = np.exp(kde.score_samples(tlin))
            bg_theta = tlin[np.argmax(bg_prob)][0]

            # get local minima by specifying serach window for
            # search up to 15 degrees away on each side
            ivar = int(round(deg_window / (360 / interp_num)))
            if ivar < 1:
                ivar = 1
            bg_xyz = np.zeros((len(zs), 3))
            for i, z in enumerate(zs):
                bg_i_esti = _find_nearest_idx(polar_0[i, 0, :].flatten(), bg_theta)

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
                _bg_xy = _pol2cart(
                    polar[
                        i,
                        :,
                        bg_i_local,
                    ].reshape(1, 2)
                )
                bg_xyz[i, :] = utils.transform_pts(np.c_[_bg_xy, 0], to_3Ds[i, :, :])

            # transform back
            bg_xyz = utils.transform_pts(
                bg_xyz, utils.inv_transform(self._transform_uobb)
            )

            # construct an estimate of the bicipital groove axis from the bg_xyz pts
            line_ends = _fit_line(bg_xyz)

            self._axis_ct = line_ends
            self._axis = line_ends
            self._points_ct = bg_xyz
            self._points = bg_xyz

        return self._axis

    def transform_landmark(self, transform) -> None:
        if self._axis is not None:
            self._points = utils.transform_pts(self._points_ct, transform)
            self._axis = utils.transform_pts(self._axis_ct, transform)

    def _graph_obj(self):
        if self._points is None:
            return None

        else:
            plot = go.Scatter3d(
                x=self._points[:, 0],
                y=self._points[:, 1],
                z=self._points[:, 2],
                name="Bicipital Groove",
            )
            return plot

    def _surgical_neck_cutoff_zs(self, bottom_pct=0.35, top_pct=0.85):
        """given cutoff perccentages with 0 being the surgical neck and 1 being the
        top of the head return the z coordaintes
        """
        # this basically calcuates where the surgical neck is
        z_max = np.max(self._mesh_oriented_uobb.bounds[:, -1])
        z_min = np.min(self._mesh_oriented_uobb.bounds[:, -1])
        z_length = abs(z_max) + abs(z_min)

        z_low_pct = self._obb_cutoff_pcts[0]
        z_high_pct = self._obb_cutoff_pcts[1]
        distal_cutoff = z_low_pct * z_length + z_min
        proximal_cutoff = z_high_pct * z_length + z_min
        # print(distal_cutoff)

        z_intervals = np.linspace(distal_cutoff, 0.99 * z_max, 100)

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

        surgical_neck_z = z_intervals[bkp[0]]
        surgical_neck_top_head = z_max - surgical_neck_z
        bottom = surgical_neck_z + (surgical_neck_top_head * bottom_pct)
        top = surgical_neck_z + (surgical_neck_top_head * top_pct)

        # interval on which to calcaulte bicipital groove
        return [bottom, top]


def _find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def _cart2pol(arr: np.ndarray) -> np.ndarray:
    """convert from cartesian coordinates to radial

    Args:
        arr (np.ndarray): cartesian coordiantes

    Returns:
        np.ndarray: radial coordinates
    """

    def _reorder_by_theta(arr):
        """reorder the array to start at the most negative theta.
        only works when it is an ordered array from a path object"""

        re_arr = np.r_[arr[np.argmin(arr[:, 0]) :], arr[: np.argmin(arr[:, 0])]]
        # there is an error that can occur when the most positive number which should be at
        # the end has a negative theta
        if re_arr[-1, 0] < 0:
            re_arr[-1, 0] = np.deg2rad(179.99)

        return re_arr

    x = arr[:, 0]
    y = arr[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r_arr = np.c_[theta, r]
    # r_arr = r_arr[r_arr[:,1].argsort()] # sort by theta angle
    r_arr = _reorder_by_theta(r_arr)
    return r_arr


def _pol2cart(arr):
    r = arr[:, 1]
    theta = arr[:, 0]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.c_[x, y]


def _resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
    """interpolate between points in array to ensure even spacing between all points

    Args:
        xy (np.ndarray): array with columns for x and y coordinates arranged by order in
            which they occur while tracing along edge of polygon.
        n_points (int, optional): number of evenly spaced points to return. Defaults to 100.

    Returns:
        np.ndarray: evenly spaced points
    """
    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.linspace(0, d.max(), n_points)

    # interpolate x and y coordinates
    xy_interp = np.c_[
        np.interp(d_sampled, d, xy[:, 0]),
        np.interp(d_sampled, d, xy[:, 1]),
    ]

    return xy_interp


def _true_propogate(arr):
    """for each true interval double the size starting at same position"""

    def true_interval(x):
        """find interval where true bools  occur"""
        z = np.concatenate(([False], x, [False]))

        start = np.flatnonzero(~z[:-1] & z[1:])
        end = np.flatnonzero(z[:-1] & ~z[1:])

        return np.column_stack((start, end))

    b_i = true_interval(arr)

    a = arr.copy()

    for i in b_i:
        start = i[0]
        end = i[1]
        length = end - start
        # make sure that there is enoguh space
        if length > len(a[end : end + length]):
            continue
        else:
            a[end : end + length] = np.repeat(True, length)

    return a


def _remove_cavitites(arr):
    # prepend -10 so first difference is positive
    theta_diff = np.diff(arr[:, 0], prepend=-10) < 0
    theta_diff = _true_propogate(theta_diff)
    cav = np.array(theta_diff, dtype=np.int32)  # make all true 1
    # flip all 0s to 1s, since we want to preserve everythin but cavities
    cav = cav ^ (cav & 1 == cav)
    cav = cav.astype(bool)

    return arr[cav]


def _fit_line(bg_xyz):
    x, y, z = bg_xyz.T
    z_dist = np.max(z) - np.min(z)
    line_fit = skspatial.objects.Line.best_fit(bg_xyz)
    ends = np.array(
        [
            line_fit.point + (line_fit.direction * (z_dist / 2)),
            line_fit.point - (line_fit.direction * (z_dist / 2)),
        ]
    )

    return ends
