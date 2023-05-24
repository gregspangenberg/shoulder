from shoulder import utils
from shoulder.base import Landmark

import ruptures
import numpy as np
import scipy.signal
import skspatial.objects
import plotly.graph_objects as go
from functools import cached_property
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

    def axis(self, cutoff_pcts=[0.35, 0.85], slice_num=35, interp_num=250):
        def _multislice(mesh, zs, interp_num, slice_num):
            # preallocate variables
            xy = np.zeros((interp_num, 2, slice_num))
            polar = np.zeros((interp_num, 2, slice_num))
            weights = np.zeros((interp_num, 2, slice_num))
            to_3Ds = np.zeros((4, 4, slice_num))

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
                # prepend -10 so first difference is positive
                theta_diff = np.diff(_pol[:, 0], prepend=-10) < 0
                cav_weight = _remove_cavitites(theta_diff)

                # log data
                xy[:, :, i] = _pts
                polar[:, :, i] = _pol
                weights[:, :, i] = cav_weight
                to_3Ds[:, :, i] = to_3D

            return xy, polar, weights, to_3Ds

        if self._axis is None:
            proximal_cutoff, distal_cutoff = self._surgical_neck_cutoff_zs(*cutoff_pcts)
            # slice_num  must use odd soo add 1 if even
            if (slice_num % 2) == 0:
                slice_num += 1

            zs = np.linspace(distal_cutoff, proximal_cutoff, num=slice_num).flatten()

            xy, polar, weights, to_3Ds = _multislice(
                self._mesh_oriented_uobb, zs, interp_num, slice_num
            )
            # make each radial slice stationary
            polar_0 = polar.copy()
            polar_0[:, 1, :] = np.apply_along_axis(
                lambda x: x - np.mean(x), axis=0, arr=polar[:, 1, :]
            )

            # calculate weighted mean across slices where cavities have a weight of 0
            polar_avg_0 = np.average(polar_0, axis=2, weights=weights)
            deg = np.rad2deg(polar_avg_0[:, 0])
            radius = polar_avg_0[:, 1]
            # weights create jagged edges
            radius = scipy.signal.savgol_filter(radius, 10, 1)

            # calulate derivatives
            dd_radius = _derivative_smooth_ends(radius, 2, 10, 2)

            peaks, _prop = scipy.signal.find_peaks(
                dd_radius, height=0, distance=interp_num / 360 * 25
            )
            peaks = peaks[
                np.argpartition(_prop["peak_heights"], -3)[-3:]
            ]  # top 3 largest

            # find the peaks that are not near the furthest point
            # the furthest point is on the articular surface so any peaks neighbouring there
            # would not be the biciptal groove
            peaks.sort()
            deg_rmax = deg[np.argmax(radius)]
            deg_peaks = deg[peaks]
            filt_vals = np.r_[deg_peaks, deg_rmax]
            deg_shft = np.r_[deg[peaks[0] :], deg[: peaks[0]], deg[peaks[0]]]
            filt = [x for x in deg_shft if x in filt_vals]
            non_bg_peaks = (
                filt[filt.index(deg_rmax) - 1],
                filt[filt.index(deg_rmax) + 1],
            )
            bg_peak = list(set(deg_peaks) - set(non_bg_peaks))[0]

            # get local minima by specifying serach window for
            # search up to 15 degrees away on each side
            deg_variance = int(round(360 / interp_num) * 15)
            bg_xyz = np.zeros((1, 3, len(zs)))
            for i, z in enumerate(zs):
                # print(polar_0)
                bg_idx_near = _find_nearest_idx(
                    polar_0[:, 0, i].flatten(), np.deg2rad(bg_peak)
                )
                # sometimes the degree variance will be higher than than the index bg is found at
                # when this occurs the indexing will start with a negative numebr causing it to fail
                # basically a wrap around problem

                if deg_variance > bg_idx_near:
                    bg_range = np.concatenate(
                        (
                            polar_0[(bg_idx_near - deg_variance) :, :, i],
                            polar_0[: (bg_idx_near + deg_variance), :, i],
                        ),
                        axis=0,
                    )
                else:
                    bg_range = polar_0[
                        (bg_idx_near - deg_variance) : (bg_idx_near + deg_variance),
                        :,
                        i,
                    ]
                bg_local_i = np.argmin(bg_range[:, 1])

                # transform back to radial coordinates
                bg_i = bg_local_i + (bg_idx_near - deg_variance)  # put back in context
                _bg_xy = _pol2cart(polar[bg_i, :, i].reshape(1, 2))

                # i think to_3D is perhaps fully broken, doesn't seem to work
                bg_xyz[:, :, i] = utils.transform_pts(np.c_[_bg_xy, 0], to_3Ds[:, :, i])

            bg_xyz = bg_xyz.transpose(2, 1, 0).reshape(-1, 3)

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


def _reorder_by_theta(arr):
    """reorder the array to start at the most negative theta.
    only works when it is an ordered array from a path object"""

    re_arr = np.r_[arr[np.argmin(arr[:, 0]) :], arr[: np.argmin(arr[:, 0])]]
    # there is an error that can occur when the most positive number which should be at
    # the end has a negative theta
    if re_arr[-1, 0] < 0:
        re_arr[-1, 0] = np.deg2rad(179.99)

    return re_arr


def _cart2pol(arr: np.ndarray) -> np.ndarray:
    """convert from cartesian coordinates to radial

    Args:
        arr (np.ndarray): cartesian coordiantes

    Returns:
        np.ndarray: radial coordinates
    """
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


def _derivative_smooth(arr, dd_order, window, smooth_order, _depth=0):
    """calculates nth order derivatives while smoothing in-between each step

    Args:
        arr (np.array): 1D array to calculate the derivative of
        dd_order (int): order of derivate to calculate
        window (int): interval upon which the smoothing occurs
        smooth_order (int): order of function which smoothing uses
        _depth (int, optional): . Defaults to 0.

    Returns:
        np.array: derivative of array of order n
    """
    if _depth == dd_order:
        return scipy.signal.savgol_filter(arr, window, smooth_order)
    arr = scipy.signal.savgol_filter(np.gradient(arr), window, smooth_order)
    return _derivative_smooth(arr, dd_order, window, smooth_order, _depth + 1)


def _derivative_smooth_ends(arr, dd_order, window, smooth_order, _depth=0):
    """calculates nth order derivatives while smoothing at the end

    Args:
        arr (np.array): 1D array to calculate the derivative of
        dd_order (int): order of derivate to calculate
        window (int): interval upon which the smoothing occurs
        smooth_order (int): order of function which smoothing uses
        _depth (int, optional): . Defaults to 0.

    Returns:
        np.array: derivative of array of order n
    """
    if _depth == dd_order:  # initial pass
        return scipy.signal.savgol_filter(arr, window, smooth_order)
    elif _depth == 0:  # final pass
        arr = scipy.signal.savgol_filter(arr, window, smooth_order)
    arr = np.gradient(arr)
    return _derivative_smooth(arr, dd_order, window, smooth_order, _depth + 1)


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
    arr = _true_propogate(arr)
    cav = np.array(arr, dtype=np.int32)  # make all true 1
    # flip all 0s to 1s, since we want to preserve everythin but cavities
    cav = cav ^ (cav & 1 == cav)
    # print(cav)
    # print(cav.shape)
    # weighting for each x,y point
    cav_weight = np.c_[cav, cav]
    # print(cav_weight)
    # print(cav_weight.shape)
    return cav_weight


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
