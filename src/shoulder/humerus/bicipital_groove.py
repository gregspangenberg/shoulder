from shoulder import utils
from shoulder.base import Landmark


import numpy as np
import scipy.signal
import skspatial.objects
import plotly.graph_objects as go


class DeepGroove(Landmark):
    def __init__(self, obb):
        self._mesh_oriented_uobb = obb.mesh
        self._transform_uobb = obb.transform
        self._points_ct = None
        self._points = None
        self._axis_ct = None
        self._axis = None

    def axis(self, slice_num=35, interp_num=250):
        if self._axis is None:
            # slice_num  must use odd soo add 1 if even
            if (slice_num % 2) == 0:
                slice_num += 1

            z_max = np.max(self._mesh_oriented_uobb.bounds[:, -1])
            zs = np.linspace(0.92 * z_max, 0.75 * z_max, num=slice_num)

            pts = np.zeros((interp_num, 2, slice_num))
            pts_r = np.zeros((interp_num, 2, slice_num))
            pts_r_sp = np.zeros((interp_num, 2, slice_num))
            weights = np.zeros((interp_num, 2, slice_num))
            to_3Ds = np.zeros((4, 4, slice_num))

            for i, z in enumerate(zs):
                origin = [0, 0, z]
                normal = [0, 0, 1]
                path = self._mesh_oriented_uobb.section(
                    plane_origin=origin, plane_normal=normal
                )

                slice, to_3D = path.to_planar(normal=normal)
                big_poly = slice.polygons_closed[
                    np.argmax([p.area for p in slice.polygons_closed])
                ]

                # resample cartesion coordinates to create evenly spaced points
                _pts = np.asarray(big_poly.exterior.xy).T
                _pts = _resample_polygon(_pts, interp_num)

                pol = _cart2pol(_pts)

                f = scipy.interpolate.interp1d(pol[:, 0], pol[:, 1])
                theta_spaced = np.linspace(
                    np.min(pol[:, 0]), np.max(pol[:, 0]), (interp_num)
                )
                r_spaced = f(theta_spaced)

                pol_spaced = np.c_[theta_spaced, r_spaced]

                xy = _pol2cart(pol_spaced)

                # if a cavity is present do not count that as a weight
                theta_diff = (
                    np.diff(pol[:, 0], prepend=-10) < 0
                )  # prepend -10 so first difference is positive
                cav = _true_propogate(theta_diff)  # all cavities are True
                cav = np.array(cav, dtype=np.int32)  # make all true 1
                cav = cav ^ (
                    cav & 1 == cav
                )  # flip all 0s to 1s, since we want to preserve everythin but cavities
                cav_weight = np.c_[cav, cav]

                # log data
                pts[:, :, i] = xy
                pts_r[:, :, i] = pol
                pts_r_sp[:, :, i] = pol_spaced
                weights[:, :, i] = cav_weight
                to_3Ds[:, :, i] = to_3D

            # make each radial slice stationary
            pts_r_0 = pts_r.copy()
            pts_r_0[:, 1, :] = np.apply_along_axis(
                lambda x: x - np.mean(x), axis=0, arr=pts_r[:, 1, :]
            )

            # calulcate mean across each slice
            mean_pts_r_0 = np.mean(pts_r_0, axis=2)
            deg = np.rad2deg(mean_pts_r_0[:, 0])
            radius = mean_pts_r_0[:, 1]
            # calculate weighted mean that has cavity weight as 0
            w_mean_pts_r_0 = np.average(pts_r_0, axis=2, weights=weights)
            deg_w = np.rad2deg(w_mean_pts_r_0[:, 0])
            radius_w = w_mean_pts_r_0[:, 1]
            radius_w = scipy.signal.savgol_filter(
                radius_w, 10, 1
            )  # weights create jagged edges

            # calulate derivatives
            dd_radius = _derivative_smooth_ends(radius, 2, 10, 2)
            dd_radius_w = _derivative_smooth_ends(radius_w, 2, 10, 2)

            # find peaks in derivative, and keep 3 largest
            m_peaks, m_peaks_prop = scipy.signal.find_peaks(
                dd_radius, height=0, distance=interp_num / 360 * 25
            )
            m_peaks = m_peaks[np.argpartition(m_peaks_prop["peak_heights"], -3)[-3:]]
            wm_peaks, wm_peaks_prop = scipy.signal.find_peaks(
                dd_radius_w, height=0, distance=interp_num / 360 * 25
            )
            wm_peaks = wm_peaks[
                np.argpartition(wm_peaks_prop["peak_heights"], -3)[-3:]
            ]  # top 3 largest

            # find the peaks that are not near the furthest point
            # the furthest point is on the articular surface so any peaks neighbouring there
            # would not be the biciptal groove
            deg_far = deg_w[np.argmax(radius_w)]
            wm_peaks = sorted(wm_peaks)
            deg_w_peaks = deg_w[wm_peaks]
            filt_vals = np.r_[deg_w_peaks, deg_far]
            deg_w_shft = np.r_[
                deg_w[wm_peaks[0] :], deg_w[: wm_peaks[0]], deg_w[wm_peaks[0]]
            ]
            filt = [x for x in deg_w_shft if x in filt_vals]
            non_bg_peaks = (
                filt[filt.index(deg_far) - 1],
                filt[filt.index(deg_far) + 1],
            )
            bg_peak = list(set(deg_w_peaks) - set(non_bg_peaks))[0]

            # print(bg_peak)

            # get local minima by specifying serach window for
            # search up to 15 degrees away on each side
            deg_idx_var = int(round(360 / interp_num) * 15)
            bg_locals = np.zeros((1, 2, len(zs)))
            bg_xy = np.zeros((1, 2, len(zs)))
            bg_xyz = np.zeros((1, 3, len(zs)))
            for i, z in enumerate(zs):
                bg_idx_near = _find_nearest_idx(
                    pts_r_0[:, 0, i].flatten(), np.deg2rad(bg_peak)
                )
                bg_range = pts_r_0[
                    (bg_idx_near - deg_idx_var) : (bg_idx_near + deg_idx_var), :, i
                ]
                bg_local_i = np.argmin(bg_range[:, 1])
                bg_local = bg_range[bg_local_i, :]
                bg_locals[:, :, i] = bg_local
                # transform back to radial coordinates
                bg_i = bg_local_i + bg_idx_near - deg_idx_var  # put back in context
                _bg_xy = _pol2cart(pts_r[bg_i, :, i].reshape(1, 2))
                bg_xy[:, :, i] = _bg_xy

                # i think to_3D is perhaps fully broken, doesn't seem to work
                bg_xyz[:, :, i] = utils.transform_pts(np.c_[_bg_xy, 0], to_3Ds[:, :, i])

            bg_xyz = bg_xyz.transpose(2, 1, 0).reshape(-1, 3)
            print(bg_xyz.shape)
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
