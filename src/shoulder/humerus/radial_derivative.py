import sklearn.linear_model
import math
import skspatial
import numpy as np
import scipy
from scipy.signal import savgol_filter

from shoulder import utils

np.set_printoptions(suppress=True)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return array[idx - 1]
    else:
        return array[idx]


def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def angle(a, b):
    return ((((a - b) % 360) + 540) % 360) - 180


def angle_in_range(alpha, lower, upper):
    return (alpha - lower) % 360 <= (upper - lower) % 360


def reorder_by_theta(arr):
    """reorder the array to start at the most negative theta.
    only works when it is an ordered array from a path object"""

    re_arr = np.r_[arr[np.argmin(arr[:, 0]) :], arr[: np.argmin(arr[:, 0])]]
    # there is an error that can occur when the most positive number which should be at
    # the end has a negative theta
    if re_arr[-1, 0] < 0:
        re_arr[-1, 0] = np.deg2rad(179.99)

    return re_arr


def cart2pol(arr: np.ndarray) -> np.ndarray:
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
    r_arr = reorder_by_theta(r_arr)
    return r_arr


def pol2cart(arr):
    r = arr[:, 1]
    theta = arr[:, 0]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.c_[x, y]


def pol2cart_1d(theta, r):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.c_[x, y]


def resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
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


def derivative_smooth(arr, dd_order, window, smooth_order, _depth=0):
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
        return savgol_filter(arr, window, smooth_order)
    arr = savgol_filter(np.gradient(arr), window, smooth_order)
    return derivative_smooth(arr, dd_order, window, smooth_order, _depth + 1)


def derivative_smooth_ends(arr, dd_order, window, smooth_order, _depth=0):
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
        return savgol_filter(arr, window, smooth_order)
    elif _depth == 0:  # final pass
        arr = savgol_filter(arr, window, smooth_order)
    arr = np.gradient(arr)
    return derivative_smooth(arr, dd_order, window, smooth_order, _depth + 1)


def true_propogate(arr):
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


def fit_line(bg_xyz, slice_num):

    x, y, z = bg_xyz.T  # unpack into vars
    A_xz = np.vstack((x, np.ones(len(x)))).T
    A_yz = np.vstack((y, np.ones(len(y)))).T

    # linear regression models
    x_reg = sklearn.linear_model.LinearRegression().fit(A_xz, z)
    y_reg = sklearn.linear_model.LinearRegression().fit(A_yz, z)

    m_xz, c_xz = x_reg.coef_[0], x_reg.intercept_
    m_yz, c_yz = y_reg.coef_[0], y_reg.intercept_

    z_p = np.linspace(np.min(z), np.max(z), slice_num)
    x_p = (z - c_xz) / m_xz
    y_p = (z - c_yz) / m_yz

    line = np.array([x_p, y_p, z_p]).T
    return line


def skspatial_fit_line(bg_xyz):
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


def radial(mesh, transform, slice_num, interp_num):

    mesh.apply_transform(transform)
    slice_num = 15  # must use odd
    interp_num = 250

    z_max = np.max(mesh.bounds[:, -1])
    zs = np.linspace(0.92 * z_max, 0.75 * z_max, num=slice_num)

    pts = np.zeros((interp_num, 2, slice_num))
    pts_r = np.zeros((interp_num, 2, slice_num))
    pts_r_sp = np.zeros((interp_num, 2, slice_num))
    weights = np.zeros((interp_num, 2, slice_num))
    to_3Ds = np.zeros((4, 4, slice_num))

    for i, z in enumerate(zs):
        origin = [0, 0, z]
        normal = [0, 0, 1]
        path = mesh.section(plane_origin=origin, plane_normal=normal)

        slice, to_3D = path.to_planar(normal=normal)
        big_poly = slice.polygons_closed[
            np.argmax([p.area for p in slice.polygons_closed])
        ]

        # resample cartesion coordinates to create evenly spaced points
        _pts = np.asarray(big_poly.exterior.xy).T
        _pts = resample_polygon(_pts, interp_num)

        pol = cart2pol(_pts)

        f = scipy.interpolate.interp1d(pol[:, 0], pol[:, 1])
        theta_spaced = np.linspace(np.min(pol[:, 0]), np.max(pol[:, 0]), (interp_num))
        r_spaced = f(theta_spaced)

        pol_spaced = np.c_[theta_spaced, r_spaced]

        xy = pol2cart(pol_spaced)

        # if a cavity is present do not count that as a weight
        theta_diff = (
            np.diff(pol[:, 0], prepend=-10) < 0
        )  # prepend -10 so first difference is positive
        cav = true_propogate(theta_diff)  # all cavities are True
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
    radius_w = savgol_filter(radius_w, 10, 1)  # weights create jagged edges

    # calulate derivatives
    dd_radius = derivative_smooth_ends(radius, 2, 10, 2)
    dd_radius_w = derivative_smooth_ends(radius_w, 2, 10, 2)

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
    deg_w_shft = np.r_[deg_w[wm_peaks[0] :], deg_w[: wm_peaks[0]], deg_w[wm_peaks[0]]]
    filt = [x for x in deg_w_shft if x in filt_vals]
    non_bg_peaks = (filt[filt.index(deg_far) - 1], filt[filt.index(deg_far) + 1])
    bg_peak = list(set(deg_w_peaks) - set(non_bg_peaks))[0]

    # print(bg_peak)

    # get local minima by specifying serach window for
    # search up to 15 degrees away on each side
    deg_idx_var = int(round(360 / interp_num) * 15)
    bg_locals = np.zeros((1, 2, len(zs)))
    bg_xy = np.zeros((1, 2, len(zs)))
    bg_xyz = np.zeros((1, 3, len(zs)))
    for i, z in enumerate(zs):
        bg_idx_near = find_nearest_idx(pts_r_0[:, 0, i].flatten(), np.deg2rad(bg_peak))
        bg_range = pts_r_0[
            (bg_idx_near - deg_idx_var) : (bg_idx_near + deg_idx_var), :, i
        ]
        bg_local_i = np.argmin(bg_range[:, 1])
        bg_local = bg_range[bg_local_i, :]
        bg_locals[:, :, i] = bg_local
        # transform back to radial coordinates
        bg_i = bg_local_i + bg_idx_near - deg_idx_var  # put back in context
        _bg_xy = pol2cart(pts_r[bg_i, :, i].reshape(1, 2))
        bg_xy[:, :, i] = _bg_xy

        # i think to_3D is perhaps fully broken, doesn't seem to work
        bg_xyz[:, :, i] = utils.transform_pts(np.c_[_bg_xy, 0], to_3Ds[:, :, i])

        # just adding back the z will work
        # bg_xyz[:, :, i] = np.c_[_bg_xy, zs[i]]

    # construct an estimate of the bicipital groove axis from the bg_xyz pts
    bg_xyz = bg_xyz.transpose(2, 1, 0).reshape(15, 3)
    # print(bg_xyz.round(2))

    # line = fit_line(bg_xyz,slice_num)
    # line_ends = np.array([line[0,:],line[-1,:]])

    line_ends = skspatial_fit_line(bg_xyz)
    # print(line_ends)
    # print(line_ends.shape,'\n')

    # transform back to CT coordinates
    # line_ends = utils.transform_pts(line_ends, utils.inv_transform(transform))
    # bg_xyz_ct = utils.transform_pts(bg_xyz, utils.inv_transform(transform))

    line_ends = line_ends
    bg_xyz_ct = bg_xyz

    print("\n")

    # print(line_ends)
    return (line_ends, bg_xyz_ct)
