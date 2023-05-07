from shoulder.humerus import mesh
from shoulder.humerus import canal
from shoulder.base import Landmark
from shoulder import utils

import plotly.graph_objects as go
import circle_fit
import random
import numpy as np
import shapely.ops
import skspatial.objects
from scipy.spatial import KDTree
from itertools import islice, cycle
import sklearn.cluster
import trimesh


class AnatomicNeck(Landmark):
    def __init__(self, obb: mesh.Obb, canal: canal.Canal):
        self._mesh_oriented = obb.mesh
        self._transform = obb.transform
        self._axis_ct = None
        self._axis = None
        self._points_ct = None
        self._points = None

    # needs implimentation
    def plane(self):
        pass

    def transform_landmark(self, transform) -> None:
        self._axis = utils.transform_pts(self._axis_ct, transform)
        self._points = utils.transform_pts(self._points_ct, transform)

    def _graph_obj(self):
        if self._axis is None:
            return None
        else:
            plot = go.Scatter3d(
                x=self._axis[:, 0],
                y=self._axis[:, 1],
                z=self._axis[:, 2],
                mode="markers",
                name="Anatomic Neck",
            )
            return plot


# supporting code


def rolling_cirle_fit(pts, seed_pt, threshold):
    # find which point is closest to seed point (articular_point)
    kdtree = KDTree(pts)
    d, i = kdtree.query(
        seed_pt
    )  # returns distance and loction in index of closest point

    r_pts = np.roll(
        pts, shift=-i, axis=0
    )  # shift (with wrap around) array until articular point is at beginning
    iters = cycle((iter(r_pts), reversed(r_pts)))
    ra_pts = np.vstack(
        [next(it) for it in islice(iters, len(r_pts))]
    )  # rolled and alterating both directions

    residuals = []
    dt_residuals = []
    skip_i = None
    fit_pts = []
    for i, pt in enumerate(ra_pts):
        if len(fit_pts) < 3:
            fit_pts.append(pt)
            continue
        else:
            if skip_i == None:
                fit_pts.append(pt)

            elif (skip_i % 2) == 0:  # if even is to be skipped
                if (i % 2) == 0:
                    continue
                else:
                    fit_pts.append(pt)

            else:  # if odd is to be skipped
                if (i % 2) == 0:
                    fit_pts.append(pt)
                else:
                    continue

            xc, yc, radius, residual = circle_fit.least_squares_circle(
                np.vstack(fit_pts)
            )
            residuals.append(residual)

            if len(fit_pts) <= 4:
                dt_residuals.append(0)
            else:
                dt_residuals.append(residuals[-1] - residuals[-2])

            if dt_residuals[-1] > threshold:
                if skip_i == None:
                    # print('\n1st THRESHOLD')
                    del fit_pts[-1]  # remove point that exceeded threshold
                    skip_i = i - 2
                else:
                    # print('\n 2nd THRESHOLD')
                    del fit_pts[-1]  # remove point that exceeded threshold
                    skip_i = [
                        skip_i,
                        len(fit_pts) - 1,
                    ]  # location in array of final stop points for each direction
                    break
    fit_pts = np.vstack(fit_pts)  # convert list of (1,3) to array of (n,3)

    return fit_pts[skip_i]


def midpoint_line(pt0, pt1):
    pt0 = pt0.flatten()
    pt1 = pt1.flatten()
    midpoint = np.mean(np.vstack([pt0, pt1]), axis=0)
    dir = skspatial.objects.Line.from_points(pt0, pt1).direction
    length = skspatial.objects.Point(pt0).distance_point(pt1)
    dir = dir / length  # create unit vector
    midpoint_line = skspatial.objects.Line(midpoint, dir)

    return midpoint_line, length


def multislice(mesh, cut_increments, normal):
    for cut in cut_increments:
        try:
            path = mesh.section(plane_origin=cut, plane_normal=normal)
            slice, to_3d = path.to_planar(normal=normal)

            if len(slice.polygons_closed) > 1:  # if more than 1 poly
                # bring each point back to 3d space
                centroids = [
                    utils.transform_pts(
                        utils.z_zero_col(np.array(p.centroid).reshape(1, -1)), to_3d
                    )
                    for p in slice.polygons_closed
                ]
                # extract just the z height
                z_centroids = [float(p[:, -1]) for p in centroids]
                # keep the largest z
                big_z_ind = z_centroids.index(max(z_centroids))

                polygon = slice.polygons_closed[big_z_ind]

            else:
                polygon = slice.polygons_closed[0]

        except:
            print("exception")
            # this will fail if at the slice location no polygon can be created
            continue

        yield [polygon, to_3d]


def rolling_circle_slices(mesh, seed_pt, locs, dir, thresh):
    end_pts = []
    for slice in multislice(mesh, locs, dir):
        polygon, to_3d = slice

        pts = np.asarray(polygon.exterior.xy).T  # extract points [nx3] matrix
        seed_pt_alt = utils.transform_pts(
            seed_pt, utils.inv_transform(to_3d)
        )  # project into plane space
        seed_pt_alt = seed_pt_alt[:, :-1]  # remove out of plane direction for now

        # find circular portion of trace with rolling least squares circle
        circle_end_pts = rolling_cirle_fit(pts, seed_pt_alt, thresh)
        circle_end_pts = utils.z_zero_col(circle_end_pts)

        circle_end_pts = utils.transform_pts(circle_end_pts, to_3d)
        end_pts.append(circle_end_pts)

    return np.vstack(end_pts)


def distal_proximal_zs_articular(end_pts):
    # the end points alternate back and forth so seperate them out
    _, labels, _ = sklearn.cluster.k_means(end_pts, 2)
    pts0 = end_pts[np.where(labels == 0)]
    pts1 = end_pts[np.where(labels == 1)]

    # filter out nonsense at weird z elevations, now that they have both been seperated
    pts0 = utils.z_score_filter(pts0, -1, 2)
    pts1 = utils.z_score_filter(pts1, -1, 2)

    filt_pts = np.vstack([pts0, pts1])

    pts0_mean = np.median(pts0[:, -1], axis=0)
    pts1_mean = np.median(pts1[:, -1], axis=0)

    # if the z values of even are higher
    if pts0_mean > pts1_mean:
        proximal_z = pts0_mean
        distal_z = pts1_mean
    else:
        proximal_z = pts1_mean
        distal_z = pts0_mean

    return filt_pts, distal_z, proximal_z


def inf_sup_articular(end_pts, minor_axis):
    # the end points alternate back and forth so seperate them out
    _, labels, _ = sklearn.cluster.k_means(end_pts, 2)
    pts0 = end_pts[np.where(labels == 0)]
    pts1 = end_pts[np.where(labels == 1)]

    # filter out nonsense at weird z elevations, now that they have both been seperated
    pts0 = utils.z_score_filter(pts0, -1, 2)
    pts1 = utils.z_score_filter(pts1, -1, 2)

    # rotate so y faces minor axis
    mnr_line = skspatial.objects.Line.best_fit(minor_axis)

    transform_mnr_y = trimesh.geometry.align_vectors(
        np.array(mnr_line.direction), np.array([-1, 0, 0])
    )
    if transform_mnr_y[0][0] and transform_mnr_y[2][2] < 0:
        transform_mnr_y = trimesh.geometry.align_vectors(
            np.array(mnr_line.direction), np.array([1, 0, 0])
        )
    pts0_mnr_y = utils.transform_pts(pts0, transform_mnr_y)
    pts1_mnr_y = utils.transform_pts(pts1, transform_mnr_y)

    # filter out any weird values in y
    pts0_mnr_y = utils.z_score_filter(pts0_mnr_y, 1, 2)
    pts1_mnr_y = utils.z_score_filter(pts1_mnr_y, 1, 2)

    # return back to og coordinate sytem
    pts0 = utils.transform_pts(pts0_mnr_y, utils.inv_transform(transform_mnr_y))
    pts1 = utils.transform_pts(pts1_mnr_y, utils.inv_transform(transform_mnr_y))

    pts0_mean = np.mean(pts0, axis=0).reshape(-1, 3)
    pts1_mean = np.mean(pts1, axis=0).reshape(-1, 3)

    # if the z values of even are higher
    if pts0_mean[:, -1] > pts1_mean[:, -1]:
        sup_pts = pts0
        inf_pts = pts1
    else:
        sup_pts = pts1
        inf_pts = pts0

    return inf_pts, sup_pts
