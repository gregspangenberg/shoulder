from shoulder.humerus import mesh
from shoulder.humerus import canal
from shoulder.humerus import epicondyle
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
    def __init__(self, obb: mesh.Obb, transepi: epicondyle.TransEpicondylar):
        self._mesh_oriented = obb.mesh
        self._transform = obb.transform
        self._transepi = transepi
        self._axis_ct = None
        self._axis = None
        self._points_ct = None
        self._points = None
        self._control_points_ct = None
        self._control_points = None

    @property
    def _transepi_axis(self):
        return self._transepi._axis_ct

    # needs implimentation
    def plane(self):
        if self._axis is None:
            circle_threshold = 0.3
            sup_inf_num = 16
            med_lat_num = 8

            articular_pt, hc_mnr_axis, medial_epicondyle_pt = self.__seed(
                self._mesh_oriented,
                50,
                utils.transform_pts(self._transepi_axis, self._transform),
            )
            # Slice along the head central minor axis
            # find which endpoint of hc_mnr_axis is closer to the medial_epicondyle_pt,
            # that side contins the greater tuberosity
            gt_side_pt, non_gt_side_pt = utils.closest_pt(
                medial_epicondyle_pt[:, :-1], hc_mnr_axis[:, :-1], return_other_pts=True
            )  # z removed from inputs
            gt_side_pt = np.c_[gt_side_pt, hc_mnr_axis[0, -1]]  # add z back in
            non_gt_side_pt = np.c_[non_gt_side_pt, hc_mnr_axis[0, -1]]

            hc_mnr_line, hc_mnr_length = self.__midpoint_line(
                gt_side_pt, non_gt_side_pt
            )
            # generate points along the middle 1/3 of the axis
            hc_mnr_axis_cut_locs = np.linspace(
                hc_mnr_line.to_point(-hc_mnr_length / 6),  # - away from GT
                hc_mnr_line.to_point(hc_mnr_length / 12),  # towards GT
                sup_inf_num,
            )  # loc of cuts, which way is positive and which is negative, i am unsure
            # find endpoints of where circle stops on each slice
            seed_pt = articular_pt.copy()
            seed_pt[:, -1] += (
                hc_mnr_length / 6
            )  # add extra z-height to offset the low z starting seed
            hc_mnr_end_pts = self.__rolling_circle_slices(
                self._mesh_oriented,
                seed_pt,
                hc_mnr_axis_cut_locs,
                hc_mnr_line.direction,
                circle_threshold,
            )
            # seperate into distal and proximal pts, and return filtered end points
            inf_pts, sup_pts = self.__inf_sup_articular(hc_mnr_end_pts, hc_mnr_axis)

            # slice along a line between the mean of inferior and superior endpoints
            inf_mean = np.mean(inf_pts, axis=0).reshape(-1, 3)
            sup_mean = np.mean(sup_pts, axis=0).reshape(-1, 3)
            inf_sup_line, inf_sup_len = self.__midpoint_line(inf_mean, sup_mean)
            inf_sup_cut_locs = np.linspace(
                inf_sup_line.to_point(-inf_sup_len / 6),
                inf_sup_line.to_point(inf_sup_len / 6),
                med_lat_num,
            )
            med_lat_pts = self.__rolling_circle_slices(
                self._mesh_oriented,
                seed_pt,
                inf_sup_cut_locs,
                inf_sup_line.direction,
                circle_threshold,
            )
            # med_pts, lat_pts = med_lat_articular(med_lat_pts, inf_pts, sup_pts)

            # reduce the importance of the superior points by removing 1/4 of all its points
            sup_pts = sup_pts[
                random.sample(range(len(sup_pts)), round(0.75 * len(sup_pts))), :
            ]
            # fit plane to fitted points
            fit_plane_pts = np.vstack([inf_pts, sup_pts, med_lat_pts])

            fit_plane_pts_ct = utils.transform_pts(
                fit_plane_pts, utils.inv_transform(self._transform)
            )  # revert back to CT space
            plane = skspatial.objects.Plane.best_fit(fit_plane_pts)  # fit plane

            # construct normal line
            amp_normal_line = skspatial.objects.Line(plane.point, plane.normal)
            amp_normal_p0 = amp_normal_line.to_point(t=0)
            amp_normal_p1 = amp_normal_line.to_point(t=25)
            if amp_normal_p1[-1] < amp_normal_p0[-1]:
                amp_normal_p1 = amp_normal_line.to_point(t=-25)
            amp_normal_axis = np.array([amp_normal_p0, amp_normal_p1])

            # get trace of plane intersecting bone
            plane_trace = np.array(
                self._mesh_oriented.section(
                    plane_origin=plane.point, plane_normal=plane.normal
                ).vertices
            )
            plane_pts = plane.to_points(
                lims_x=(-25, 25), lims_y=(-25, 25)
            )  # sets the spacing away from center point

            plane_pts_ct = utils.transform_pts(
                plane_pts, utils.inv_transform(self._transform)
            )

            self._control_points_ct = fit_plane_pts_ct
            self._control_points = fit_plane_pts_ct
            self._points_ct = plane_pts_ct
            self._points = plane_pts_ct
        return self._points

    def transform_landmark(self, transform) -> None:
        if self._points is not None:
            # self._axis = utils.transform_pts(self._axis_ct, transform)
            self._points = utils.transform_pts(self._points_ct, transform)
            self._control_points = utils.transform_pts(
                self._control_points_ct, transform
            )

    def _graph_obj(self):
        if self._points is None:
            return None
        else:
            plot = [
                go.Mesh3d(
                    x=self._points[:, 0],
                    y=self._points[:, 1],
                    z=self._points[:, 2],
                    opacity=0.8,
                    showlegend=True,
                    name="Anatomic Neck Plane",
                ),
                go.Scatter3d(
                    x=self._control_points[:, 0],
                    y=self._control_points[:, 1],
                    z=self._control_points[:, 2],
                    mode="markers",
                    name="Anatomic Neck Plane Control Points",
                ),
            ]

            return plot

    def __seed(self, mesh, slice_num, transepi):
        # transepi must be  in same csys as mesh

        def head_direction(polygon, hc_maj_axis_pts):
            """finds which pt of the head central axis corresponds to the aritcular surface
            and returns the row the point is in the array
            """
            bound = polygon.minimum_rotated_rectangle
            mnr = utils.minor_axis(bound)
            mnr_line = shapely.geometry.LineString(mnr)

            # split it
            half_slices = shapely.ops.split(polygon, mnr_line)
            half_poly_residual = []
            for half_poly in half_slices.geoms:
                pts = np.asarray(half_poly.exterior.xy).T
                _, _, _, residual = circle_fit.least_squares_circle(pts)
                half_poly_residual.append(residual)
            articular_half_centroid = np.asarray(
                half_slices.geoms[np.argmin(half_poly_residual)].centroid
            )

            hc_axis_pt0_dist = np.abs(
                utils._dist(articular_half_centroid, hc_maj_axis_pts[0, :])
            )
            hc_axis_pt1_dist = np.abs(
                utils._dist(articular_half_centroid, hc_maj_axis_pts[1, :])
            )

            if hc_axis_pt0_dist < hc_axis_pt1_dist:
                return 0
            else:
                return 1

        def seed_multislice(mesh, num_slice):
            # get length of the tranformed bone
            total_length = np.sum(abs(mesh.bounds[:, -1]))  # entire length of bone
            neg_length = mesh.bounds[
                mesh.bounds[:, -1] <= 0, -1
            ]  # bone in negative space, bone past the centerline midpoint

            distal_cutoff = 0.85 * total_length + neg_length
            proximal_cutoff = 0.99 * total_length + neg_length

            # spacing of cuts
            cuts = np.linspace(distal_cutoff, proximal_cutoff, num=num_slice).flatten()

            for cut in cuts:
                try:
                    path = mesh.section(
                        plane_origin=[0, 0, cut], plane_normal=[0, 0, 1]
                    )
                    slice, to_3d = path.to_planar()
                except:
                    break

                # get shapely object from path
                polygon = slice.polygons_closed[0]

                yield [polygon, to_3d]

        # find maximmum major axis
        max_length = None
        for slice in seed_multislice(mesh, slice_num):
            polygon, to_3d = slice
            length = utils.major_axis_dist(polygon.minimum_rotated_rectangle)
            if max_length is None or length > max_length:
                max_length = length
                max_poly = polygon
                max_to_3d = to_3d

        # find axes points
        maj_axis_pts = utils.major_axis(max_poly.minimum_rotated_rectangle)
        min_axis_pts = utils.minor_axis(max_poly.minimum_rotated_rectangle)

        # find location in array of the pt  that corrresponds to the articular portion
        dir = head_direction(max_poly, maj_axis_pts)
        # add in column of zeros
        maj_axis_pts = utils.z_zero_col(maj_axis_pts)
        min_axis_pts = utils.z_zero_col(min_axis_pts)
        # transform to 3d
        maj_axis_pts = utils.transform_pts(maj_axis_pts, max_to_3d)
        min_axis_pts = utils.transform_pts(min_axis_pts, max_to_3d)
        # grab pt
        articular_pt = maj_axis_pts[dir, :].reshape(1, 3)
        seed = articular_pt

        medial_epicondyle = utils.closest_pt(
            articular_pt[:, :-1], transepi[:, :-1]  # removed z from inputs
        )
        medial_epicondyle = np.c_[medial_epicondyle, transepi[0, -1]]  # add z back in

        return seed, min_axis_pts, medial_epicondyle

    def __rolling_circle_fit(self, pts, seed_pt, threshold):
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

    def __midpoint_line(self, pt0, pt1):
        pt0 = pt0.flatten()
        pt1 = pt1.flatten()
        midpoint = np.mean(np.vstack([pt0, pt1]), axis=0)
        dir = skspatial.objects.Line.from_points(pt0, pt1).direction
        length = skspatial.objects.Point(pt0).distance_point(pt1)
        dir = dir / length  # create unit vector
        midpoint_line = skspatial.objects.Line(midpoint, dir)

        return midpoint_line, length

    def __rolling_circle_slices(self, mesh, seed_pt, locs, dir, thresh):
        def multislice(mesh, cut_increments, normal):
            for cut in cut_increments:
                try:
                    path = mesh.section(plane_origin=cut, plane_normal=normal)
                    slice, to_3d = path.to_planar(normal=normal)

                    if len(slice.polygons_closed) > 1:  # if more than 1 poly
                        # bring each point back to 3d space
                        centroids = [
                            utils.transform_pts(
                                utils.z_zero_col(np.array(p.centroid).reshape(1, -1)),
                                to_3d,
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

        end_pts = []
        for slice in multislice(mesh, locs, dir):
            polygon, to_3d = slice

            pts = np.asarray(polygon.exterior.xy).T  # extract points [nx3] matrix
            seed_pt_alt = utils.transform_pts(
                seed_pt, utils.inv_transform(to_3d)
            )  # project into plane space
            seed_pt_alt = seed_pt_alt[:, :-1]  # remove out of plane direction for now

            # find circular portion of trace with rolling least squares circle
            circle_end_pts = self.__rolling_circle_fit(pts, seed_pt_alt, thresh)
            circle_end_pts = utils.z_zero_col(circle_end_pts)

            circle_end_pts = utils.transform_pts(circle_end_pts, to_3d)
            end_pts.append(circle_end_pts)

        return np.vstack(end_pts)

    def __distal_proximal_zs_articular(self, end_pts):
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

    def __inf_sup_articular(self, end_pts, minor_axis):
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
