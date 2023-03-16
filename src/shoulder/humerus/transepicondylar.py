from shoulder import utils

import shapely
import trimesh
import numpy as np
import itertools
import skspatial.objects

"""appraoch is to find largest extents slicing down shaft, then find furthest points away radially from 
centroid within the slice. Then calculate the distance of all point pairs.

alternate approach would be to create a line between starting at the centroid that passes through the 
furthest points away and cut it with the outer shape, then rotate the line +-10 degrees until it is 
at it's longest
"""


def medial_lateral_dist_multislice(mesh, num_slice):
    """slices along distal humerus and computes the medial lateral distance with a rotated bounding box

    Args:
        mesh (trimesh.mesh): rotatec trimesh mesh object
        num_slice (int): number of slices to make between 10% and 0%

    """

    # get length of the tranformed bone
    total_length = np.sum(abs(mesh.bounds[:, -1]))  # entire length of bone
    pos_length = mesh.bounds[
        mesh.bounds[:, -1] >= 0, -1
    ]  # bone in positive space, bone before the centerline midpoint

    proximal_cutoff = -0.8 * total_length + pos_length
    distal_cutoff = -0.99 * total_length + pos_length

    # spacing of cuts
    cuts = np.linspace(proximal_cutoff, distal_cutoff, num=num_slice)

    dist = []
    for cut in cuts:
        try:
            path = mesh.section(plane_origin=[0, 0, cut], plane_normal=[0, 0, 1])
            slice, to_3d = path.to_planar()
        except:
            break

        # get shapely object from path
        polygon = slice.polygons_closed[0]

        # create rotated bounding box
        majr_dist = utils.major_axis_dist(
            polygon.minimum_rotated_rectangle
        )  # maximize this distance
        dist.append(majr_dist)

    idx_max_dist = dist.index(max(dist))
    max_dist_cut = cuts[idx_max_dist]

    return max_dist_cut


def axis(mesh, transform, num_slice):

    # copy mesh then make changes
    mesh_rot = mesh.copy()
    mesh_rot.apply_transform(transform)

    # find z distance where medial lateral distance is longest
    z_dist = medial_lateral_dist_multislice(mesh_rot, num_slice)

    # slice at location of max medial-lateral distance
    path = mesh_rot.section(plane_normal=[0, 0, 1], plane_origin=[0, 0, z_dist])
    slice, to_3d = path.to_planar()

    # get shapely object from path
    polygon = slice.polygons_closed[0]

    # create rotated bounding box
    bound = polygon.minimum_rotated_rectangle
    bound_angle = utils.azimuth(bound)

    # cut ends off at edge of bounding box that align with major axis
    bound_scale = shapely.affinity.rotate(bound, bound_angle)
    bound_scale = shapely.affinity.scale(bound_scale, xfact=1.5, yfact=0.999)
    bound_scale = shapely.affinity.rotate(bound_scale, -bound_angle)
    ends = polygon.difference(bound_scale)

    # now we have the most medial and lateral points
    # sometimes one of the end sections can be split in two leaving more than 2 total ends
    if len(list(ends.geoms)) > 2:
        ab_dists = []
        # iterate through all distance combos
        for a, b in itertools.combinations(list(ends.geoms), 2):
            ab_dists.append(
                [
                    a,
                    b,
                    utils._dist(
                        np.array(a.centroid.xy).flatten(),
                        np.array(b.centroid.xy).flatten(),
                    ),
                ]
            )  # [obj,obj,distance]
        end_geoms = list(
            np.array(ab_dists)[np.argmax(np.array(ab_dists)[:, 2]), :2]
        )  # find location of max distance return shapely objs
        end_pts = np.array(
            [end_geoms[0].centroid.xy, end_geoms[1].centroid.xy]
        ).reshape(2, 2)
    else:
        end_pts = np.array(
            [ends.geoms[0].centroid.xy, ends.geoms[1].centroid.xy]
        ).reshape(2, 2)

    # add z distance back ins
    end_pts = utils.z_zero_col(end_pts)

    # transform back
    end_pts = utils.transform_pts(end_pts, to_3d)
    end_pts_ct = utils.transform_pts(end_pts, utils.inv_transform(transform))

    # calculate transform so trans-e axis algins with an axis in new CSYS
    etran_line = skspatial.objects.Line.best_fit(end_pts)
    transform_etran = trimesh.geometry.align_vectors(
        np.array(etran_line.direction), np.array([1, 0, 0])
    )  # calculate rotation matrix so z+

    # for right shoulders aligning the vectors involves flipping the humeral head, undo this
    if transform_etran[0][0] and transform_etran[2][2] < 0:
        # print('flipped again')

        transform_etran = trimesh.geometry.align_vectors(
            np.array(etran_line.direction), np.array([-1, 0, 0])
        )  # calculate rotation matrix so z+

    return end_pts_ct, transform_etran
