from shoulder import utils

import trimesh
import numpy as np
import pandas as pd
import plotly.express as px
import circle_fit
from skspatial.objects import Line, Points


"""this approach grabs the centroid from slices
"""


def orient_humerus(mesh):
    """rotates the humerus so the humeral head faces up (+ y-axis)

    Args:
        mesh (trimesh.mesh: mesh to rotate up

    Returns:
        mesh: rotated mesh
        flip_y_T: transform to flip axis if it was performed
        ct_T: transform back to CT space
    """
    # fit bounding box to mesh and orient view to along 'approximate' humeral canal axis
    ct_T = mesh.apply_obb()  # applies transform to mesh and returns matrix used

    """ The center of volume is now at (0,0,0) with the y axis of the CSYS being the long axis of the humerus.
    The z being left-right and the x being up-down when viewed along the y-axis (humeral-axis)
    Whether the humeral head lies in +y space or -y space is unkown. The approach to discover which end is which
    is to take a slice on each end and see which shape is more circular. The more circular end is obviously the
    humeral head.
    """
    # Get z bounds of box
    y_limits = (mesh.bounds[0][-1], mesh.bounds[1][-1])

    # look at slice shape on each end
    humeral_end = (
        0,
        np.inf,
    )  # (y_coordinate, residual_of_circle_fit), there is perhaps a better way of recording data
    for y_limit in y_limits:
        # make the slice
        y_slice = 0.95 * y_limit  # move 5% inwards on the half, so 2.5% of total length
        slice = mesh.section(plane_origin=[0, 0, y_slice], plane_normal=[0, 0, 1])
        (
            slice,
            to_3d,
        ) = (
            slice.to_planar()
        )  # returns the 2d view at plane and the transformation back to 3d space for each point

        # pull out the points along the shapes edge
        slice_pts = np.array(slice.vertices)
        xc, yc, r, residu = circle_fit.least_squares_circle(slice_pts)

        if (
            residu < humeral_end[1]
        ):  # 1st pass, less than inf record, 2nd pass if less than 1st
            humeral_end = (y_limit, residu)

    # if the y-coordinate of the humeral head is in negative space then
    # we are looking to see if a flip was performed and if it was needed
    if (
        humeral_end[0] < 0
    ):  # humeral_end is a set containing (y-coordinate, residual from circle fit)
        # print('flipped')
        # flip was reversed so update the ct_transform to refelct that
        flip_y_T = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        mesh.apply_transform(flip_y_T)
    else:
        # print('not flipped')
        flip_y_T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    return mesh, ct_T, flip_y_T


def centroid_multislice(mesh, cutoff_pcts, num_centroids):
    """Slices bone along long axis between the specified cutoff points and returns
    the centroids along the length

    Args:
        mesh (trimesh.mesh): trimesh mesh object to find centroids along length
        cutoff_pcts (list): list of two cutoff percentages i.e [0.2,0.8] would remove the upper 20% and lower 20%
        num_centroids (int): number of slices beween cutoff points to calculate centroids for

    Returns:
        centroids (np.array): array of xyz points for centroids along length
        cutoff_length (float): length between the cutoff percentages on the bone
    """

    # get length of the bone
    y_length = 2 * (
        abs(mesh.bounds[0][-1])
    )  # mesh centered at 0, multiply by 2 to get full length along humeral canal

    # find distance that the cutoff percentages are at
    cutoff_pcts.sort()  # ensure bottom slice pct is first
    distal_cutoff = cutoff_pcts[0] * y_length - (
        y_length / 2
    )  # pct of total y-length then subtract to return center to 0
    proximal_cutoff = cutoff_pcts[1] * y_length - (y_length / 2)
    # length between cutoff pts
    cutoff_length = abs(proximal_cutoff - distal_cutoff)

    # spacing of cuts
    cuts = np.linspace(distal_cutoff, proximal_cutoff, num=num_centroids)

    centroids = []  # record data
    for cut in cuts:
        slice = mesh.section(plane_origin=[0, 0, cut], plane_normal=[0, 0, 1])
        centroids.append(np.array(slice.centroid).reshape(1, 3))

    centroids = np.concatenate(centroids, axis=0)

    return centroids, cutoff_length


def centerline_plot(mesh, centerline, num_centroids):
    # sample mesh surface
    dots = trimesh.sample.sample_surface_even(mesh, 500)
    df_d = pd.DataFrame(dots[0])
    df_d = df_d.rename({0: "x", 1: "y", 2: "z"}, axis=1)

    # sample centerline
    df_c = pd.DataFrame(centerline)
    z_100 = np.linspace(
        df_c.iloc[0, :],
        df_c.iloc[
            1:,
        ],
        num=num_centroids,
    ).reshape(-1, 3)
    df_c = pd.DataFrame(z_100.round(3))  # remove uneeded precision
    df_c = df_c.rename({0: "x", 1: "y", 2: "z"}, axis=1)

    # plot
    join_dfs = {"bone": df_d, "axis": df_c}
    df = pd.concat([df.assign(identity=k) for k, df in join_dfs.items()])

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="identity")
    fig.update_layout(
        scene_aspectmode="data"
    )  # plotly defualts into focing 3d plots to be distorted into cubes, this prevents that

    return fig


def axis(mesh, cutoff_pcts, num_centroids):
    """calculates the centerline in region of humerus

    Args:
        mesh_file (str): path to mesh file
        cutoff_pcts (list): cutoff for where centerline is to be fit between i.e [0.2,0.8] -> middle 60% of the bone

    Returns:
        centerline: 2x3 matrix of xyz points at ends of centerline
        cenerline_dir: 1x3 matrix of xyz direction of lline normal
    """
    alt_mesh = mesh.copy()  # alt_mesh means altered mesh

    # rotate so humerus is up
    alt_mesh, to_ct_transform, flip_transform = orient_humerus(alt_mesh)

    # slice it !
    centroids, cutoff_length = centroid_multislice(alt_mesh, cutoff_pcts, num_centroids)

    # add in flip that perhaps occured
    to_ct_transform = np.matmul(flip_transform, to_ct_transform)

    # transform back
    centroids_ct = utils.transform_pts(centroids, flip_transform)
    centroids_ct = utils.transform_pts(centroids, utils.inv_transform(to_ct_transform))

    # calculate centerline
    points = Points(centroids_ct)
    centerline_fit = Line.best_fit(points)

    # repersent centerline as two points at the extents of the cutoff
    centerline1 = centerline_fit.point + (
        centerline_fit.direction * (cutoff_length / 2)
    )  # centerline_fit.point is in the middle
    centerline2 = centerline_fit.point - (
        centerline_fit.direction * (cutoff_length / 2)
    )

    # calculate the transform needed to go to new CSYS from CT CSYS
    # transform = utils.rot_matrix_3d(np.array(centerline_fit.direction), [0, 0, 1])
    transform = trimesh.geometry.align_vectors(np.array(centerline_fit.direction), [0, 0, 1])[:3,:3] #remove row and column
    # calculate rotation matrix so z+
    pt = mesh.centroid.reshape(3, 1)  # new CSYS has centroid at [0,0,0]
    transform = np.c_[
        transform, -1 * np.matmul(transform, pt)
    ]  # add in translation to centroid
    transform = np.r_[
        transform, np.array([[0, 0, 0, 1]])
    ]  # no scaling occurs so leave as default

    centerline_pts_ct = np.array([centerline1, centerline2])

    return centerline_pts_ct, transform
