""" angles

This module contains functions for the matrix math that determines neck shaft angle and retroversion.
"""

from shoulder import utils

import trimesh.geometry
import skspatial.objects
import numpy as np


def neck_shaft(anp_normal, _transform, _transform_arp):
    """measures angle in degrees between anatomic neck plane and canal axis

    Args:
        anp_normal (array): array of vector for anatomic neck plane normal
        _transform (array): transform to move to bone specific coordiante system
        _transform_arp (array): transform view to perpedicular to anp_normal and remove retroversion angle

    Returns:
        array: anatomic neck shaft angle
    """

    _transform_ns = np.matmul(_transform_arp, _transform)
    _anp_normal = utils.transform_pts(anp_normal, _transform_ns)
    _anp_normal = skspatial.objects.Line.from_points(
        _anp_normal[0, :].reshape(
            3,
        ),
        _anp_normal[1, :].reshape(
            3,
        ),
    ).direction
    # out of plane for neck shaft plane is now the y direction
    _anp_normal[1] = 0

    _, neck_shaft_angle = trimesh.geometry.align_vectors(
        _anp_normal, np.array([0, 0, 1]), return_angle=True
    )

    neck_shaft_angle = 180 - np.rad2deg(neck_shaft_angle)
    # print(neck_shaft_angle)
    return neck_shaft_angle


def retroversion(anp_normal, side, _transform):
    """measured between transepicondylar axis and the articular margin plane normal projected onto the axial(z) plane

    Args:
        trep_axis (array): transepiconylar axis
        anp_normal (array): articular margin plane normal vector

    Returns:
        _transform_arp (array): transfrom to rotate so med-lat axis is parallel to articular margin plane normal projected in axial
        rv_angle (float): angle in degrees of the anatomic retroversion between the transepicondylar axis and the anatomic neck plane
    """
    # project the anp_normal onto the z-plane
    _anp_normal = utils.transform_pts(anp_normal, _transform)

    # _anp_normal = skspatial.objects.Line(_anp_normal[0, :], _anp_normal[1, :]).direction
    _anp_normal = skspatial.objects.Line.from_points(
        _anp_normal[0, :].reshape(
            3,
        ),
        _anp_normal[1, :].reshape(
            3,
        ),
    ).direction

    # ignore z direction
    _anp_normal[-1] = 0

    if side == "left":
        _transform_arp, rv_angle = trimesh.geometry.align_vectors(
            _anp_normal, np.array([-1, 0, 0]), return_angle=True
        )
        # posterior should always be negative, currently is positive
        transform_lr = np.array(
            [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        _transform_arp = np.matmul(transform_lr, _transform_arp)
    else:
        _transform_arp, rv_angle = trimesh.geometry.align_vectors(
            _anp_normal, np.array([-1, 0, 0]), return_angle=True
        )

    # depending on clockwise or counterclockwise rotation correct it.
    rv_angle = np.rad2deg(rv_angle)
    if rv_angle > 90:
        rv_angle = 360 - rv_angle

    # transform_amptrep = utils.inv_transform(transform_amptrep)
    # print(rv_angle)
    return _transform_arp, rv_angle
