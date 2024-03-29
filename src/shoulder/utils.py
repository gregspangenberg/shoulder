import numpy as np
import scipy.stats
import scipy.spatial
import skspatial.objects


def write_iges_line(line, filepath):
    x, y, z = line[0]
    x1, y1, z1 = line[1]

    # i known  this string looks jank but it has to be this way for the whitespace to be printed correct
    s = """                                                                        S0000001
1H,,1H;,8Hpart.mco,10Hglobal.tmp,21HMedcad by Materialise,              G0000001
24HMedical CAD Modelling sw,32,38,8,308,16,8Hpart.med,1,2,2HMM,3,0.5,   G0000002
13H220408.155753,9.9999999E-09,1000,2HPN,14HMaterialise nv,1,0;         G0000003
     110       1       0       1       0       0       0       000000000D0000001
     110       0       0       1       0                    LINE       0D0000002\n"""
    s1 = f"110,{x},{y},{z},{x1},{y1},{z1};"
    s1 = s1.ljust(71) + "1P0000001\n"
    s2 = "S      1G      3D      2P      1                                        T0000001"
    iges = s + s1 + s2

    with open(filepath, "w") as f:
        f.write(iges)


def z_score_filter(arr, idx, threshold):
    i = arr - np.median(arr, axis=0)
    i = np.abs(scipy.stats.zscore(i)[:, idx]) < threshold

    return arr[i]


# BOUNDING BOX MATH #
#####################
def azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    bbox = np.asarray(mrr.exterior.xy).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])
    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])

    return az


def _azimuth(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""
    import numpy as np

    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180


def _dist(a, b):
    """distance between points"""
    import math

    return math.hypot(b[0] - a[0], b[1] - a[1])


def major_axis(mrr):
    bbox = np.array((mrr.exterior.xy)).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 > axis2:
        coords = np.stack(
            [
                np.mean(np.stack([bbox[0], bbox[1]], axis=0), axis=0),
                np.mean(np.stack([bbox[2], bbox[3]], axis=0), axis=0),
            ],
            axis=0,
        )
    else:
        coords = np.stack(
            [
                np.mean(np.stack([bbox[0], bbox[3]], axis=0), axis=0),
                np.mean(np.stack([bbox[1], bbox[2]], axis=0), axis=0),
            ],
            axis=0,
        )

    return coords


def major_axis_dist(mrr):
    bbox = np.array((mrr.exterior.xy)).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 > axis2:
        return axis1
    else:
        return axis2


def minor_axis(mrr):
    bbox = np.array((mrr.exterior.xy)).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 < axis2:
        coords = np.stack(
            [
                np.mean(np.stack([bbox[0], bbox[1]], axis=0), axis=0),
                np.mean(np.stack([bbox[2], bbox[3]], axis=0), axis=0),
            ],
            axis=0,
        )
    else:
        coords = np.stack(
            [
                np.mean(np.stack([bbox[0], bbox[3]], axis=0), axis=0),
                np.mean(np.stack([bbox[1], bbox[2]], axis=0), axis=0),
            ],
            axis=0,
        )

    return coords


def minor_axis_dist(mrr):
    bbox = np.array((mrr.exterior.xy)).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 > axis2:
        return axis2
    else:
        return axis1


def closest_pt(pt, pts, return_other_pts=False):
    """find closest point to the array of points"""
    kdtree = scipy.spatial.cKDTree(pts)
    d, i = kdtree.query(pt)  # returns distance and loction in index of closest point
    if return_other_pts:
        return [
            pts[i],
            np.delete(pts, i, axis=0),
        ]  # delete doesn't modify the orginal array
    else:
        return pts[i]


# MATRIX MATH #
###############
def rot_matrix_3d(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def z_zero_col(xy):
    return np.c_[xy, np.zeros(len(xy))]


def transform_pts(pts, transform):
    """Applies a transform to a set of xyz points

    Args:
        pts (np.array [nx3]): points to transform
        transform (np.array [4x4]): transformation matrix

    Returns:
        pts_transform(np.array [nx3]): transformed points
    """
    pts = np.c_[pts, np.ones(len(pts))].T  # add column of ones then transpose -> 4xn
    pts = np.matmul(transform, pts)
    pts = pts.T  # transpose back
    pts_transform = np.delete(
        pts, 3, axis=1
    )  # remove added ones now that transform is complete
    return pts_transform


def transform_plane(
    plane: skspatial.objects.Plane, transform: np.ndarray
) -> skspatial.objects.Plane:
    """transforms an skspatial plane with the provided transformation matrix"""
    point = plane.point.copy()
    normal = plane.normal.copy()
    point = np.array(point).reshape(1, 3)
    normal = np.array(normal).reshape(1, 3)

    # transform point
    point = transform_pts(point, transform)

    # transfrom normal
    normal = np.matmul(transform[:3, :3], normal.T).T

    return skspatial.objects.Plane(point.flatten(), normal.flatten())


def transform_line(
    line: skspatial.objects.Line, transform: np.ndarray
) -> skspatial.objects.Line:
    """transforms an skspatial plane with the provided transformation matrix"""
    point = line.point.copy()
    direction = line.direction.copy()
    point = np.array(point).reshape(1, 3)
    direction = np.array(direction).reshape(1, 3)

    # transform point
    point = transform_pts(point, transform)

    # transfrom normal
    normal = np.matmul(transform[:3, :3], direction.T).T

    return skspatial.objects.Line(point.flatten(), direction.flatten())


def inv_transform(transform):
    """inverses a transformation matrix

    Args:
        transform (np.array [4x4]): transformation matrix

    Returns:
        transform (np.array [4x4]): inverse transformation matrix
    """
    # make translation its own 4x4 matrix
    translate = transform[:3, -1]  # pull out 1x3 translation column matrix
    translate = np.c_[
        np.identity(3), translate
    ]  # add 3x3 identity matrix -> 3x4 matrix
    translate = np.r_[
        translate, np.array([[0, 0, 0, 1]])
    ]  # add row of 0 0 0 1 -> 4x4 matrix

    # make rotation its own 4x4 matrix
    rotate = transform[:, :-1]  # take everythin but last column -> 4x3 matrix
    rotate = np.c_[
        rotate, np.array([[0], [0], [0], [1]])
    ]  # replace with null translation -> 4x4 matrix

    # multiply the inverted matrices
    transform = np.matmul(
        np.linalg.inv(rotate), np.linalg.inv(translate)
    )  # R^-1 x T^-1

    return transform


def translate_transform(translate) -> np.ndarray:
    """creates a transform matrx of shape 4x4 from a translation"""
    translate = translate.reshape(1, 3)
    transform = np.identity(4)
    transform[:-1, -1] = translate
    return transform


def unit_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    vec = p1 - p2
    unit_vec = vec / np.linalg.norm(vec)

    return unit_vec


def angle_between(v1, v2):
    """calculates angle between 2 3d vectors"""
    # Calculate the dot product and the product of the magnitudes
    dot_product = np.dot(v1, v2)
    mag_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Calculate the cosine of the angle between the vectors
    cos_angle = dot_product / mag_product

    # Calculate the angle between the vectors in radians
    angle = np.arccos(cos_angle)

    return np.rad2deg(angle)


def construct_csys(vec_z, vec_y):
    # define center and two axes
    pos = np.average(vec_z, axis=0)
    pos = pos.flatten()
    z_hat = unit_vector(vec_z[0], vec_z[1])
    x_hat = unit_vector(vec_y[0], vec_y[1])

    # calculate remaing axis
    y_hat = np.cross(x_hat, z_hat)
    y_hat /= np.linalg.norm(y_hat)

    # transepicondylar axis is not quite perpendicular so do it again
    # this is but a temporary fix, maybe switchinf back the transepi to
    # being dependent on the canal would be wise
    x_hat = np.cross(y_hat, z_hat)
    x_hat /= np.linalg.norm(x_hat)

    # construct transform
    transform = np.c_[x_hat, y_hat, z_hat, pos]
    transform = np.r_[transform, np.array([0, 0, 0, 1]).reshape(1, 4)]

    # if the determinant is 0 then this is a reflection, to undo that the direciton of the
    # epicondylar axis should be switched

    if np.round(np.linalg.det(transform)) == -1:
        transform[:, 0] *= -1

    # return transform for CT csys -> canal-epi csys
    transform = inv_transform(transform)
    return transform


def unitxyz_to_spherical(xyz: np.ndarray) -> np.ndarray:
    """returns [r,theta,phi] i.e.[vector_length, retroversion, neckshaft] in units mm and degrees"""
    r = np.sqrt(np.sum(xyz**2))
    theta = np.arctan2(xyz[1], xyz[0])
    phi = np.arccos(xyz[2] / r)

    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)

    return np.array([r, theta, phi])


def spherical_to_unitxyz(sphr: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(sphr[1])
    phi = np.deg2rad(sphr[2])
    x = sphr[0] * np.sin(phi) * np.cos(theta)
    y = sphr[0] * np.sin(phi) * np.sin(theta)
    z = sphr[0] * np.cos(phi)
    return np.array([x, y, z])
