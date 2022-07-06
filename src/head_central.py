import trimesh
import matplotlib.pyplot as plt
import circle_fit
import pandas as pd
import shapely
import numpy as np

def _azimuth(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""
    import numpy as np

    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

def _dist(a, b):
    """distance between points"""
    import math

    return math.hypot(b[0] - a[0], b[1] - a[1])

def major_axis_dist(mrr):
    bbox = np.asarray(mrr.exterior.xy).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        return axis2
    else:
        return axis1

def minor_axis(mrr):
    bbox = np.asarray((mrr.exterior.xy)).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 < axis2:
        coords = np.stack([
            np.mean(np.stack([bbox[0], bbox[1]], axis=0), axis=0), 
            np.mean(np.stack([bbox[2], bbox[3]], axis=0), axis=0)
        ], axis=0)
    else:
        coords = np.stack([
            np.mean(np.stack([bbox[0], bbox[3]], axis=0), axis=0), 
            np.mean(np.stack([bbox[1], bbox[2]], axis=0), axis=0)
        ], axis=0)

    return coords
    
def major_axis(mrr):
    bbox = np.asarray(mrr.exterior.xy).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 > axis2:
        coords = np.stack([
            np.mean(np.stack([bbox[0], bbox[1]], axis=0), axis=0), 
            np.mean(np.stack([bbox[2], bbox[3]], axis=0), axis=0)
        ], axis=0)
    else:
        coords = np.stack([
            np.mean(np.stack([bbox[0], bbox[3]], axis=0), axis=0), 
            np.mean(np.stack([bbox[1], bbox[2]], axis=0), axis=0)
        ], axis=0)

    return coords

def azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    bbox = np.asarray(mrr.exterior.xy).T
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])
    # print(axis1,axis2)
    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])

    return az

def head_direction(polygon, hc_axis_pts):
    """ finds which pt of the head central axis corresponds to the aritcular surface
    and returns the row the point is in the array
    """
    bound = polygon.minimum_rotated_rectangle
    mnr = minor_axis(bound)
    mnr_line = shapely.geometry.LineString(mnr)

    #split it
    half_slices = shapely.ops.split(polygon,mnr_line)
    half_poly_residual = []
    for half_poly in half_slices.geoms:
        pts = np.asarray(half_poly.exterior.xy).T
        _,_,_, residual = circle_fit.least_squares_circle(pts)
        half_poly_residual.append(residual)
    articular_half_centroid = np.asarray(half_slices.geoms[np.argmin(half_poly_residual)].centroid)

    hc_axis_pt0_dist =np.abs(_dist(articular_half_centroid, hc_axis_pts[0,:]))
    hc_axis_pt1_dist =np.abs(_dist(articular_half_centroid, hc_axis_pts[1,:]))

    if  hc_axis_pt0_dist < hc_axis_pt1_dist:
        return 0
    else:
        return 1


def multislice(mesh, num_slice):
    # get length of the tranformed bone
    total_length = np.sum(abs(mesh.bounds[:,-1])) # entire length of bone
    neg_length = mesh.bounds[mesh.bounds[:,-1]<=0,-1] # bone in negative space, bone past the centerline midpoint
    
    distal_cutoff = 0.85*total_length + neg_length
    proximal_cutoff = 0.99*total_length + neg_length

    # spacing of cuts
    cuts = np.linspace(distal_cutoff, proximal_cutoff , num = num_slice)

    polygons, zs = [], []
    for cut in cuts:
        try:
            path = mesh.section(plane_origin=[0,0,cut], plane_normal=[0,0,1])
            slice,to_3d = path.to_planar()
        except:
            break
        zs.append(cut)
        translation = to_3d[:2,3:].T[0]

        # get shapely object from path
        polygon = slice.polygons_closed[0]
        # undo translation that is applied when moving from path3d to path2d
        polygon = shapely.affinity.translate(polygon, xoff=translation[0], yoff=translation[1])
        polygons.append(polygon)

    return polygons, zs

def axis(mesh, transform):
    
    # copy mesh then make changes    
    mesh_rot = mesh.copy()
    mesh_rot.apply_transform(transform)
    # print(transform)
    # mesh_rot.show()


    polygons,zs = multislice(mesh_rot,100)
    
    angle = [azimuth(p.minimum_rotated_rectangle) for p in polygons]
    length = [major_axis_dist(p.minimum_rotated_rectangle) for p in polygons]

    df = pd.DataFrame({
        'poly':polygons,
        'z':zs,
        'angle':angle,
        'length': length
        })
    
    df_max =df.iloc[df['length'].idxmax()]
    max_poly = df_max.poly # find max length poly
    axis_pts = major_axis(max_poly.minimum_rotated_rectangle)

    # find location in array of the pt  that corrresponds to the articular portion
    dir = head_direction(max_poly,axis_pts)

    # add z distance back ins
    axis_pts = np.c_[axis_pts,np.array([df_max.z,df_max.z])]

    #transform back
    axis_pts_ct = np.c_[axis_pts, np.ones(len(axis_pts))].T
    axis_pts_ct = np.matmul(np.linalg.inv(transform),axis_pts_ct)
    axis_pts_ct = axis_pts_ct.T # transpose back
    axis_pts_ct = np.delete(axis_pts_ct,3,axis=1) # remove added ones now that transform is complete

    # pull out id'd articular pt
    articular_pt = axis_pts_ct[dir,:].reshape(1,3)

    # version should be the smaller of the two anlges the line makes
    version = df_max.angle
    if version >90:
        version = 180 - version

    # version is being measure from y-axis, switch to x-axis
    version = 90-version

    return axis_pts_ct, version, articular_pt
