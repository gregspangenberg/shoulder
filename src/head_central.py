import time
import utils
import trimesh
import matplotlib.pyplot as plt
import circle_fit
import pandas as pd
import shapely
import numpy as np

def decoratortimer(decimal):
    def decoratorfunction(f):
        def wrap(*args, **kwargs):
            time1 = time.monotonic()
            result = f(*args, **kwargs)
            time2 = time.monotonic()
            print('{:s} function took {:.{}f} ms'.format(f.__name__, ((time2-time1)*1000.0), decimal ))
            return result
        return wrap
    return decoratorfunction

# @decoratortimer(3)
def head_direction(polygon, hc_maj_axis_pts):
    """ finds which pt of the head central axis corresponds to the aritcular surface
    and returns the row the point is in the array
    """
    bound = polygon.minimum_rotated_rectangle
    mnr = utils.minor_axis(bound)
    mnr_line = shapely.geometry.LineString(mnr)

    #split it
    half_slices = shapely.ops.split(polygon,mnr_line)
    half_poly_residual = []
    for half_poly in half_slices.geoms:
        pts = np.asarray(half_poly.exterior.xy).T
        _,_,_, residual = circle_fit.least_squares_circle(pts)
        half_poly_residual.append(residual)
    articular_half_centroid = np.asarray(half_slices.geoms[np.argmin(half_poly_residual)].centroid)

    hc_axis_pt0_dist =np.abs(utils._dist(articular_half_centroid, hc_maj_axis_pts[0,:]))
    hc_axis_pt1_dist =np.abs(utils._dist(articular_half_centroid, hc_maj_axis_pts[1,:]))

    if  hc_axis_pt0_dist < hc_axis_pt1_dist:
        return 0
    else:
        return 1

# @decoratortimer(3)
def multislice(mesh, num_slice):
    # get length of the tranformed bone
    total_length = np.sum(abs(mesh.bounds[:,-1])) # entire length of bone
    neg_length = mesh.bounds[mesh.bounds[:,-1]<=0,-1] # bone in negative space, bone past the centerline midpoint
    
    distal_cutoff = 0.85*total_length + neg_length
    proximal_cutoff = 0.99*total_length + neg_length

    # spacing of cuts
    cuts = np.linspace(distal_cutoff, proximal_cutoff , num = num_slice)

    polygons, to_3ds = [], []
    for cut in cuts:
        try:
            path = mesh.section(plane_origin=[0,0,cut], plane_normal=[0,0,1])
            slice,to_3d = path.to_planar()
        except:
            break
        to_3ds.append(to_3d)

        # get shapely object from path
        polygons.append(slice.polygons_closed[0])


    return polygons, to_3ds

# @decoratortimer(3)
def axis(mesh, transform):
    
    # copy mesh then make changes    
    mesh_rot = mesh.copy()
    mesh_rot.apply_transform(transform)
    # print(transform)
    # mesh_rot.show()


    polygons,to_3ds = multislice(mesh_rot,50)
    
    angle = [utils.azimuth(p.minimum_rotated_rectangle) for p in polygons]
    length = [utils.major_axis_dist(p.minimum_rotated_rectangle) for p in polygons]

    df = pd.DataFrame({
        'poly':polygons,
        'to_3d':to_3ds,
        'angle':angle,
        'length': length
        })
    
    # pull out info from poly with the maximmum major axis 
    df_max =df.iloc[df['length'].idxmax()]
    max_poly = df_max.poly # find max length poly
    max_to_3d = df_max.to_3d

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

    # transform back 
    maj_axis_pts_ct = utils.transform_pts(maj_axis_pts, utils.inv_transform(transform))
    min_axis_pts_ct = utils.transform_pts(min_axis_pts, utils.inv_transform(transform))

    # pull out id'd articular pt
    articular_pt = maj_axis_pts_ct[dir,:].reshape(1,3)

    # version should be the smaller of the two anlges the line makes
    version = df_max.angle
    if version >90:
        version = 180 - version

    # version is being measure from y-axis, switch to x-axis
    version = 90-version



    return maj_axis_pts_ct, min_axis_pts_ct, version, articular_pt
