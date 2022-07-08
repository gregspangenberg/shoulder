import utils
import trimesh
import matplotlib.pyplot as plt
import circle_fit
import pandas as pd
import shapely
import numpy as np


def head_direction(polygon, hc_axis_pts):
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

    hc_axis_pt0_dist =np.abs(utils._dist(articular_half_centroid, hc_axis_pts[0,:]))
    hc_axis_pt1_dist =np.abs(utils._dist(articular_half_centroid, hc_axis_pts[1,:]))

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

def plane(mesh, transform):
    
    # copy mesh then make changes    
    mesh_rot = mesh.copy()
    mesh_rot.apply_transform(transform)
    # print(transform)
    # mesh_rot.show()


    polygons,zs = multislice(mesh_rot,100)
    
    angle = [utils.azimuth(p.minimum_rotated_rectangle) for p in polygons]
    length = [utils.major_axis_dist(p.minimum_rotated_rectangle) for p in polygons]

    df = pd.DataFrame({
        'poly':polygons,
        'z':zs,
        'angle':angle,
        'length': length
        })
    
    df_max =df.iloc[df['length'].idxmax()]
    max_poly = df_max.poly # find max length poly
    axis_pts = utils.major_axis(max_poly.minimum_rotated_rectangle)

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
