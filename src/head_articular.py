import utils
import matplotlib.pyplot as plt
import circle_fit
import pandas as pd
import numpy as np
import shapely.geometry
import shapely.ops
import skspatial.objects
from scipy.spatial import KDTree
from itertools import islice, cycle


def rolling_cirle_fit(pts, seed_pt, threshold=0.4):
    #find which point is closest to seed point (articular_point)
    kdtree = KDTree(pts)
    d, i = kdtree.query(seed_pt) # returns distance and loction in index of closest point

    r_pts = np.roll(pts, shift=-i, axis=0) # shift (with wrap around) array until articular point is at beginning
    iters = cycle((iter(r_pts), reversed(r_pts)))
    ra_pts = np.vstack([next(it) for it in islice(iters, len(r_pts))]) # rolled and alterating both directions
    
    residuals = []
    dt_residuals = []
    skip_i = None
    fit_pts = []
    for i,pt in enumerate(ra_pts):
        if len(fit_pts)<3:
            fit_pts.append(pt)
            continue
        else:
            if skip_i == None:
                fit_pts.append(pt)

            elif (skip_i%2) == 0: # if even is to be skipped
                if (i%2) == 0:
                    continue
                else:
                    fit_pts.append(pt)

            else: #if odd is to be skipped
                if (i%2) == 0:
                    fit_pts.append(pt)
                else:
                    continue

            xc,yc, radius, residual = circle_fit.least_squares_circle(np.vstack(fit_pts))
            residuals.append(residual)

            if len(fit_pts)<=4:
                dt_residuals.append(0)
            else:
                dt_residuals.append(residuals[-1]-residuals[-2])
            
            if dt_residuals[-1] > threshold:
                if skip_i == None:
                    print('\n1st THRESHOLD')
                    del fit_pts[-1] # remove point that exceeded threshold
                    skip_i = len(fit_pts)-1
                else:
                    print('\n 2ndTHRESHOLD')
                    del fit_pts[-1] # remove point that exceeded threshold
                    skip_i = [skip_i,len(fit_pts)] # location in array of final stop points for each direction
                    break
    fit_pts = np.vstack(fit_pts) # convert list of (1,3) to array of (n,3)
    return fit_pts[skip_i]


def multislice(mesh, cut_increments, normal):
    for cut in cut_increments:
        try:
            path = mesh.section(plane_origin=cut, plane_normal=normal)
            slice,to_3d = path.to_planar(normal=normal)
        except:
            break
        # get shapely object from path
        polygon = slice.polygons_closed[0]

        yield [polygon, to_3d]


def plane(mesh, transform, articular_pt, head_central_axis):
    # copy mesh then make changes    
    mesh_csys = mesh.copy()
    mesh_csys.apply_transform(transform)

    # get conditions for z direction slicing of the bone (z-dir)
    # get length of the tranformed bone
    total_length = np.sum(abs(mesh_csys.bounds[:,-1])) # entire length of bone
    neg_length = mesh_csys.bounds[mesh_csys.bounds[:,-1]<=0,-1] # bone in negative space, bone past the centerline midpoint
    # specify percentage of cutoffs
    distal_cutoff = 0.8*total_length + neg_length
    proximal_cutoff = 0.95*total_length + neg_length
    # spacing of cuts
    cuts = np.linspace(distal_cutoff, proximal_cutoff , num = 10)
    cuts_z = np.c_[np.zeros(len(cuts)), np.zeros(len(cuts)), cuts]
    normal = [0,0,1]

    # get conditions for the direction perpendicular to the central axis direction (normala90-dir)
    # get width of the transformed bone
    total_width = np.sum(abs(mesh_csys.bounds[:, 1])) # entire length of bone
    pos_width = mesh_csys.bounds[mesh_csys.bounds[:, 1]>=0, 1] # bone in positive space, bone before the centerline 
    neg_width = mesh_csys.bounds[mesh_csys.bounds[:, 1]<=0, 1] # bone in negative space, bone past the centerline 
    #specify percentage of cutoffs
    posterior_cutoff = 0.2*pos_width # yes posterior is in positive space for some reason
    anterior_cutoff = 0.2*neg_width
    # spacing of cuts
    cuts = np.linspace(posterior_cutoff, anterior_cutoff , num = 10)
    cuts_y = np.c_[np.zeros(len(cuts)), cuts, np.zeros(len(cuts))]
    # direction of section
    normala90 = skspatial.objects.Line.best_fit(utils.transform_pts(head_central_axis,transform)).direction
    rotate90_z = np.array([
        [0,1,0,0],
        [-1,0,0,0],
        [0,0,1,0],
        [0,0,0,1]]) # 90 rotation, could need to try a -90 rotation based on sample
    normala90 = utils.transform_pts(normala90.reshape(1,3), rotate90_z).reshape(3,)
    
    # iterate through views and slices
    fit_plane_pts = []
    for incr, dir in [[cuts_z, normal], [cuts_y,normala90]]:
        for slice in multislice(mesh_csys, incr, dir):
            polygon, to_3d = slice
            
            pts = np.asarray(polygon.exterior.xy).T # extract points [nx3] matrix
            # move seed point(articular_point) from CT csys to new csys
            hc_pt = utils.transform_pts(articular_pt, transform)

            # find seed point along plane perpendicular to head_central axis
            hc_pt_to2d = utils.transform_pts(hc_pt, utils.inv_transform(to_3d)) #project into plane space
            seed_pt = hc_pt_to2d[:,:-1] #remove out of plane direction for now

            # find circular portion of trace with rolling least squares circle
            circle_end_pts = rolling_cirle_fit(pts,seed_pt)
            circle_end_pts = utils.transform_pts(circle_end_pts, to_3d)
            fit_plane_pts.append(circle_end_pts)

    plane = skspatial.objects.Plane.best_fit(fit_plane_pts)

    return [plane.point, plane.normal]

    