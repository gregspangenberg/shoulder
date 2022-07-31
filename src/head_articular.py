import utils

import circle_fit
import scipy.stats
import numpy as np
import shapely.geometry
import shapely.ops
import skspatial.objects
from scipy.spatial import KDTree
from itertools import islice, cycle
import sklearn.cluster

 
def rolling_cirle_fit(pts, seed_pt, threshold=0.3):
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
                    # print('\n1st THRESHOLD')
                    del fit_pts[-1] # remove point that exceeded threshold
                    skip_i = i-2
                else:
                    # print('\n 2nd THRESHOLD')
                    del fit_pts[-1] # remove point that exceeded threshold
                    skip_i = [skip_i,len(fit_pts)-1] # location in array of final stop points for each direction
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
    
def rolling_circle_slices(mesh, seed_pt, locs, dir):
    end_pts = []
    for slice in multislice(mesh, locs, dir):
        polygon, to_3d = slice
        
        pts = np.asarray(polygon.exterior.xy).T # extract points [nx3] matrix
        seed_pt_alt = utils.transform_pts(seed_pt, utils.inv_transform(to_3d)) #project into plane space
        seed_pt_alt = seed_pt_alt[:,:-1] #remove out of plane direction for now
        
        # find circular portion of trace with rolling least squares circle
        circle_end_pts = rolling_cirle_fit(pts,seed_pt_alt)
        circle_end_pts = utils.z_zero_col(circle_end_pts)
        circle_end_pts = utils.transform_pts(circle_end_pts, to_3d)
        end_pts.append(circle_end_pts)
    
    return np.vstack(end_pts)

def distal_proximal_zs_articular(end_pts):
    # the end points alternate back and forth so seperate them out
    _,labels,_ = sklearn.cluster.k_means(end_pts,2)
    pts0 = end_pts[np.where(labels==0)]
    pts1 = end_pts[np.where(labels==1)]

    #filter out nonsense at weird z elevations, now that they have both been seperated
    pts0 = utils.z_score_filter(pts0,-1,2)
    pts1 = utils.z_score_filter(pts1,-1,2)
    filt_pts = np.vstack([pts0,pts1])

    pts0_med_z = np.median(pts0[:,-1],axis=0)
    pts1_med_z = np.median(pts1[:,-1],axis=0)

    # if the z values of even are higher 
    if pts0_med_z > pts1_med_z:
        proximal_z = pts0_med_z
        distal_z = pts1_med_z
    else:
        proximal_z = pts1_med_z
        distal_z = pts0_med_z


    return filt_pts, distal_z, proximal_z

def plane(mesh, transform, articular_pt, hc_mnr_axis, hc_mjr_axis, circle_threshold):
    # transform into new csys   
    mesh_csys = mesh.copy()
    mesh_csys.apply_transform(transform)
    articular_pt = utils.transform_pts(articular_pt, transform)
    hc_mnr_axis = utils.transform_pts(hc_mnr_axis, transform)
    
    # Slice along the head central minor axis
    hc_dir = skspatial.objects.Line.best_fit(hc_mnr_axis).direction # direction cuts are made
    # generate line along head central minor axis
    _hc_pt = np.mean(hc_mnr_axis, axis=0)
    _hc_mnr_length = skspatial.objects.Point(hc_mnr_axis[0]).distance_point(hc_mnr_axis[1])
    _hc_mnr_line = skspatial.objects.Line(point=_hc_pt, direction=hc_dir)
    # generate points along the middle 1/3 of the axis
    hc_mnr_axis_cut_locs = np.linspace(_hc_mnr_line.to_point(t=-_hc_mnr_length/6), _hc_mnr_line.to_point(t=_hc_mnr_length/6), 10) #loc of cuts
    # find endpoints of where circle stops on each slice
    hc_mnr_end_pts = rolling_circle_slices(mesh_csys, articular_pt, hc_mnr_axis_cut_locs, hc_dir)
    # seperate into distal and proximal pts, and return filtered end points
    hc_mnr_end_pts, _z_distal, _z_proximal = distal_proximal_zs_articular(hc_mnr_end_pts)
    
    # Slice along canal axis between disatl and proximal end points of the articular surface previously found

    z_axis_cut_locs = np.linspace(
        (_z_distal + 0.1*(_z_proximal-_z_distal)), 
        (_z_distal + 0.6*(_z_proximal-_z_distal)), 10)
    z_axis_cut_locs = np.c_[np.zeros((len(z_axis_cut_locs),2)),z_axis_cut_locs] # nx3
    z_dir = np.array([0,0,1])
    # find endpoints of where circle stops on each slice
    z_axis_end_pts = rolling_circle_slices(mesh_csys, articular_pt, z_axis_cut_locs, z_dir )

    # fit plane to fitted points
    fit_plane_pts = np.vstack([hc_mnr_end_pts, z_axis_end_pts])
    # fit_plane_pts = np.r_[hc_mnr_end_pts, z_axis_end_pts]
    fit_plane_pts = utils.transform_pts(fit_plane_pts, utils.inv_transform(transform)) # revert back to CT space
    plane = skspatial.objects.Plane.best_fit(fit_plane_pts) #fit plane

    # #project the major axes onto plane and slice along as well 
    # _hc_mjr_line = skspatial.objects.Line.best_fit(hc_mjr_axis)
    # _hc_mjr_length = skspatial.objects.Point(hc_mjr_axis[0]).distance_point(hc_mjr_axis[1])
    # _hc_mjr_dir_proj = plane.project_line(_hc_mjr_line).direction
    # hc_mjr_proj_line = skspatial.objects.Line(point=_hc_pt, direction=_hc_mjr_dir_proj)
    # hc_mjr_proj_cut_locs = np.linspace(hc_mjr_proj_line.to_point(t=-_hc_mjr_length/6), hc_mjr_proj_line.to_point(t=_hc_mjr_length/6), 10)
    # hc_mjr_proj_end_pts = rolling_circle_slices(mesh_csys, articular_pt, hc_mjr_proj_cut_locs, hc_mjr_proj_line.direction)
    
    # # refit the plane to the points
    # fit_plane_pts = np.r_[fit_plane_pts, hc_mjr_proj_end_pts]
    # fit_plane_pts = utils.transform_pts(fit_plane_pts, utils.inv_transform(transform)) # revert back to CT space
    # plane = skspatial.objects.Plane.best_fit(fit_plane_pts) #fit plane


    # get trace of plane intersecting bone
    plane_trace = np.array(mesh.section(plane_origin=plane.point, plane_normal=plane.normal).vertices)
    plane_pts = plane.to_points(lims_x=(-30,30), lims_y=(-30,30)) # sets the spacing away from center point
    """ Create function that calculates the anatomic neck shaft angle, from the normal
    """
    return plane_trace, fit_plane_pts, None

    