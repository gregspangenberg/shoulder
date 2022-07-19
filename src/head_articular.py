import utils
import matplotlib.pyplot as plt
import circle_fit
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

def distal_proximal_zs_articular(end_pts):
    # the end points alternate back and forth so seperate them out
    end_pts_odd = end_pts[1::2]
    end_pts_even = end_pts[::2]

    even_z = np.mean(end_pts_even,axis=0)[-1]
    odd_z = np.mean(end_pts_odd,axis=0)[-1]

    # if the z values of even are higher return odd as distal, even as proximal
    if even_z > odd_z:
        proximal_z = even_z
        distal_z = odd_z
    else:
        distal_z = even_z
        proximal_z = odd_z

    return distal_z, proximal_z

def plane(mesh, transform, articular_pt, hc_mnr_axis):
    # transform into new csys   
    mesh_csys = mesh.copy()
    mesh_csys.apply_transform(transform)
    articular_pt = utils.transform_pts(articular_pt, transform)


    # Slice along the head central minor axis
    # generate line along head central minor axis
    hc_pt = np.mean(hc_mnr_axis, axis=0)
    hc_length = skspatial.objects.Point(hc_mnr_axis[0]).distance_point(hc_mnr_axis[1])
    hc_dir = skspatial.objects.Line.best_fit(hc_mnr_axis).direction # direction cuts are made
    hc_line = skspatial.objects.Line(points=hc_pt, direction=hc_dir)
    # generate points along the middle 1/3 of the axis
    hc_mnr_axis_cut_locs = np.linspace(hc_line.to_point(t=-hc_length/6), hc_line.to_point(t=hc_length/6), 10) #loc of cuts

    hc_mnr_end_pts = []
    for slice in multislice(mesh_csys, hc_mnr_axis_cut_locs, hc_dir):
        polygon, to_3d = slice
        
        pts = np.asarray(polygon.exterior.xy).T # extract points [nx3] matrix
        seed_pt = utils.transform_pts(articular_pt, utils.inv_transform(to_3d)) #project into plane space
        seed_pt = seed_pt[:,:-1] #remove out of plane direction for now
        
        # find circular portion of trace with rolling least squares circle
        circle_end_pts = rolling_cirle_fit(pts,seed_pt)
        circle_end_pts = utils.transform_pts(circle_end_pts, to_3d)
        hc_mnr_end_pts.append(circle_end_pts)


    # Slice along canal axis between disatl and proximal end points of the articular surface previously found
    # get the z interval of the distal and proximal articular surface normal to mnr axis
    z_distal, z_proximal = distal_proximal_zs_articular(hc_mnr_end_pts)
    z_axis_cut_locs = np.linspace(z_distal, z_proximal, 10)
    z_dir = [0,0,1]
    
    z_axis_end_pts = []
    for slice in multislice(mesh_csys, z_axis_cut_locs, z_dir):
        polygon, to_3d = slice
        
        pts = np.asarray(polygon.exterior.xy).T # extract points [nx3] matrix
        seed_pt = utils.transform_pts(articular_pt, utils.inv_transform(to_3d)) #project into plane space
        seed_pt = seed_pt[:,:-1] #remove out of plane direction for now
        
        # find circular portion of trace with rolling least squares circle
        circle_end_pts = rolling_cirle_fit(pts,seed_pt)
        circle_end_pts = utils.transform_pts(circle_end_pts, to_3d)
        z_axis_end_pts.append(circle_end_pts)
    

    # fit plane to fitted points
    fit_plane_pts = np.r_[hc_mnr_end_pts, z_axis_end_pts]
    fit_plane_pts = utils.transform_pts(fit_plane_pts, utils.inv_transform(transform)) # revert back to CT space
    plane = skspatial.objects.Plane.best_fit(fit_plane_pts)

    return [plane.point, plane.normal]

    