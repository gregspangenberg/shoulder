import matplotlib.pyplot as plt
import circle_fit
import pandas as pd
import numpy as np
import shapely.geometry
import shapely.ops
from scipy.spatial import KDTree
from itertools import islice, cycle


def rolling_cirle_fit(pts, seed_pt):
    #find which point is closest to seed point (articular_point)
    kdtree = KDTree(pts)
    d, i = kdtree.query(seed_pt) # returns distance and loction in index of closest point

    r_pts = np.roll(pts, shift=-i, axis=0) # shift (with wrap around) array until articular point is at beginning
    iters = cycle((iter(r_pts), reversed(r_pts)))
    ra_pts = np.vstack([next(it) for it in islice(iters, len(r_pts))]) # rolled and alterating both directions

    residuals = []
    dt_residuals = []
    skip_threshold = 0.3
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

            if len(fit_pts)==3:
                dt_residuals.append(0)
            else:
                dt_residuals.append(residual-residuals[-1])
            
            if dt_residuals[-1] > skip_threshold:
                if skip_i != None:
                    skip_i = i
                else:
                    skip_i = [skip_i,i] # location in array of final stop points for each direction
                    break


def plane(mesh, transform, articular_pt):

    # get length of the tranformed bone
    total_length = np.sum(abs(mesh.bounds[:,-1])) # entire length of bone
    neg_length = mesh.bounds[mesh.bounds[:,-1]<=0,-1] # bone in negative space, bone past the centerline midpoint

    distal_cutoff = 0.8*total_length + neg_length
    proximal_cutoff = 0.95*total_length + neg_length

    # spacing of cuts
    cuts = np.linspace(distal_cutoff, proximal_cutoff , num = 10)
    polygons, zs = [], []
    for cut in cuts:
        try:
            path = mesh.section(plane_origin=[0,0,cut], plane_normal=[0,0,1])
            slice,to_3d = path.to_planar()
        except:
            break

        # get shapely object from path
        polygon = slice.polygons_closed[0]
        # undo translation that is applied when moving from path3d to path2d
        translation = to_3d[:2,3:].T[0] # find x and y translation that occured
        polygon = shapely.affinity.translate(polygon, xoff=translation[0], yoff=translation[1])

        polygons.append(polygon)
        zs.append(cut)
    df = pd.DataFrame({
    'poly':polygons,
    'z':zs
    })


    # get width of the transformed bone
    total_width = np.sum(abs(mesh.bounds[:, 1])) # entire length of bone
    pos_width = mesh.bounds[mesh.bounds[:, 1]>=0, 1] # bone in positive space, bone before the centerline 
    neg_width = mesh.bounds[mesh.bounds[:, 1]<=0, 1] # bone in negative space, bone past the centerline 

    posterior_cutoff = 0.75*total_width + neg_width # yes posterior is in positive space for some reason
    anterior_cutoff = -0.75*total_width + pos_width
    
    # spacing of cuts
    cuts = np.linspace(posterior_cutoff, anterior_cutoff , num = 10)
    polygons, ys = [], []
    for cut in cuts:
        try:
            path = mesh.section(plane_origin=[0,cut,0], plane_normal=[0,1,0])
            slice,to_3d = path.to_planar()
        except:
            break

        # get shapely object from path
        polygon = slice.polygons_closed[0]
        # undo translation that is applied when moving from path3d to path2d
        translation = to_3d[:2,3:].T[0] # find x and y translation that occured
        polygon = shapely.affinity.translate(polygon, xoff=translation[0], yoff=translation[1])

        polygons.append(polygon)
        ys.append(cut)
    df = pd.DataFrame({
    'poly':polygons,
    'y':ys
    })