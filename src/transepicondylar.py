import utils
import shapely
import trimesh
import numpy as np
import itertools
import skspatial.objects

"""appraoch is to find largest extents slicing down shaft, then find furthest points away radially from 
centroid within the slice. Then calculate the distance of all point pairs.

alternate approach would be to create a line between starting at the centroid that passes through the 
furthest points away and cut it with the outer shape, then rotate the line +-10 degrees until it is 
at it's longest
"""


def medial_lateral_dist_multislice(mesh, num_slice):
    """slices along distal humerus and computes the medial lateral distance with a rotated bounding box

    Args:
        mesh (trimesh.mesh): rotatec trimesh mesh object
        num_slice (int): number of slices to make between 10% and 0%

    """

    # get length of the tranformed bone
    total_length = np.sum(abs(mesh.bounds[:,-1])) # entire length of bone
    pos_length = mesh.bounds[mesh.bounds[:,-1]>=0,-1] # bone in positive space, bone before the centerline midpoint

    proximal_cutoff = -0.8*total_length + pos_length
    distal_cutoff = -0.99*total_length + pos_length

    # spacing of cuts
    cuts = np.linspace(proximal_cutoff, distal_cutoff , num = num_slice)

    dist = []
    for cut in cuts:
        try:
            path = mesh.section(plane_origin=[0,0,cut], plane_normal=[0,0,1])
            slice,to_3d = path.to_planar()
        except:
            break
        translation = to_3d[:2,3:].T[0]

        # get shapely object from path
        polygon = slice.polygons_closed[0]
        # undo translation that is applied when moving from path3d to path2d
        polygon = shapely.affinity.translate(polygon, xoff=translation[0], yoff=translation[1])

        # create rotated bounding box
        bound = polygon.minimum_rotated_rectangle
        majr_dist = utils.major_axis_dist(bound) # maximize this distance 
        dist.append(majr_dist)

    idx_max_dist = dist.index(max(dist))
    max_dist_cut = cuts[idx_max_dist]
    
    return max_dist_cut

# def medial_epicondyle(mesh_rot, transepi_axis):
#     # the medial epicondyle will always have less volume than the lateral epicondyle when taken as a
#     # percentage inwards from a bounding box

#     # cut in half and keep bottom
#     mesh_cut = trimesh.intersections.slice_mesh_plane(mesh_rot, plane_origin=[0,0,0], plane_normal=[0,0,-1]) 


def axis(mesh, transform, num_slice):

    # copy mesh then make changes    
    mesh_rot = mesh.copy()
    mesh_rot.apply_transform(transform)
    # print(transform)
    # mesh_rot.show()


    # find z distance where medial lateral distance is longest
    z_dist = medial_lateral_dist_multislice(mesh_rot, num_slice)

    # slice at location of max medial-lateral distance
    path = mesh_rot.section(plane_normal = [0,0,1],plane_origin=[0,0,z_dist])
    slice,to_3d = path.to_planar()
    translation = to_3d[:2,3:].T[0]

    # get shapely object from path
    polygon = slice.polygons_closed[0]
    # undo translation that is applied when moving from path3d to path2d
    polygon = shapely.affinity.translate(polygon, xoff=translation[0], yoff=translation[1])

    # create rotated bounding box
    bound = polygon.minimum_rotated_rectangle
    bound_angle = utils.azimuth(bound)

    # cut ends off at edge of bounding box that align with major axis
    bound_scale = shapely.affinity.rotate(bound, bound_angle)
    bound_scale = shapely.affinity.scale(bound_scale, xfact=1.5, yfact=0.999)
    bound_scale = shapely.affinity.rotate(bound_scale, -bound_angle)
    ends = polygon.difference(bound_scale)

    # now we have the most medial and lateral points
    # sometimes one of the end sections can be split in two leaving more than 2 total ends
    if len(list(ends.geoms)) > 2:
        ab_dists = []
        # iterate through all distance combos
        for a,b in itertools.combinations(list(ends.geoms), 2):
            ab_dists.append([a,b,utils._dist(np.array(a.centroid.xy).flatten(),np.array(b.centroid.xy).flatten())]) # [obj,obj,distance]
        end_geoms = list(np.array(ab_dists)[np.argmax(np.array(ab_dists)[:,2]),:2]) # find location of max distance return shapely objs
        end_pts = np.array([end_geoms[0].centroid.xy,end_geoms[1].centroid.xy]).reshape(2,2)
    else:
        end_pts = np.array([ends.geoms[0].centroid.xy,ends.geoms[1].centroid.xy]).reshape(2,2)
    
    # add z distance back ins
    end_pts = np.c_[end_pts,np.array([z_dist,z_dist])]

    #transform back
    end_pts_ct = np.c_[end_pts, np.ones(len(end_pts))].T
    end_pts_ct = np.matmul(np.linalg.inv(transform),end_pts_ct)
    end_pts_ct = end_pts_ct.T # transpose back
    end_pts_ct = np.delete(end_pts_ct,3,axis=1) # remove added ones now that transform is complete
    
    # calculate transform so trans-e axis algins with an axis in new CSYS
    etran_line = skspatial.objects.Line.best_fit(end_pts)
    transform_etran = trimesh.geometry.align_vectors(np.array(etran_line.direction), np.array([-1,0,0])) # calculate rotation matrix so z+
    
    # for right shoulders aligning the vectors involves flipping the humeral head, undo this
    if transform_etran[0][0] and transform_etran[2][2] < 0:
        # print('flipped again')
        
        transform_etran = trimesh.geometry.align_vectors(np.array(etran_line.direction), np.array([1,0,0])) # calculate rotation matrix so z+
       
        # flip_y = np.array([
        #     [-1,0,0,0],
        #     [0,1,0,0],
        #     [0,0,-1,0],
        #     [0,0,0,1]])
        # transform_etran = transform_etran*flip_y

    

    
    return end_pts_ct, transform_etran