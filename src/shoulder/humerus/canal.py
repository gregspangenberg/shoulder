from shoulder import utils

import numpy as np
import circle_fit
from skspatial.objects import Line, Points
from functools import cached_property



class Canal:
    def __init__(self, mesh, cutoff_pcts, total_centroids):
        self.cutoff_pcts = cutoff_pcts
        self.total_centroids = total_centroids
        self.mesh = mesh
        self.mesh_oriented = self.mesh.copy()
        # orient the above with a bounding box
        self._transform_obb = self.mesh_oriented.apply_obb() 
        
    @cached_property
    def transform_orient(self):
        """rotates the humerus so the humeral head faces up (+ y-axis)

        Args:
            mesh (trimesh.mesh: mesh to rotate up

        Returns:
            mesh: rotated mesh
            flip_y_T: transform to flip axis if it was performed
            ct_T: transform back to CT space
        """

        """ The center of volume is now at (0,0,0) with the y axis of the CSYS being the long axis of the humerus.
        The z being left-right and the x being up-down when viewed along the y-axis (humeral-axis)
        Whether the humeral head lies in +y space or -y space is unkown. The approach to discover which end is which
        is to take a slice on each end and see which shape is more circular. The more circular end is obviously the
        humeral head.
        """
        # Get z bounds of box
        y_limits = (self.mesh_oriented.bounds[0][-1], self.mesh_oriented.bounds[1][-1])

        # look at slice shape on each end
        humeral_end = (
            0,
            np.inf,
        )  # (y_coordinate, residual_of_circle_fit), there is perhaps a better way of recording data
        for y_limit in y_limits:
            # make the slice
            y_slice = 0.95 * y_limit  # move 5% inwards on the half, so 2.5% of total length
            slice = self.mesh_oriented.section(plane_origin=[0, 0, y_slice], plane_normal=[0, 0, 1])
            # returns the 2d view at plane and the transformation back to 3d space for each point
            slice, to_3d =  slice.to_planar()  

            # pull out the points along the shapes edge
            slice_pts = np.array(slice.vertices)
            xc, yc, r, residu = circle_fit.least_squares_circle(slice_pts)

            # 1st pass, less than inf record, 2nd pass if less than 1st
            if residu < humeral_end[1]: 
                humeral_end = (y_limit, residu)

        # if the y-coordinate of the humeral head is in negative space then
        # we are looking to see if a flip was performed and if it was needed
        # humeral_end is a set containing (y-coordinate, residual from circle fit)
        if humeral_end[0] < 0:  
            print('flipped')
            # flip was reversed so update the ct_transform to refelct that
            transform_flip_y = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            self.mesh_oriented.apply_transform(transform_flip_y)
        else:
            print('not flipped')
            transform_flip_y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        # add in flip that perhaps occured
        transform = np.matmul(transform_flip_y, self._transform_obb)

        return transform


    @cached_property
    def axis(self):
        """calculates the centerline in region of humerus

        Args:
            mesh_file (str): path to mesh file
            cutoff_pcts (list): cutoff for where centerline is to be fit between i.e [0.2,0.8] -> middle 60% of the bone

        Returns:
            centerline: 2x3 matrix of xyz points at ends of centerline
            cenerline_dir: 1x3 matrix of xyz direction of lline normal
        """
        

        # slice it !
        centroids, cutoff_length = centroid_multislice(self.mesh_oriented, self.cutoff_pcts, self.total_centroids)

        # transform back
        centroids_ct = utils.transform_pts(centroids, utils.inv_transform(self.transform_orient))

        # calculate centerline
        canal_fit = Line.best_fit(Points(centroids_ct))

        # repersent centerline as two points at the extents of the cutoff
        canal_prox = canal_fit.point + (
            canal_fit.direction * (cutoff_length / 2)
        )  # canal_fit.point is in the middle
        canal_dstl = canal_fit.point - (
            canal_fit.direction * (cutoff_length / 2)
        )

        canal_pts_ct = np.array([canal_prox, canal_dstl])

        return canal_pts_ct
    

def centroid_multislice(mesh, cutoff_pcts, num_centroids):
    """Slices bone along long axis between the specified cutoff points and returns
    the centroids along the length

    Args:
        mesh (trimesh.mesh): trimesh mesh object to find centroids along length
        cutoff_pcts (list): list of two cutoff percentages i.e [0.2,0.8] would remove the upper 20% and lower 20%
        num_centroids (int): number of slices beween cutoff points to calculate centroids for

    Returns:
        centroids (np.array): array of xyz points for centroids along length
        cutoff_length (float): length between the cutoff percentages on the bone
    """

    # get length of the bone
    y_length = 2 * (
        abs(mesh.bounds[0][-1])
    )  # mesh centered at 0, multiply by 2 to get full length along humeral canal

    # find distance that the cutoff percentages are at
    cutoff_pcts.sort()  # ensure bottom slice pct is first
    distal_cutoff = cutoff_pcts[0] * y_length - (
        y_length / 2
    )  # pct of total y-length then subtract to return center to 0
    proximal_cutoff = cutoff_pcts[1] * y_length - (y_length / 2)
    # length between cutoff pts
    cutoff_length = abs(proximal_cutoff - distal_cutoff)

    # spacing of cuts
    cuts = np.linspace(distal_cutoff, proximal_cutoff, num=num_centroids)

    centroids = []  # record data
    for cut in cuts:
        slice = mesh.section(plane_origin=[0, 0, cut], plane_normal=[0, 0, 1])
        centroids.append(np.array(slice.centroid).reshape(1, 3))

    centroids = np.concatenate(centroids, axis=0)

    return centroids, cutoff_length