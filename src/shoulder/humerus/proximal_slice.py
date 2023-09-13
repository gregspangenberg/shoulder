import numpy as np
import trimesh

# cutoff_pcts=[0.35, 0.75]


class Slices:
    def __init__(self, mesh, zslice_num=300, interp_num=1000) -> None:
        zs = np.linspace(distal_cutoff, proximal_cutoff, num=zslice_num).flatten()
        self._multislice(mesh)

    def _cart2pol(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """convert from cartesian coordinates to radial"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        sorter = np.argsort(theta)
        r_arr = np.vstack((theta[sorter], r[sorter]))

        return r_arr

    def _resample_polygon(self, xy: np.ndarray, n_points: int = 100) -> np.ndarray:
        """interpolate between points in array to ensure even spacing between all points

        Args:
            xy (np.ndarray): array with columns for x and y coordinates arranged by order in
                which they occur while tracing along edge of polygon.
            n_points (int, optional): number of evenly spaced points to return. Defaults to 100.

        Returns:
            np.ndarray: evenly spaced points
        """
        # Cumulative Euclidean distance between successive polygon points.
        # This will be the "x" for interpolation
        d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=1) ** 2).sum(axis=0))])

        # get linearly spaced points along the cumulative Euclidean distance
        d_sampled = np.linspace(0, d.max(), n_points)

        # interpolate x and y coordinates
        xy_interp = np.vstack(
            (
                np.interp(d_sampled, d, xy[0, :]),
                np.interp(d_sampled, d, xy[1, :]),
            )
        )

        return xy_interp

    def _multislice(self, mesh, zs, interp_num, zslice_num):
        # preallocate variables
        polar = np.zeros(
            (
                zslice_num,
                2,
                interp_num,
            )
        )
        weights = np.zeros((zslice_num, 2, interp_num))
        to_3Ds = np.zeros((zslice_num, 4, 4))

        for i, z in enumerate(zs):
            # grab the polygon of the slice
            origin = [0, 0, z]
            normal = [0, 0, 1]
            path = mesh.section(plane_origin=origin, plane_normal=normal)
            slice, to_3D = path.to_planar(normal=normal)
            # keep only largest polygon
            big_poly = slice.polygons_closed[
                np.argmax([p.area for p in slice.polygons_closed])
            ]
            # resample cartesion coordinates to create evenly spaced points
            _pts = np.asarray(big_poly.exterior.xy)
            _pts = _resample_polygon(_pts, interp_num)

            # convert to polar and ensure even degree spacing
            _pol = self._cart2pol(_pts[0, :], _pts[1, :])

            # assign
            polar[i, :, :] = _pol
            to_3Ds[i, :, :] = to_3D

        return polar, to_3Ds
