from . import bone
from . import utils

import skspatial.objects
import scipy.spatial
import numpy as np
import trimesh


from typing import Tuple


class HumeralHeadOsteotomy:
    """resescts the humeral head at the anp or offest from the anp, will also transform humerus to ANP CSYS"""

    def __init__(self, humerus: bone.ProximalHumerus | bone.Humerus) -> None:
        self._humerus = humerus
        # record current csys
        self._tfrm_og = self._humerus._tfrm.matrix.copy()

        # in ANP csys, easier to do version and neckshaft calcs
        self._humerus.apply_csys_canal_articular()
        self._tfrm_anp = self._humerus._tfrm.matrix.copy()
        self._anp_plane_csys_anp = self._humerus.anatomic_neck.plane()
        self._res_plane_csys_anp = self._humerus.anatomic_neck.plane()

        # return to original csys
        # need to convert back to CT csys before moving to old csys as all
        # tfrm matrices are based upon the CT csys
        self._humerus.apply_csys_ct()
        self._humerus.apply_csys_custom(self._tfrm_og)

    @property
    def plane(self) -> skspatial.objects.Plane:
        """returns the resection plane in the current csys"""
        _plane = utils.transform_plane(
            self._res_plane_csys_anp, utils.inv_transform(self._tfrm_anp)
        )
        _plane = utils.transform_plane(_plane, self._humerus._tfrm.matrix)
        return _plane

    @property
    def neckshaft_rel(self):
        """neckshaft angle of cut relative to native"""
        ns = utils.unitxyz_to_spherical(self._res_plane_csys_anp.normal)[2]
        ns_og = utils.unitxyz_to_spherical(self._anp_plane_csys_anp.normal)[2]

        # convert to 0 -> 2Pi from -Pi -> Pi
        ns = 180 - ns
        ns_og = 180 - ns_og
        # make relative
        ns -= ns_og

        return ns

    @property
    def retroversion_rel(self):
        """retroversion angel of cut relative to native"""
        an = self._res_plane_csys_anp.normal
        an[0] = -1 * an[0]  # measure from -x so since retroversion is clockwise
        ret = utils.unitxyz_to_spherical(an)[1]

        if self._humerus.side() == "right":
            ret *= -1
        # measure is already relative so no need to subtract original retroversion

        return ret

    def points(self):
        """calculate the points of intersection of the resection plane and the mesh"""
        slice = self._humerus.mesh.section(self.plane.normal, self.plane.point)

        if len(slice.entities) > 1:
            # keep only largest polygon if more than 1
            pts = slice.discrete[np.argmax([p.area for p in slice.polygons_closed])]
        else:
            pts = slice.discrete[0]
        return pts

    def resect_mesh(self) -> Tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
        """resects the mesh in the current csys and returns a tuple of the head and resected humerus"""
        head = self._humerus.mesh.slice_plane(self.plane.point, self.plane.normal)
        resected_humerus = self._humerus.mesh.slice_plane(
            self.plane.point, -1 * self.plane.normal
        )

        return (head, resected_humerus)

    # modify the _plane
    def offset_retroversion(self, deg: float) -> None:
        """offset the retroversion by the given degrees

        Args:
            deg: the angle in degrees to offset the retroversion
        """
        sphr = utils.unitxyz_to_spherical(self._res_plane_csys_anp.normal)
        if self._humerus.side() == "left":
            sphr[1] += -1 * deg  # increasing retroversion is negative
        else:
            sphr[1] += deg  # for right negative is already applied
        new_normal = utils.spherical_to_unitxyz(sphr)
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=self._res_plane_csys_anp.point, normal=new_normal
        )

    def offest_neckshaft(self, deg: float) -> None:
        """offest the neckshaft angle by the given degrees

        Args:
            deg: the angle in degrees to offset the neckshaft angle
        """
        sphr = utils.unitxyz_to_spherical(self._res_plane_csys_anp.normal)
        sphr[2] += -1 * deg  # increasing neckshaft angle is negative

        new_normal = utils.spherical_to_unitxyz(sphr)
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=self._res_plane_csys_anp.point, normal=new_normal
        )

    def offset_depth(self, mm, direction="canal") -> None:
        """offset the resection depth in the specified direction

        Args:
            mm: the depth in mm to offset the resection plane
            direction: Options are "canal", "anp", or "resection".  Defaults to "canal".


        Raises:
            ValueError: if the direction is not one of the valid options
        """

        new_point = self._res_plane_csys_anp.point
        if direction == "canal":
            new_point[2] += mm
        elif direction == "anp":
            new_point += mm * np.array(self._anp_plane_csys_anp.normal)
        elif direction == "resection":
            new_point += mm * np.array(self._res_plane_csys_anp.normal)
        else:
            raise ValueError(
                "Invalid direction. Choose from: 'canal', 'anp', or 'resection'"
            )
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=new_point, normal=self._res_plane_csys_anp.normal
        )

    def offset_anterior_posterior(self, mm):
        """offset the resection plane in the anterior(+) posterior(-) direction

        Args:
            mm: Offset distance in milimeters. Anterior is positive, posterior is negative.
        """
        new_point = self._res_plane_csys_anp.point

        if self._humerus.side() == "left":
            new_point[0] -= mm
        else:
            new_point[0] += mm

        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=new_point, normal=self._res_plane_csys_anp.normal
        )

    def offset_medial_lateral(self, mm):
        """offset the resection plane in the medial(+) lateral(-) direction

        Args:
            mm: Offset distance in milimeters. Medial is positive, lateral is negative.
        """
        new_point = self._res_plane_csys_anp.point
        new_point[1] -= mm

        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=new_point, normal=self._res_plane_csys_anp.normal
        )


# class HumeralImplantation:
#     """continues from the humeral head osteotomy and places the implant"""

#     def __init__(self, osteotomy: HumeralHeadOsteotomy) -> None:
#         self._osteotomy = osteotomy
