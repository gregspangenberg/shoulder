from . import bone
from . import utils

import skspatial.objects
import scipy.spatial
import numpy as np
import trimesh


from typing import Tuple


class HumeralHeadOsteotomy:
    """resescts the humeral head at the anp or offest from the anp"""

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
        # convert to 0 -> 2Pi from -Pi -> Pi
        ns = 180 - ns
        return ns

    @property
    def retroversion_rel(self):
        """retroversion angel of cut relative to native"""
        ret = utils.unitxyz_to_spherical(self._res_plane_csys_anp.normal)[1]
        # convert to 0 -> 2Pi from -Pi -> Pi
        ret = 180 - ret
        # measure from other direction if past 180
        if ret > 180:
            ret -= 360
        return ret

    def resect_mesh(self) -> Tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
        """resects the mesh in the current csys and returns a tuple of the head and resected humerus"""
        head = self._humerus.mesh.slice_plane(self.plane.point, self.plane.normal)
        resected_humerus = self._humerus.mesh.slice_plane(
            self.plane.point, -1 * self.plane.normal
        )

        return (head, resected_humerus)

    # modify the _plane
    def offset_retroversion(self, deg: float) -> None:
        sphr = utils.unitxyz_to_spherical(self._res_plane_csys_anp.normal)
        sphr[1] += -1 * deg  # increasing retroversion is negative

        new_normal = utils.spherical_to_unitxyz(sphr)
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=self._res_plane_csys_anp.point, normal=new_normal
        )

    def offest_neckshaft(self, deg: float) -> None:
        sphr = utils.unitxyz_to_spherical(self._res_plane_csys_anp.normal)
        sphr[2] += -1 * deg  # increasing neckshaft angle is negative

        new_normal = utils.spherical_to_unitxyz(sphr)
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=self._res_plane_csys_anp.point, normal=new_normal
        )

    def offset_depth_canal_axis(self, mm: float) -> None:
        """offset depth by moving along the canal axis"""
        # depth offset is controlled by changing the z height
        new_point = self._res_plane_csys_anp.point
        new_point[2] += mm
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=new_point, normal=self._res_plane_csys_anp.normal
        )

    def offset_depth_anp_normal(self, mm: float) -> None:
        """offset depth by moving along the anatomic neck plane normal axis"""
        new_point = self._res_plane_csys_anp.point
        new_point += mm * np.array(self._anp_plane_csys_anp)
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=new_point, normal=self._res_plane_csys_anp.normal
        )

    def offset_depth_res_normal(self, mm: float) -> None:
        """offset depth by moving along the current resction plane normal"""
        new_point = self._res_plane_csys_anp.point
        new_point += mm * np.array(self._res_plane_csys_anp)
        self._res_plane_csys_anp = skspatial.objects.Plane(
            point=new_point, normal=self._res_plane_csys_anp.normal
        )


class HumeralImplantation:
    """continues from the humeral head osteotomy and places the implant"""

    def __init__(self, osteotomy: HumeralHeadOsteotomy) -> None:
        self._osteotomy = osteotomy
