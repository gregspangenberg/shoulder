from shoulder.humerus import slice
from shoulder import utils

from shoulder.humerus import anatomic_neck
from shoulder.humerus import epicondyle
from shoulder.humerus import canal

import numpy as np


class RetroVersion:
    def __init__(
        self,
        cn: canal.Canal,
        an: anatomic_neck.AnatomicNeck,
        te: epicondyle.TransEpicondylar,
    ):
        self._cn = cn
        self._an = an
        self._te = te

    def calc(self) -> float:
        """calculates retroversion as the angle between the head central axis and transepicondylar axis"""
        # central axis
        # calc  needed
        self._cn.axis()
        self._te.axis()
        # construct csys
        transform = utils.construct_csys(self._cn._axis_ct, self._te._axis_ct)

        self._an.axis_central()
        axc = self._an._central_axis_ct
        axc = utils.transform_pts(axc, transform)
        axc = utils.unit_vector(axc[0], axc[1])

        # should return the obtuse angle
        ang = (180 - utils.unitxyz_to_spherical(axc)[2]) % 360

        # currently no distinction is made between -1 dir and 1 dir transepi axis
        # need to impliment correction for this

        return ang


class NeckShaft:
    def __init__(self, cn: canal.Canal, an: anatomic_neck.AnatomicNeck) -> None:
        self._cn = cn
        self._an = an

    def calc(self) -> float:
        """calculates neck shaft angle as the angle between the canal axis and the anatomic neck axis"""

        # calc needed
        self._cn.axis()
        self._an.axis_central()

        # construct csys
        transform = utils.construct_csys(self._cn._axis_ct, self._an._central_axis_ct)

        # anatomic neck plane normal axis
        self._an.axis_normal()
        axan = self._an._normal_axis_ct
        axan = utils.transform_pts(axan, transform)
        axan = utils.unit_vector(axan[0], axan[1])

        # should return the obtuse angle
        ang = 180 - utils.unitxyz_to_spherical(axan)[1]

        return ang


class RadiusCurvature:
    def __init__(self, an: anatomic_neck.AnatomicNeck) -> None:
        self._an = an

    def calc(self):
        """calculates the radius of curvature of the humeral head by fitting a sphere to the articular surface"""
        # calc needed
        if self._an._points_ct is None:
            self._an.points()
        radius, center = self._spherefit(self._an._points_all_articular_obb)
        return radius

    def _spherefit(self, pts):
        #   Assemble the A matrix
        spX = pts[:, 0]
        spY = pts[:, 1]
        spZ = pts[:, 2]

        A = np.zeros((len(spX), 4))
        A[:, 0] = spX * 2
        A[:, 1] = spY * 2
        A[:, 2] = spZ * 2
        A[:, 3] = 1

        #   Assemble the f matrix
        f = np.zeros((len(spX), 1))
        f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
        C, residules, rank, singval = np.linalg.lstsq(A, f)

        #   solve for the radius
        t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
        radius = np.sqrt(t)[0]  # extract from array

        # extract center
        center = C[:-1].reshape(-1, 3)
        return radius, center
