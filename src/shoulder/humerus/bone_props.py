from shoulder.humerus import slice
from shoulder import utils

from shoulder.humerus import bicipital_groove
from shoulder.humerus import anatomic_neck
from shoulder.humerus import epicondyle
from shoulder.humerus import canal

import numpy as np


class Side:
    def __init__(
        self,
        cn: canal.Canal,
        an: anatomic_neck.AnatomicNeck,
        bg: bicipital_groove.DeepGroove,
    ):
        self._cn = cn
        self._an = an
        self._bg = bg
        self._side = None

    def calc(self) -> str:
        """calculates whether the humerus is a left or right based
        on the location of bicipital groove with respect to the humeral head axis

        Returns:
            "left" or "right"
        """
        if self._side is None:
            # calc needed
            self._cn.axis()
            self._an.axis_central()
            self._bg.points()

            # construct csys
            transform = utils.construct_csys(
                self._cn._axis_ct, self._an._central_axis_ct
            )
            bg = utils.transform_pts(self._bg._points_ct, transform)
            bg = np.mean(bg, axis=0)

            if bg[1] <= 0:
                self._side = "left"
            else:
                self._side = "right"
        return self._side


class RetroVersion:
    def __init__(
        self,
        cn: canal.Canal,
        an: anatomic_neck.AnatomicNeck,
        te: epicondyle.TransEpicondylar,
        side,
    ):
        self._cn = cn
        self._an = an
        self._te = te
        self._side = side

    def calc(self) -> float:
        """calculates retroversion as the angle between the head central axis and transepicondylar axis"""

        self._cn.axis()
        self._te.axis()
        # construct csys
        transform = utils.construct_csys(self._cn._axis_ct, self._te._axis_ct)

        # Calculate anatomic neck axis
        self._an.axis_normal()
        an = self._an.axis_normal()
        an = utils.transform_pts(an, transform)
        an = utils.unit_vector(an[0], an[1])

        # find angle in xy plane measure from y axis
        an[0] = -1 * an[0]
        theta = utils.unitxyz_to_spherical(an)[1]

        if self._side() == "right":
            theta *= -1

        return theta


class NeckShaft:
    def __init__(self, cn: canal.Canal, an: anatomic_neck.AnatomicNeck) -> None:
        self._cn = cn
        self._an = an

    def calc(self) -> float:
        """calculates neck shaft angle as the angle between the canal axis and the anatomic neck axis"""

        # calc needed
        self._cn.axis()
        self._an.axis_normal()

        # construct csys
        transform = utils.construct_csys(self._cn._axis_ct, self._an._normal_axis_ct)

        # anatomic neck plane normal axis
        an = self._an._normal_axis_ct
        an = utils.transform_pts(an, transform)
        an = utils.unit_vector(an[0], an[1])

        # should return the obtuse angle
        ang = 180 - utils.unitxyz_to_spherical(an)[2]

        return ang


class RadiusCurvature:
    def __init__(self, an: anatomic_neck.AnatomicNeck) -> None:
        self._an = an

    def calc(self) -> float:
        """calculates the radius of curvature of the humeral head by fitting a sphere to the articular surface"""
        # calc needed
        if self._an._points_ct is None:
            self._an.points()
        radius, center = self._spherefit(self._an._points_all_articular_obb)
        return radius

    def _spherefit(self, pts: np.ndarray) -> tuple:
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
