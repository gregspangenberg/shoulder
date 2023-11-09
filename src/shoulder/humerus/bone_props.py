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

        # transepicondylar axis
        axte = self._te._axis_ct
        axte = utils.transform_pts(axte, transform)
        axte = utils.unit_vector(axte[0], axte[1])

        ang = utils.angle_between(axc, axte)

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

        # canal axis aligned with z but we want obtuse angle therefore negative z
        axcn = np.array([0, 0, -1])

        # anatomic neck plane normal axis
        self._an.axis_normal()
        axan = self._an._normal_axis_ct
        axan = utils.transform_pts(axan, transform)
        axan = utils.unit_vector(axan[0], axan[1])

        # should return the obtuse angle
        ang = utils.angle_between(axcn, axan)

        return ang
