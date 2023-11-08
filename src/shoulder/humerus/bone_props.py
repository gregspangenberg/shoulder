from shoulder.humerus import slice
from shoulder import utils

from shoulder.humerus import anatomic_neck
from shoulder.humerus import epicondyle
from shoulder.humerus import canal


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

    def calc(self):
        """calculates retroversion as the angle between the head central axis and transepicondylar axis"""
        # central axis
        transform = utils.construct_csys(self._cn.axis(), self._te.axis())

        self._an.axis_central()
        axc = self._an._central_axis_ct
        axc = utils.transform_pts(axc, transform)
        axc = utils.unit_vector(axc[0], axc[1])

        # transepicondylar axis
        self._te.axis()
        axte = self._te._axis_ct
        axte = utils.transform_pts(axte, transform)
        axte = utils.unit_vector(axte[0], axte[1])
        print(axte, axc)
        ang = utils.angle_between(axc, axte)

        # currently no distinction is made between -1 dir and 1 dir transepi axis
        # need to impliment correction for this

        return ang
