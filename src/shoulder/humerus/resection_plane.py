from typing import Any
from shoulder.base import Landmark, Transform
from shoulder import utils
from shoulder.humerus import anatomic_neck
from shoulder.humerus import canal

import plotly.graph_objects as go
import skspatial.objects
import numpy as np


class ResectionPlane:
    def __init__(
        self, cnl: canal.Canal, anp: anatomic_neck.AnatomicNeck, tfrm: Transform
    ) -> None:
        self._cnl = cnl
        self._anp = anp
        self._tfrm = tfrm
        self._anp_plane_ct = anp._plane_ct
        self.plane_ct = anp._plane_ct

    @property
    def plane(self) -> skspatial.objects.Plane:
        return utils.transform_plane(self.plane_ct, self._tfrm.matrix)

    def __call__(self) -> skspatial.objects.Plane:
        return self.plane_ct

    # def __repr__(self):
    #     return f"{self.__class__.__name__}(point={self.plane.point}, normal={self.plane.normal})"


class ResectionPlaneFactory:
    def __init__(
        self, cnl: canal.Canal, anp: anatomic_neck.AnatomicNeck, tfrm: Transform
    ) -> None:
        self._cnl = cnl
        self._anp = anp
        self._tfrm = tfrm

    def create(self) -> ResectionPlane:
        """creates a resection plane at the antomic neck plane which can be modified later"""
        # ensure that anatomic neck has been calculated
        self._anp.plane()
        # create resection plane
        return ResectionPlane(self._cnl, self._anp, self._tfrm)
