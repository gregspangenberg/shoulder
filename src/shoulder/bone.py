from . import utils
from .humerus import mesh
from .humerus import canal
from .humerus import epicondyle
from .humerus import surgical_neck
from .humerus import anatomic_neck
from .humerus import bicipital_groove
from .humerus import bone_props
from .humerus import resection_plane
from .humerus import slice
from .base import Bone, Transform

from abc import ABC, abstractmethod
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

# ignore warnings
# This is bad practice but one of the libraries has an issue that generates a nonsense warning
import warnings

warnings.filterwarnings("ignore")


class ProximalHumerus(Bone):
    def __init__(self, stl_file):
        self._tfrm = Transform()
        self.transform = self._tfrm.matrix
        self._obb = mesh.ProxObb(stl_file)
        self.stl_file = self._obb.file
        self.mesh = self._obb.mesh_ct
        self._full_slices = slice.FullSlices(self._obb)

        # landmarks
        self.surgical_neck = surgical_neck.SurgicalNeck(
            self._full_slices, self._tfrm, only_proximal=True
        )
        self._proximal_slices = slice.ProximalSlices(self._obb, self.surgical_neck)
        self.canal = canal.Canal(self._full_slices, self._tfrm, proximal=True)
        self.bicipital_groove = bicipital_groove.DeepGroove(
            self._proximal_slices, self.canal, self._tfrm
        )
        self.anatomic_neck = anatomic_neck.AnatomicNeck(
            self._proximal_slices, self.bicipital_groove, self._tfrm
        )

        # resection
        self.resection = resection_plane.ResectionPlaneFactory(
            self.canal, self.anatomic_neck, self._tfrm
        )

        # metrics
        self.neckshaft = bone_props.NeckShaft(self.canal, self.anatomic_neck).calc
        self.radius_curvature = bone_props.RadiusCurvature(self.anatomic_neck).calc

    def apply_csys_canal_articular(self) -> np.ndarray:
        """applies a coordinate system constructed from the canal axis (+z) and the head central axis(+y) to previously calculated landmarks"""
        self.canal.axis()
        self.anatomic_neck.axis_central()
        self.anatomic_neck.axis_normal()
        self._tfrm.matrix = utils.construct_csys(
            self.canal._axis_ct, self.anatomic_neck._normal_axis_ct
        )
        self._update_landmark_data()
        self.mesh = self._obb.mesh_ct.copy().apply_transform(self._tfrm.matrix)
        self.transform = self._tfrm.matrix
        return self.transform

    def apply_csys_obb(self) -> np.ndarray:
        """applies a coordinate system constructed from an oriented bounding box to previously calculated landmarks"""

        self._tfrm.matrix = self._obb.transform
        self._update_landmark_data()
        self.mesh = self._obb.mesh.copy()
        self.transform = self._tfrm.matrix
        return self.transform

    def apply_csys_ct(self) -> np.ndarray:
        """applies a the native CT coordinate system to previously calculated landmarks"""

        self._tfrm.reset()
        self._update_landmark_data()
        self.mesh = self._obb.mesh_ct.copy()
        self.transform = self._tfrm.matrix
        return self.transform

    def apply_csys_custom(self, transform, from_ct=True) -> np.ndarray:
        """applies a user defined coordinate system defined as a transformation matrix between the CT coordinate system and the user defined coordiante system to previously calculated landmarks"""
        if from_ct:
            self._tfrm.matrix = transform
            self._update_landmark_data()
            self.mesh = self._obb.mesh_ct.copy().apply_transform(self._tfrm.matrix)
            self.transform = self._tfrm.matrix
        else:
            self._tfrm.matrix = np.dot(transform, self._tfrm.matrix)
            self._update_landmark_data()
            self.mesh = self.mesh.apply_transform(self._tfrm.matrix)
            self.transform = self._tfrm.matrix
        return self.transform

    def apply_translation(self, translation) -> np.ndarray:
        """applies a user defined translation in the CT coordinate system to previously calculated landmarks"""
        _transform = utils.translate_transform(translation)
        self._tfrm.matrix = np.dot(_transform, self._tfrm.matrix)
        self._update_landmark_data()
        self.mesh = self.mesh.apply_transform(self._tfrm.matrix)
        self.transform = self._tfrm.matrix
        return self.transform


# we are inheriting the functions but the init will be unique
class Humerus(ProximalHumerus):
    def __init__(self, stl_file):
        self._tfrm = Transform()
        self.transform = self._tfrm.matrix
        self._obb = mesh.FullObb(stl_file)
        self.stl_file = self._obb.file
        self.mesh = self._obb.mesh_ct
        self._full_slices = slice.FullSlices(self._obb)
        self._distal_slices = slice.DistalSlices(self._obb)

        # landmarks
        self.surgical_neck = surgical_neck.SurgicalNeck(self._full_slices, self._tfrm)
        self._proximal_slices = slice.ProximalSlices(self._obb, self.surgical_neck)
        self.canal = canal.Canal(self._full_slices, self._tfrm)
        self.bicipital_groove = bicipital_groove.DeepGroove(
            self._proximal_slices, self.canal, self._tfrm
        )
        self.anatomic_neck = anatomic_neck.AnatomicNeck(
            self._proximal_slices, self.bicipital_groove, self._tfrm
        )
        self.trans_epiconylar = epicondyle.TransEpicondylar(
            self._distal_slices, self.canal, self.anatomic_neck, self._tfrm
        )

        # resection
        self.resection = resection_plane.ResectionPlaneFactory(
            self.canal, self.anatomic_neck, self._tfrm
        )

        # metrics
        self.retroversion = bone_props.RetroVersion(
            self.canal, self.anatomic_neck, self.trans_epiconylar
        ).calc
        self.neckshaft = bone_props.NeckShaft(self.canal, self.anatomic_neck).calc
        self.radius_curvature = bone_props.RadiusCurvature(self.anatomic_neck).calc

    def apply_csys_canal_transepiconylar(self) -> np.ndarray:
        """applies a coordinate system constructed from the canal axis (z+) and transepicondylar axis (+y) to previously calculated landmarks"""
        self.canal.axis()
        self.trans_epiconylar.axis()
        self._tfrm.matrix = utils.construct_csys(
            self.canal._axis_ct, self.trans_epiconylar._axis_ct
        )

        self._update_landmark_data()
        self.mesh = self._obb.mesh_ct.copy().apply_transform(self._tfrm.matrix)
        self.transform = self._tfrm.matrix
        return self.transform
