from . import utils
from .humerus import mesh
from .humerus import canal
from .humerus import epicondyle
from .humerus import surgical_neck
from .humerus import anatomic_neck
from .humerus import bicipital_groove
from .humerus import bone_props
from .humerus import slice
from .base import Bone

from abc import ABC, abstractmethod
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

# ignore warnings
# This is bad practice but one of the libraries has an issue that generates a nonsense warning
import warnings

warnings.filterwarnings("ignore")

# these classes have redundancies but sometimes nee to be treated very differently in subtle ways. Combine them in the future once package is in a more stable state


class Humerus(Bone):
    def __init__(self, stl_file):
        self.transform = np.identity(4)
        self._obb = mesh.FullObb(stl_file)
        self.stl_file = self._obb.file
        self.mesh = self._obb.mesh_ct
        self._full_slices = slice.FullSlices(self._obb)
        self._distal_slices = slice.DistalSlices(self._obb)

        self.surgical_neck = surgical_neck.SurgicalNeck(self._full_slices)
        self._proximal_slices = slice.ProximalSlices(
            self._obb, self.surgical_neck, return_odd=False
        )
        self.canal = canal.Canal(self._full_slices)
        self.bicipital_groove = bicipital_groove.DeepGroove(
            self._proximal_slices, self.canal
        )
        self.anatomic_neck = anatomic_neck.AnatomicNeck(
            self._proximal_slices, self.bicipital_groove
        )
        self.trans_epiconylar = epicondyle.TransEpicondylar(
            self._distal_slices, self.canal, self.anatomic_neck
        )
        self.retroversion = bone_props.RetroVersion(
            self.canal, self.anatomic_neck, self.trans_epiconylar
        ).calc
        self.neckshaft = bone_props.NeckShaft(self.canal, self.anatomic_neck).calc

    def apply_csys_canal_transepiconylar(self) -> np.ndarray:
        """applies a coordinate system constructed from the canal axis (z+) and transepicondylar axis (+y) to previously calculated landmarks"""
        self.transform = utils.construct_csys(
            self.canal.axis(), self.trans_epiconylar.axis()
        )
        self._update_landmark_data(self.transform)
        self.mesh = self._obb.mesh_ct.copy().apply_transform(self.transform)
        return self.transform

    def apply_csys_canal_articular(self) -> np.ndarray:
        """applies a coordinate system constructed from the canal axis (+z) and the head central axis(+y) to previously calculated landmarks"""

        self.transform = utils.construct_csys(
            self.canal.axis(), self.anatomic_neck.axis_central()
        )
        self._update_landmark_data(self.transform)
        self.mesh = self._obb.mesh_ct.copy().apply_transform(self.transform)
        return self.transform

    def apply_csys_obb(self) -> np.ndarray:
        """applies a coordinate system constructed from an oriented bounding box to previously calculated landmarks"""
        self.transform = self._obb.transform
        self._update_landmark_data(self.transform)
        self.mesh = self._obb.mesh.copy()
        return self.transform

    def apply_csys_ct(self) -> np.ndarray:
        """applies a the native CT coordinate system to previously calculated landmarks"""
        self.transform = np.identity(4)
        self._update_landmark_data(self.transform)
        self.mesh = self._obb.mesh_ct.copy()
        return self.transform

    def apply_csys_custom(self, transform, from_ct=True):
        """applies a user defined coordinate system defined as a transformation matrix between the CT coordinate system and the user defined coordiante system to previously calculated landmarks"""
        if from_ct:
            self.transform = transform
            self._update_landmark_data(self.transform)
            self.mesh = self._obb.mesh_ct.copy().apply_transform(self.transform)
        else:
            self.transform = np.dot(transform, self.transform)
            self._update_landmark_data(self.transform)
            self.mesh = self.mesh.apply_transform(self.transform)

    def apply_translation(self, translation):
        """applies a user defined translation in the CT coordinate system to previously calculated landmarks"""
        _transform = utils.translate_transform(translation)
        self.transform = np.dot(_transform, self.transform)
        self._update_landmark_data(self.transform)
        self.mesh = self.mesh.apply_transform(self.transform)


class ProximalHumerus(Bone):
    def __init__(self, stl_file):
        self.transform = np.identity(4)
        self._obb = mesh.ProxObb(stl_file)
        self.stl_file = self._obb.file
        self.mesh = self._obb.mesh_ct
        self._full_slices = slice.FullSlices(self._obb)
        self.surgical_neck = surgical_neck.SurgicalNeck(
            self._full_slices,
            only_proximal=True,
        )
        self._proximal_slices = slice.ProximalSlices(
            self._obb,
            self.surgical_neck,
            return_odd=False,
        )
        self.canal = canal.Canal(self._full_slices, proximal=True)
        self.bicipital_groove = bicipital_groove.DeepGroove(
            self._proximal_slices, self.canal
        )
        self.anatomic_neck = anatomic_neck.AnatomicNeck(
            self._proximal_slices, self.bicipital_groove
        )
        self.neckshaft = bone_props.NeckShaft(self.canal, self.anatomic_neck).calc

    def apply_csys_canal_articular(self) -> np.ndarray:
        """applies a coordinate system constructed from the canal axis (+z) and the head central axis(+y) to previously calculated landmarks"""

        self.transform = utils.construct_csys(
            self.canal.axis(), self.anatomic_neck.axis_central()
        )
        self._update_landmark_data(self.transform)
        self.mesh = self._obb.mesh_ct.copy().apply_transform(self.transform)
        return self.transform

    def apply_csys_obb(self) -> np.ndarray:
        """applies a coordinate system constructed from an oriented bounding box to previously calculated landmarks"""

        self.transform = self._obb.transform
        self._update_landmark_data(self.transform)
        self.mesh = self._obb.mesh.copy()
        return self.transform

    def apply_csys_ct(self) -> np.ndarray:
        """applies a the native CT coordinate system to previously calculated landmarks"""

        self.transform = np.identity(4)
        self._update_landmark_data(self.transform)
        self.mesh = self._obb.mesh_ct.copy()
        return self.transform

    def apply_csys_custom(self, transform, from_ct=True):
        """applies a user defined coordinate system defined as a transformation matrix between the CT coordinate system and the user defined coordiante system to previously calculated landmarks"""
        if from_ct:
            self.transform = transform
            self._update_landmark_data(self.transform)
            self.mesh = self._obb.mesh_ct.copy().apply_transform(self.transform)
        else:
            self.transform = np.dot(transform, self.transform)
            self._update_landmark_data(self.transform)
            self.mesh = self.mesh.apply_transform(self.transform)
