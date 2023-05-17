from . import utils
from .humerus import mesh
from .humerus import canal
from .humerus import epicondyle
from .humerus import anatomic_neck
from .humerus import bicipital_groove
from .base import Bone

from abc import ABC, abstractmethod
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

# ignore warnings
# This is bad practice but one of the libraries has an issue that generates a nonsense warning
import warnings

warnings.filterwarnings("ignore")


class Humerus(Bone):
    def __init__(self, stl_file):
        self.stl_file = stl_file
        self.transform = np.identity(4)
        msh = mesh.FullObb(stl_file)
        self._mesh = msh

        self.canal = canal.Canal(msh)
        self.trans_epiconylar = epicondyle.TransEpicondylar(msh)
        self.anatomic_neck = anatomic_neck.AnatomicNeck(msh, self.trans_epiconylar)
        self.bicipital_groove = bicipital_groove.DeepGroove(msh)

    def apply_csys_canal_transepiconylar(self) -> np.ndarray:
        self.transform = construct_csys(self.canal._axis, self.trans_epiconylar._axis)
        self._update_landmark_data(self.transform)
        return self.transform


class ProximalHumerus(Bone):
    def __init__(self, stl_file):
        self.stl_file = stl_file
        self.transform = np.identity(4)

        msh = mesh.ProxObb(stl_file)
        self._mesh = msh

        self.canal = canal.Canal(msh)
        self.bicipital_groove = bicipital_groove.DeepGroove(msh)

    def apply_csys_canal_articular(self, articular) -> np.ndarray:
        self.transform = construct_csys(self.canal._axis, articular)
        self._update_landmark_data(self.transform)
        return self.transform


def construct_csys(vec_z, vec_y):
    # define center and two axes
    pos = np.average(vec_z, axis=0)
    pos = pos.flatten()
    z_hat = utils.unit_vector(vec_z[0], vec_z[1])
    x_hat = utils.unit_vector(vec_y[0], vec_y[1])

    # calculate remaing axis
    y_hat = np.cross(x_hat, z_hat)
    y_hat /= np.linalg.norm(y_hat)

    # transepicondylar axis is not quite perpendicular so do it again
    # this is but a temporary fix, maybe switchinf back the transepi to
    # being dependent on the canal would be wise
    x_hat = np.cross(y_hat, z_hat)
    x_hat /= np.linalg.norm(x_hat)

    # construct transform
    transform = np.c_[x_hat, y_hat, z_hat, pos]
    transform = np.r_[transform, np.array([0, 0, 0, 1]).reshape(1, 4)]

    # if the determinant is 0 then this is a reflection, to undo that the direciton of the
    # epicondylar axis should be switched

    if np.round(np.linalg.det(transform)) == -1:
        transform[:, 0] *= -1

    # return transform for CT csys -> canal-epi csys
    transform = utils.inv_transform(transform)
    return transform
