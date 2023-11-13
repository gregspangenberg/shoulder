import typing
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import pathlib
import numpy as np


class Landmark(ABC):
    @abstractmethod
    def _graph_obj(self) -> typing.Union[go.Scatter3d, go.Surface]:
        """Defines how landmark should be plotted. Must return a graph object"""

    @abstractmethod
    def transform_landmark(self) -> None:
        """calls the function if it has been previously called to update the value with the new transform"""


class Bone(ABC):
    stl_file: typing.Union[str, pathlib.Path]
    transform: np.ndarray

    def _list_landmarks(self) -> typing.List[Landmark]:
        landmarks = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Landmark):
                landmarks.append(attr)
        return landmarks

    def _update_landmark_data(self):
        landmarks = self._list_landmarks()
        for land in landmarks:
            land.transform_landmark()

    def _list_landmarks_graph_obj(
        self,
    ) -> typing.List[typing.Union[go.Scatter3d, go.Surface]]:
        """list of all graph objects from each landmark i.e canal, transepicondylar etc."""
        lndmrks = self._list_landmarks()
        return [l._graph_obj() for l in lndmrks if l._graph_obj() is not None]


class Transform:
    def __init__(self, matrix=None):
        self._matrix = np.identity(4) if matrix is None else matrix

    @property
    def matrix(self) -> np.ndarray:
        """Getter for the transformation matrix."""
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix):
        """Setter for the transformation matrix."""
        if not isinstance(new_matrix, np.ndarray) or new_matrix.shape != (4, 4):
            raise ValueError("Invalid transformation matrix shape")
        self._matrix = new_matrix

    def reset(self):
        """Reset the transformation matrix to the identity matrix."""
        self._matrix = np.identity(4)
