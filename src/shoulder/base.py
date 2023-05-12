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
    def transform_landmark(self, transform) -> None:
        """transforms the stored axis or points value"""


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

    def _update_landmark_data(self, transform):
        landmarks = self._list_landmarks()
        for land in landmarks:
            land.transform_landmark(transform)

    def _list_landmarks_graph_obj(
        self,
    ) -> typing.List[typing.Union[go.Scatter3d, go.Surface]]:
        """list of all graph objects from each landmark i.e canal, transepicondylar etc."""
        lndmrks = self._list_landmarks()
        return [l._graph_obj() for l in lndmrks if l._graph_obj() is not None]
