from shoulder.base import Bone

from abc import ABC, abstractmethod
import numpy as np
import stl
import plotly.graph_objects as go
import pathlib
from typing import Union


class Landmark(ABC):
    @abstractmethod
    def _add_plot(self) -> Union[go.Scatter3d, go.Surface]:
        """Defines how landmark should be plotted. Must return a graph object"""


class Plot:
    def __init__(self, bone: Bone, transform):
        self.stl_name = bone.stl_file.name
        self.stl_mesh = stl.mesh.Mesh.from_file(bone.stl_file)
        self.stl_mesh.transform(transform)

    def stl2mesh3d(stl_mesh):
        # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
        # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
        p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
        # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
        # extract unique vertices from all mesh triangles
        vertices, ixr = np.unique(
            stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
        )
        I = np.take(ixr, [3 * k for k in range(p)])
        J = np.take(ixr, [3 * k + 1 for k in range(p)])
        K = np.take(ixr, [3 * k + 2 for k in range(p)])
        return vertices, I, J, K

    @property
    def figure(self):
        vertices, I, J, K = self.stl2mesh3d(self.stl_mesh)
        x, y, z = vertices.T

        # add stl
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=I,
                    j=J,
                    k=K,
                    # opacity=1.0,
                    opacity=0.7,
                    color="grey",
                    lighting=dict(
                        ambient=0.18,
                        diffuse=0.8,
                        fresnel=0.1,
                        specular=1.2,
                        roughness=0.05,
                        facenormalsepsilon=1e-15,
                        vertexnormalsepsilon=1e-15,
                    ),
                    lightposition=dict(x=1000, y=1000, z=-1000),
                    flatshading=False,
                )
            ]
        )
        fig.update_layout(
            title=self.stl_nameile, scene_aspectmode="data"
        )  # plotly defualts into focing 3d plots to be distorted into cubes, this prevents that
        fig.add_trace()
        return fig

    # figureout how to plot all landmarks
    def add(self, landmark: Landmark):
        self.figure.add_trace(landmark.add_plot())
