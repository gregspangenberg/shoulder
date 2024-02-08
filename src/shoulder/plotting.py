# prevent circular import errors
from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from . import base

# import other packages
import numpy as np
import stl
import plotly.graph_objects as go


class Plot:
    def __init__(
        self,
        bone: base.Bone,
        opacity=1.0,
    ):
        self.opacity = opacity
        self.stl_name = bone.stl_file.name
        self.transform = bone.transform
        self.stl_mesh = stl.mesh.Mesh.from_file(bone.stl_file)
        self.stl_mesh.transform(self.transform)
        self._landmarks_graph_obj = bone._list_landmarks_graph_obj()

    def stl2mesh3d(self, stl_mesh):
        """
        This function converts an STL mesh into a format that's useable for Plotly's mesh3d.

        Args:
        stl_mesh (numpy-stl object): This is the STL file read by numpy-stl. It contains an array of faces/triangles (i.e., three 3D points).

        Returns:
        vertices (numpy array): This is a numpy array of unique vertices extracted from the STL mesh.
        I, J, K (numpy arrays): These are arrays that define the mesh triangles.
        """

        p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
        # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex
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
                    opacity=self.opacity,
                    color="#DFDAC0",
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
            title=self.stl_name, scene_aspectmode="data"
        )  # plotly defualts into focing 3d plots to be distorted into cubes, this prevents that

        for lgo in self._landmarks_graph_obj:
            # if more than one graph object per landmark class
            if isinstance(lgo, list):
                for l in lgo:
                    fig.add_trace(l)
            else:
                fig.add_trace(lgo)

        return fig
