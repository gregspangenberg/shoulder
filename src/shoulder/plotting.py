# prevent circular import errors
from __future__ import annotations
import typing

from . import base
from . import arthroplasty

# import other packages
import numpy as np
import trimesh
import plotly.graph_objects as go


def trimesh2plotly(mesh: trimesh.Trimesh):
    vertices = mesh.vertices
    faces = mesh.faces
    trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
    )
    return trace


def bone_mesh_settings(trace):
    trace.color = "#F5ECD6"  # Warm yellow-white color
    trace.lighting = dict(
        ambient=0.3,  # Lower ambient for contrast
        diffuse=0.6,  # Reduced diffuse
        fresnel=0.01,  # Minimal fresnel
        specular=0.05,  # Low specular
        roughness=0.9,  # High roughness
        facenormalsepsilon=1e-15,
        vertexnormalsepsilon=1e-15,
    )
    trace.lightposition = dict(x=300, y=300, z=1200)  # Higher angled light
    trace.flatshading = False
    return trace


class Plot:
    """
    Class for plotting objects in 3D.

    Parameters:
        obj2plot (base.Bone | arthroplasty.HumeralHeadOsteotomy): The object to plot.
        opacity (float): The opacity of the plot (default is 0.7).
    """

    def __init__(
        self,
        obj2plot: base.Bone | arthroplasty.HumeralHeadOsteotomy,
        opacity=0.7,
    ):
        if isinstance(obj2plot, arthroplasty.HumeralHeadOsteotomy):
            self._plotter = PlotSurgery(obj2plot, opacity)
        elif isinstance(obj2plot, base.Bone):
            self._plotter = PlotLandmarks(obj2plot, opacity)
        else:
            raise ValueError("Object to plot must be either a Bone or HumeralHeadOjson")
        self.figure = self._plotter.figure
        self.figure.update_layout(
            title=self._plotter.name,
            scene_aspectmode="data",  # prevents distorition
        )


class PlotSurgery:

    def __init__(
        self,
        ost: arthroplasty.HumeralHeadOsteotomy,
        opacity,
    ):
        self.mesh_top, self.mesh_bot = ost.resect_mesh()
        self.opacity = opacity
        self.name = ost._humerus.stl_file.name

    @property
    def figure(self):

        fig = go.Figure()

        # add meshs
        mesh_top_plot = trimesh2plotly(self.mesh_top)
        mesh_top_plot.opacity = self.opacity
        mesh_top_plot = bone_mesh_settings(mesh_top_plot)

        mesh_bot_plot = trimesh2plotly(self.mesh_bot)
        mesh_bot_plot = bone_mesh_settings(mesh_bot_plot)

        fig.add_traces([mesh_top_plot, mesh_bot_plot])
        return fig


class PlotLandmarks:
    def __init__(
        self,
        bone: base.Bone,
        opacity,
    ):
        self.mesh = bone.mesh
        self.opacity = opacity
        self.name = bone.stl_file.name
        self._landmarks_graph_obj = bone._list_landmarks_graph_obj()

    @property
    def figure(self):

        fig = go.Figure()
        mesh_plot = trimesh2plotly(self.mesh)
        mesh_plot = bone_mesh_settings(mesh_plot)
        mesh_plot.opacity = self.opacity
        fig.add_trace(mesh_plot)

        for lgo in self._landmarks_graph_obj:
            # if more than one graph object per landmark class
            if isinstance(lgo, list):
                for l in lgo:
                    fig.add_trace(l)
            else:
                fig.add_trace(lgo)

        return fig
