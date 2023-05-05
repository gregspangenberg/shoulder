from . import utils
from .humerus import mesh
from .humerus import canal
from .humerus import epicondyle
from pathlib import Path
import trimesh
import plotly.graph_objects as go
import numpy as np
import stl
from functools import cached_property

# ignore warnings
# This is bad practice but one of the libraries has an issue that generates a nonsense warning
import warnings

warnings.filterwarnings("ignore")


class Humerus:
    def __init__(self, stl_file):
        msh = mesh.FullObb(stl_file)
        cnl = canal.Canal(msh)

        self.canal_axis = cnl.axis
        self.tep_axis = epicondyle.TransEpicondylar(msh, cnl).axis

    def canal_transepi_csys(self):
        return construct_csys(self.canal_axis, self.tep_axis)


class ProximalHumerus:
    def __init__(self, stl_file):
        msh = mesh.ProxObb(stl_file)
        cnl = canal.Canal(msh)
        self.canal_axis = cnl.axis(msh.cutoff_pcts)


def construct_csys(vec_z, vec_y):
    # define center and two axes
    pos = np.average(vec_z, axis=0).flatten()
    z_hat = utils.unit_vector(vec_z[0], vec_z[1])
    x_hat = utils.unit_vector(vec_y[0], vec_y[1])

    # calculate remaing axis
    y_hat = np.cross(x_hat, z_hat)
    y_hat /= np.linalg.norm(y_hat)

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


def plot(stl_file, csys_transform):
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

    # load into numpy-stl
    stl_mesh = stl.mesh.Mesh.from_file(stl_file)
    stl_mesh.transform(csys_transform)

    vertices, I, J, K = stl2mesh3d(stl_mesh)
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
        title=stl_file, scene_aspectmode="data"
    )  # plotly defualts into focing 3d plots to be distorted into cubes, this prevents that

    fig.show()


if __name__ == "__main__":
    # file = "tests/test_bones/humerus_right.stl"
    # mesh = MeshLoader(file)
    # land = LandmarkAxes(mesh)
    # tran = TransformCsys(land)
    # plot(file, tran.canal_transepi)
    """"""
