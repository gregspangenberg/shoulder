from shoulder import utils
from shoulder.humerus import canal_epicondyle as ce

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

np.remainder


class MeshLoader:
    def __init__(self, stl_file) -> None:
        self.file = Path(stl_file)
        self.name = Path(stl_file).stem

    @cached_property
    def mesh(self):
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f"{self.name} is not watertight!")
        return m


class LandmarkAxes:
    def __init__(self, mesh_loader: MeshLoader) -> None:
        # orient mesh with a bounding box
        _mesh_pos = ce.MeshObb(mesh_loader.mesh)
        # get canal and transepicondylar axes
        self.canal = ce.Canal(_mesh_pos).axis([0.4, 0.8], 50)
        self.transepicondylar = ce.TransEpicondylar(
            ce.MeshCanal(_mesh_pos, self.canal)
        ).axis(40)


class TransformCsys:
    def __init__(self, landmark_axes: LandmarkAxes):
        self._landmark_axes = landmark_axes

    @property
    def canal_transepi(self):
        # grab values
        cnl = self._landmark_axes.canal
        tep = self._landmark_axes.transepicondylar

        # define center and two axes
        pos = np.average(cnl, axis=0).flatten()
        z_hat = utils.unit_vector(cnl[0], cnl[1])
        x_hat = utils.unit_vector(tep[0], tep[1])

        # calculate remaing axis
        y_hat = np.cross(x_hat, z_hat)
        y_hat /= np.linalg.norm(y_hat)

        # construct transform
        transform = np.c_[x_hat, y_hat, z_hat, pos]
        transform = np.r_[transform, np.array([0, 0, 0, 1]).reshape(1, 4)]

        # if the determinant is 0 then this is a reflection, to undo that the direciton of the
        # epicondylar axis should be switched
        if np.linalg.det(transform) == -1:
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
    file = "tests/test_bones/humerus_right.stl"
    mesh = MeshLoader(file)
    mesh.mesh.show()
    land = LandmarkAxes(mesh)
    tran = TransformCsys(land)
    mesh.mesh.apply_transform(tran.canal_transepi).show()

    plot(file, tran.canal_transepi)
