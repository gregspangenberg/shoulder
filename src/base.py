import utils
import canal
import transepicondylar
import head_central
import head_articular

import time
import pathlib
import trimesh
import pathlib
import plotly.graph_objects as go
import skspatial.objects
import numpy as np
import stl
from functools import cached_property

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


def decoratortimer(decimal):
    def decoratorfunction(f):
        def wrap(*args, **kwargs):
            time1 = time.monotonic()
            result = f(*args, **kwargs)
            time2 = time.monotonic()
            print(
                "{:s} function took {:.{}f} ms".format(
                    f.__name__, ((time2 - time1) * 1000), decimal
                )
            )
            return result

        return wrap

    return decoratorfunction


class Humerus:
    def __init__(self, stl_file) -> None:
        self.name = pathlib.Path(stl_file).stem  # name of bone
        self.file = pathlib.Path(stl_file)  # path to file
        self.transform = None
        self._transform_c = None
        self._transform_e = None
        self.canal = None
        self.canal_csys = None
        self.transepicondylar = None
        self.transepicondylar_csys = None
        self.head_central = None
        self.head_central_csys = None
        self._head_central_minor_axis = None
        self._head_central_articular_pt = None
        self._medial_epicondyle_pt = None
        self.version = None
        self.head_articular_plane = None
        self.head_articular_plane_csys = None
        self._head_articular_plane_pts = None
        self._head_articular_plane_pts_csys = None
        self.anatomic_neck_shaft_angle = None

    @property
    def mesh(self):
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f"{self.name} is not watertight!")
        return m

    @cached_property
    def mesh_new(self):
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f"{self.name} is not watertight!")
        return m

    @decoratortimer(3)
    def canal_calc(self, cutoff_pcts=[0.4, 0.8], num_centroids=50):
        self.canal, c_transform = canal.axis(self.mesh, cutoff_pcts, num_centroids)
        self.mesh_new = self.mesh_new.apply_transform(c_transform)
        self._transform_c = c_transform
        self.transform = c_transform
        self.canal_csys = utils.transform_pts(self.canal, self.transform)

        return self.canal

    @decoratortimer(3)
    def transepicondylar_calc(self, num_slice=50):
        self.transepicondylar, e_transform = transepicondylar.axis(
            self.mesh, self.transform, num_slice
        )
        self.mesh_new = self.mesh_new.apply_transform(e_transform)
        self._transform_e = e_transform
        self.transform = np.matmul(self._transform_e, self._transform_c)
        self.transepicondylar_csys = utils.transform_pts(
            self.transepicondylar, self.transform
        )

        return self.transepicondylar
        # add in rotatino to transform so the transepiconylar axis is the x axis

    @decoratortimer(3)
    def head_central_calc(self):
        (
            self.head_central,
            self._head_central_minor_axis,
            self.version,
            self._head_central_articular_pt,
            self._medial_epicondyle_pt,
        ) = head_central.axis(
            self.mesh, self.transform, self.transepicondylar_csys, slice_num=20
        )
        self.head_central_csys = utils.transform_pts(self.head_central, self.transform)

        return self.head_central

    @decoratortimer(3)
    def head_articular_calc(self):
        (
            self.head_articular_plane,
            self._head_articular_plane_pts,
            self.anatomic_neck_shaft_angle,
        ) = head_articular.plane(
            self.mesh,
            self.transform,
            self._head_central_articular_pt,
            self._head_central_minor_axis,
            self.head_central,
            self._medial_epicondyle_pt,
            circle_threshold=0.3,
        )
        self.head_articular_plane_csys = utils.transform_pts(
            self.head_articular_plane, self.transform
        )
        self._head_articular_plane_pts_csys = utils.transform_pts(
            self._head_articular_plane_pts, self.transform
        )

    def create_csys(self):
        self.canal_calc()
        self.transepicondylar_calc()
        self.head_central_calc()
        self.head_articular_calc()

    def export_stl_new_csys(self, filename):
        export_mesh = self.mesh_new
        export_mesh.export(filename)

    def export_iges_line(line, filepath):
        utils.write_iges_line(line, filepath)

    @decoratortimer(3)
    def line_plot(self, new_csys=False):
        def stl2mesh3d(stl_mesh):
            # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
            # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
            # stl_mesh = mesh.Mesh.from_file(stl_mesh)
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
        stl_mesh = stl.mesh.Mesh.from_file(self.file)

        if new_csys:
            stl_mesh.transform(self.transform)
            canal = self.canal_csys
            transepicondylar = self.transepicondylar_csys
            head_central = self.head_central_csys
            head_articular = self.head_articular_plane_csys
            head_articular_pts = self._head_articular_plane_pts_csys

        else:
            canal = self.canal
            transepicondylar = self.transepicondylar
            head_central = self.head_central
            head_articular = self.head_articular_plane
            head_articular_pts = self._head_articular_plane_pts

        vertices, I, J, K = stl2mesh3d(stl_mesh)
        x, y, z = vertices.T

        # add stl
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, opacity=0.5)])
        # add lines=
        line_list = [
            [canal, "canal"],
            [transepicondylar, "transepicondylar"],
            [head_central, "head central"],
        ]
        line_list = [
            x for x in line_list if x[0] is not None
        ]  # remove ones which don't have values yet
        for line in line_list:
            fig.add_trace(
                go.Scatter3d(
                    x=line[0][:, 0], y=line[0][:, 1], z=line[0][:, 2], name=line[1]
                )
            )

        # add planes
        plane_list = [
            [head_articular, "head articular"],
            [head_articular_pts, "head articular points"],
        ]
        plane_list = [x for x in plane_list if x[0] is not None]
        for plane in plane_list:

            fig.add_trace(
                go.Scatter3d(
                    x=plane[0][:, 0],
                    y=plane[0][:, 1],
                    z=plane[0][:, 2],
                    name=plane[1],
                    mode="markers",
                )
            )

        fig.update_layout(
            title=self.name, scene_aspectmode="data"
        )  # plotly defualts into focing 3d plots to be distorted into cubes, this prevents that

        fig.show()


np.set_printoptions(suppress=True)

if __name__ == "__main__":
    for stl_bone in pathlib.Path("bones/uncut").glob("*.stl"):
        print(stl_bone.name)
        h = Humerus(str(stl_bone))

        h.create_csys()

        h.line_plot(new_csys=True)
        print("\n")

    # h = Humerus("bones/uncut/S202479R_humerus_uncut.stl")

    # h.create_csys()
    # h.line_plot(new_csys=True)
    # h.line_plot(new_csys=False)
