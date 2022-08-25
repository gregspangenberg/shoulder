import utils
import canal
import transepicondylar
import left_right
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


class Bone:
    def __init__(self, stl_file) -> None:
        self.name = pathlib.Path(stl_file).stem  # name of bone
        self.file = pathlib.Path(stl_file)  # path to file
        # transform to CT space is 0 rotation, and translation
        self._transform_c = None
        self._transform_e = None
        self._transform_lr = None
        self.canal = None
        self.transepicondylar = None
        self._head_central_mjr = None
        self._head_central_mnr = None
        self._head_central_articular_pt = None
        self._medial_epicondyle_pt = None
        self.head_articular_plane = None
        self._head_articular_plane_pts = None
        self.side = None
        self.version = None
        self.neck_shaft_angle = None

    @property
    def mesh(self):
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f"{self.name} is not watertight!")
        return m

    @property
    def _transform(self):
        transforms = [self._transform_c, self._transform_e, self._transform_lr]
        transforms = [t for t in transforms if t is not None]
        if (
            self._transform_c is None
            and self._transform_e is None
            and self._transform_lr is None
        ):
            return None
        elif self._transform_e is None and self._transform_lr is None:
            return self._transform_c
        elif self._transform_lr is None:
            return np.matmul(self._transform_e, self._transform_c)
        else:
            return np.matmul(
                self._transform_lr, np.matmul(self._transform_e, self._transform_c)
            )

    @decoratortimer(3)
    def canal_calc(self, cutoff_pcts=[0.4, 0.8], num_centroids=50):

        self.canal, self._transform_c = canal.axis(
            self.mesh, cutoff_pcts, num_centroids
        )

        return self.canal

    @decoratortimer(3)
    def transepicondylar_calc(self, num_slice=50):
        if self._transform_c is None:
            raise ValueError("missing transform from canal_calc()")

        self.transepicondylar, self._transform_e = transepicondylar.axis(
            self.mesh, self._transform, num_slice
        )

        return self.transepicondylar
        # add in rotatino to transform so the transepiconylar axis is the x axis

    @decoratortimer(3)
    def left_right_calc(self):
        if self._transform_c is None:
            raise ValueError("missing transform from canal_calc()")
        if self._transform_e is None:
            raise ValueError("missing transform from transepicondylar_calc()")

        (
            self._head_central_mjr,
            self._head_central_mnr,
            self._head_central_articular_pt,
            self._medial_epicondyle_pt,
            self.side,
            self._transform_lr,
        ) = left_right.axis(
            self.mesh, self._transform, self.transepicondylar, slice_num=20
        )
        # this is the final transform needed to define the new coordinate system

        return self.side

    @decoratortimer(3)
    def head_articular_calc(self):
        (
            self.head_articular_plane,
            self._head_articular_plane_pts,
            self.anatomic_neck_shaft_angle,
        ) = head_articular.plane(
            self.mesh,
            self._transform,
            self._head_central_articular_pt,
            self._head_central_mnr,
            self._head_central_mjr,
            self._medial_epicondyle_pt,
            circle_threshold=0.3,
        )

    def calc_features(self):
        self.canal_calc()
        self.transepicondylar_calc()
        self.left_right_calc()
        self.head_articular_calc()

    def export_stl(self, filename):
        if self.transform is None:
            export_mesh = self.mesh
        else:
            export_mesh = self.mesh.apply_transform(self.transform)
        export_mesh.export(filename)

    def export_iges_line(line, filepath):
        utils.write_iges_line(line, filepath)

    @decoratortimer(3)
    def line_plot(self):
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

        if self.transform is not None:
            stl_mesh.transform(self.transform)

        vertices, I, J, K = stl2mesh3d(stl_mesh)
        x, y, z = vertices.T

        # add stl
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, opacity=0.5)])
        # add lines=
        line_list = [
            [self.canal, "canal"],
            [self.transepicondylar, "transepicondylar"],
            [self._head_central_mjr, "head central"],
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
            [self.head_articular_plane, "head articular"],
            [self._head_articular_plane_pts, "head articular points"],
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


class CsysBone(Bone):
    def __init__(self, stl_file, csys=None):
        super().__init__(stl_file)
        self.csys = csys
        
        self.calc_features()
        if self.csys == "transepi":
            print('activated')
            self.transform = (
                self._transform
            )  # the transform created when ID'ing features
        else:
            self.transform = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
        self.transform_to()

    def transform_to(self):
        attributes = [
            "canal",
            "transepicondylar",
            "_head_central_mjr",
            "_head_central_mnr",
            "_head_central_articular_pt",
            "_medial_epicondyle_pt",
            "head_articular_plane",
            "_head_articular_plane_pts",
        ]
        for attr in attributes:
            print(attr)
            feature_pts = getattr(self, attr)
            print(feature_pts)
            feature_pts_csys = utils.transform_pts(feature_pts, self.transform)
            print(feature_pts_csys)
            setattr(self, attr, feature_pts_csys)


np.set_printoptions(suppress=True)

if __name__ == "__main__":
    for stl_bone in pathlib.Path("bones/uncut").glob("*.stl"):
        print(stl_bone.name)
        h = CsysBone(str(stl_bone), csys='transepi')

        # h.calc_features()

        h.line_plot()
        print("\n")

    # h = Humerus("bones/uncut/S202479L_humerus_uncut.stl")

    # h.calc_features()
    # h.line_plot(new_csys=True)
    # h.line_plot(new_csys=False)
