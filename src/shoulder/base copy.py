from shoulder import utils
from shoulder.humerus import canal_epicondyle
from shoulder.humerus import transepicondylar
from shoulder.humerus import left_right
from shoulder.humerus import radial_derivative
from shoulder.humerus import head_articular
from shoulder.humerus import angles

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

# This is bad practice but one of the libraries has an issue that generates a nonsense warning
warnings.filterwarnings("ignore")


def timer(decimal):
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
    """holds all attributes inherent to the bone in it's original CT coordinate system"""

    def __init__(self, stl_file) -> None:
        self.name = pathlib.Path(stl_file).stem  # name of bone
        self.file = pathlib.Path(stl_file)  # path to file
        self._transform_c = None
        self._transform_e = None
        self._transform_lr = None
        self._transform_arp = None
        self._transform_nsp = None
        self.canal = None
        self.transepicondylar = None
        self._head_central_mjr = None
        self._head_central_mnr = None
        self._head_central_articular_pt = None
        self._medial_epicondyle_pt = None
        self.head_articular_plane_normal = None
        self.head_articular_plane_pts = None
        self._head_articular_plane_fit_pts = None
        self.bicipital_groove = None
        self.bicipital_groove_pts = None
        self.side = None
        self.retroversion_angle = None
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

    # @decoratortimer(3)
    def canal_calc(self, cutoff_pcts=[0.4, 0.8], num_centroids=50):

        self.canal, self._transform_c = canal_epicondyle.axis(
            self.mesh, cutoff_pcts, num_centroids
        )

        return self.canal

    # @decoratortimer(3)
    def transepicondylar_calc(self, num_slice=50):
        if self._transform_c is None:
            raise ValueError("missing transform from canal_calc()")

        self.transepicondylar, self._transform_e = transepicondylar.axis(
            self.mesh, self._transform, num_slice
        )

        return self.transepicondylar
        # add in rotatino to transform so the transepiconylar axis is the x axis

    def bicipital_groove_calc(self, slice_num=15, interp_num=250):

        self.bicipital_groove, self.bicipital_groove_pts = radial_derivative.axis(
            self.mesh, self._transform, slice_num, interp_num
        )
        return self.bicipital_groove

    # @decoratortimer(3)
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

    # @decoratortimer(3)
    def head_articular_calc(self):
        (
            self.head_articular_plane_normal,
            self.head_articular_plane_pts,
            self._head_articular_plane_fit_pts,
        ) = head_articular.plane(
            self.mesh,
            self._transform,
            self._head_central_articular_pt,
            self._head_central_mnr,
            self._head_central_mjr,
            self._medial_epicondyle_pt,
            circle_threshold=0.3,
        )

    def angles_calc(self):
        self._transform_arp, self.retroversion_angle = angles.retroversion(
            self.head_articular_plane_normal,
            self.side,
            self._transform,
        )
        self.neck_shaft_angle = angles.neck_shaft(
            self.head_articular_plane_normal,
            self._transform,
            self._transform_arp,
        )

    def calc_features(self):
        self.canal_calc()
        self.transepicondylar_calc()
        self.left_right_calc()
        self.bicipital_groove_calc()
        self.head_articular_calc()
        self.angles_calc()

    def export_iges_line(line, filepath):
        utils.write_iges_line(line, filepath)

    # @decoratortimer(3)
    def line_plot(self, show_lines=True):
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
        if show_lines:
            # add lines=
            line_list = [
                [self.canal, "canal"],
                [self.transepicondylar, "transepicondylar"],
                [self._head_central_mjr, "head central"],
                [self.bicipital_groove, "bicipital groove"],
                [self.head_articular_plane_normal, "amp_normal"],
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
                [self.head_articular_plane_pts, "head articular"],
                [self._head_articular_plane_fit_pts, "head articular points"],
                [self.bicipital_groove_pts, "bicipital groove points"],
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

        return fig


class CsysBone(Bone):
    """transform bone into new coordinate system

    Args:
        stl_file (str): path to bone stl
        csys (str, optional): coordinate system to transform to ('transepi','articular')
    """

    def __init__(self, stl_file, csys=None):
        super().__init__(stl_file)
        self.csys = csys

        self.calc_features()
        self.transform_assign()
        self.transform_to()

    def transform_assign(self):
        if self.csys == "transepi":
            # the transform created when ID'ing features
            self.transform = self._transform

        elif self.csys == "articular":
            self.transform = np.matmul(self._transform_arp, self._transform)

        else:
            self.transform = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

    def export_stl(self, filename):
        if self.transform is None:
            export_mesh = self.mesh
        else:
            export_mesh = self.mesh.apply_transform(self.transform)
        export_mesh.export(filename)

    def transform_to(self):
        # this is a super dumb way of doing things make a class that is just attirubtes called bone attribs that inherits everythin from bone. Then apply only to variables local to bone attributes.
        attributes = [
            "canal",
            "transepicondylar",
            "_head_central_mjr",
            "_head_central_mnr",
            "_head_central_articular_pt",
            "_medial_epicondyle_pt",
            "head_articular_plane_normal",
            "head_articular_plane_pts",
            "_head_articular_plane_fit_pts",
        ]
        for attr in attributes:
            feature_pts = getattr(self, attr)
            feature_pts_csys = utils.transform_pts(feature_pts, self.transform)
            setattr(self, attr, feature_pts_csys)