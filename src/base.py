import centerlines
import transepicondylar
import head_central

import trimesh
import pathlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import numpy as np
from stl import mesh
from functools import cached_property

class Humerus():
    def __init__(self,stl_file) -> None:
        self.name = pathlib.Path(stl_file).stem #name of bone
        self.file = pathlib.Path(stl_file) # path to file
        self.transform = None
        self._transform_c = None
        self._transform_e = None
        self.centerline = None
        self.centerline_csys = None
        self.transepicondylar = None
        self.transepicondylar_csys = None
        self.head_central = None
        self.head_central_csys = None
        self.head_central_articular_pt = None
        self.version = None

    @property
    def mesh(self):
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f'{self.name} is not watertight!')
        return m

    @cached_property
    def mesh_new(self):
        m = trimesh.load_mesh(str(self.file))
        if not m.is_watertight:
            warnings.warn(f'{self.name} is not watertight!')
        return m
    
    def centerline_calc(self, cutoff_pcts=[0.4,0.8], num_centroids=50):
        self.centerline, c_transform = centerlines.axis(self.mesh, cutoff_pcts, num_centroids)
        self.mesh_new = self.mesh_new.apply_transform(c_transform)
        self._transform_c = c_transform
        self.transform = c_transform
        self.centerline_csys = transform_pts(self.centerline, self.transform)

        return self.centerline

    def transepicondylar_calc(self, num_slice=50):
        self.transepicondylar, e_transform = transepicondylar.axis(self.mesh, self.transform, num_slice)
        self.mesh_new = self.mesh_new.apply_transform(e_transform)
        self._transform_e = e_transform
        self.transform = np.matmul(self._transform_e, self._transform_c)
        self.transepicondylar_csys = transform_pts(self.transepicondylar, self.transform)

        return self.transepicondylar
        # add in rotatino to transform so the transepiconylar axis is the x axis

    def head_central_calc(self):
        self.head_central, self.version, self.head_central_articular_pt = head_central.axis(self.mesh, self.transform)
        self.head_central_csys = transform_pts(self.head_central, self.transform)

        return self.head_central

    def create_csys(self):
        self.centerline_calc()
        self.transepicondylar_calc()
        self.head_central_calc()
    
    
    def export_stl_new_csys(self, filename):
        export_mesh = self.mesh_new
        export_mesh.export(filename)
        

    def export_iges_line(line, filepath):
        x,y,z = line[0]
        x1,y1,z1 = line[1]

        # i known  this string looks jank but it has to be this way for the whitespace to be printed correct
        s = """                                                                        S0000001
1H,,1H;,8Hpart.mco,10Hglobal.tmp,21HMedcad by Materialise,              G0000001
24HMedical CAD Modelling sw,32,38,8,308,16,8Hpart.med,1,2,2HMM,3,0.5,   G0000002
13H220408.155753,9.9999999E-09,1000,2HPN,14HMaterialise nv,1,0;         G0000003
    110       1       0       1       0       0       0       000000000D0000001
    110       0       0       1       0                    LINE       0D0000002\n"""
        s1 = f'110,{x},{y},{z},{x1},{y1},{z1};'
        s1 = s1.ljust(71)+'1P0000001\n'
        s2 = 'S      1G      3D      2P      1                                        T0000001'
        iges = s+s1+s2

        with open(filepath,'w') as f:
            f.write(iges)

    def line_plot(self, new_csys=False):

        def stl2mesh3d(stl_mesh):
            # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points) 
            # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
            # stl_mesh = mesh.Mesh.from_file(stl_mesh)
            p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
            # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
            # extract unique vertices from all mesh triangles
            vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
            I = np.take(ixr, [3*k for k in range(p)])
            J = np.take(ixr, [3*k+1 for k in range(p)])
            K = np.take(ixr, [3*k+2 for k in range(p)])
            return vertices, I, J, K
        
        # load into numpy-stl
        stl_mesh = mesh.Mesh.from_file(self.file)

        if new_csys:
            stl_mesh.transform(self.transform)
            centerline = self.centerline_csys
            transepicondylar = self.transepicondylar_csys
            head_central = self.head_central_csys
        
        else:
            centerline = self.centerline
            transepicondylar = self.transepicondylar
            head_central =self.head_central

        vertices, I, J, K = stl2mesh3d(stl_mesh)
        x, y, z = vertices.T

        # add stl
        fig = go.Figure(
            data = [go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, opacity=0.4)]
        )
        # add lines
        line_list = [
            [centerline,'canal'],
            [transepicondylar,'transepicondylar'],
            [head_central, 'head central']
        ]
        line_list = [x for x in line_list if x[0] is not None] # remove ones which don't have values yet
        for line in line_list:
            fig.add_trace(go.Scatter3d(x=line[0][:,0], y=line[0][:,1], z=line[0][:,2], name=line[1]))

        fig.update_layout(scene_aspectmode='data') # plotly defualts into focing 3d plots to be distorted into cubes, this prevents that

        fig.show()


def transform_pts(pts, transform):
    """Applies a transform to a set of xyz points

    Args:
        pts (np.array [nx3]): points to transform
        transform (np.array [4x4]): transformation matrix

    Returns:
        pts_transform(np.array [nx3]): transformed points
    """
    pts = np.c_[pts, np.ones(len(pts))].T
    pts = np.matmul(transform,pts)
    pts = pts.T # transpose back
    pts_transform = np.delete(pts,3,axis=1) # remove added ones now that transform is complete
    return pts_transform   

def rev_transform(transform):
    """reverses a transformation matrix

    Args:
        transform (np.array [4x4]): transformation matrix

    Returns:
        transform (np.array [4x4]): reversed transformation matrix
    """
    translate = transform[:3,-1]
    translate = np.c_[np.identity(3),translate]
    translate = np.r_[translate,np.array([[0,0,0,1]])]

    rotate = transform[:,:-1]
    rotate = np.c_[rotate,np.array([[0],[0],[0],[1]])]

    transform = np.matmul(np.linalg.inv(rotate), np.linalg.inv(translate))
    
    return transform

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    # h = Humerus('S202017L_humerus_uncut.stl')
    h = Humerus('S202501R_humerus_uncut.stl')
    # h = Humerus('S202479L_humerus.stl')

    h.centerline_calc([0.4,0.8])
    h.transepicondylar_calc()
    h.head_central_calc()

    print(f'centerline:\n{h.centerline}\ntransepicondylar:\n{h.transepicondylar}\nhead central:\n{h.head_central}')
    # print(f'centerline:\n{h.centerline_csys}\ntransepicondylar:\n{h.transepicondylar_csys}\nhead central:\n{h.head_central_csys}')
    print(f'verion:\n{h.version}')
    print(f'articular_pt:\n{h.head_central_articular_pt}')
    h.line_plot()
    h.line_plot(new_csys=True)
