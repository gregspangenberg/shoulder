import shoulder
from shoulder import utils
import pathlib
import numpy as np


begin = 10
for i, stl_bone in enumerate(pathlib.Path("./validate/bones/arthritic").glob("*.stl")):
    if i < begin:
        continue
    print(stl_bone.name)

    h = shoulder.ProximalHumerus(stl_bone)

    h.canal.axis()
    # print(h._mesh.mesh.bounds)
    h.bicipital_groove.axis()
    h.apply_csys_canal_articular(np.random.rand(2, 3))
    p = shoulder.Plot(h, opacity=1.0).figure

    p.show()

    print(h.canal._axis)
