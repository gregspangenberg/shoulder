import shoulder
from shoulder import utils
import pathlib
import numpy as np

print(pathlib.Path("./validate/bones/arthritic").glob("*.stl"))
for stl_bone in pathlib.Path("./validate/bones/arthritic").glob("*.stl"):
    print(stl_bone.name)

    h = shoulder.ProximalHumerus(stl_bone)

    h.canal.axis()
    # h.apply_csys_canal_articular(np.random.rand(2, 3))
    p = shoulder.Plot(h).figure

    p.show()

    print(h.canal._axis)
    break
