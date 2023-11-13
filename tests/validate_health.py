import shoulder
import pathlib
import numpy as np

stls = pathlib.Path("/home/gspangen/projects/shoulder_data/bones/non_arthritic").glob(
    "*humerus*.stl"
)
for i, stl_bone in enumerate(stls):
    print()
    print(stl_bone.name)
    h = shoulder.Humerus(stl_bone)
    print(h.anatomic_neck.points())

    h.apply_csys_canal_articular()
    print(h.anatomic_neck.points())

    h.apply_csys_obb()
    print(h.anatomic_neck.points())

    # cnl = h.canal.axis()
    # print(cnl)
    # h.apply_csys_obb()
    # cnl = h.canal.axis()
    # print(cnl)

    p = shoulder.Plot(h, opacity=0.9)
    # p.figure.show()

    p.figure.write_html(
        "/home/gspangen/projects/shoulder_data/viz/" + stl_bone.stem + ".html"
    )
    break
