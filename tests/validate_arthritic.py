import shoulder
import pathlib
import numpy as np

stls = pathlib.Path("/home/gspangen/projects/shoulder_data/bones/arthritic").glob(
    "*.stl"
)
for i, stl_bone in enumerate(stls):
    print(stl_bone.name)
    if stl_bone.name == "171265R_humerus.stl":
        h = shoulder.Humerus(stl_bone)
    else:
        h = shoulder.ProximalHumerus(stl_bone)
    h.canal.axis()
    h.bicipital_groove.axis()
    # h.anatomic_neck.points()
    h.apply_csys_obb()
    p = shoulder.Plot(h, opacity=1.0)
    p.figure.write_html(
        "/home/gspangen/projects/shoulder_data/viz/" + stl_bone.stem + ".html"
    )
