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

    h.apply_csys_canal_articular()

    ost = shoulder.HumeralHeadOsteotomy(h)
    # ost.offest_neckshaft(10)
    ost.offset_retroversion(-90)
    # ost.offest_neckshaft(-10)
    # ost.offset_retroversion(-10)
    print(h.anatomic_neck.plane())
    print(ost.plane)
    print(ost.neckshaft_rel, ost.retroversion_rel)

    p = shoulder.Plot(h, opacity=0.9)
    # p.figure.show()

    p.figure.write_html(
        "/home/gspangen/projects/shoulder_data/viz/" + stl_bone.stem + ".html"
    )
    break
