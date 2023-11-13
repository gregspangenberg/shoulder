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

    cnl = h.canal.axis()
    h.apply_csys_obb()
    print(cnl)
    print(h.canal.axis())
    # h.trans_epiconylar.axis()
    h.apply_csys_obb()
    bg = h.bicipital_groove.axis()
    print(bg)
    h.apply_csys_obb()
    print(h.bicipital_groove.axis())
    print(bg)
    print(type(bg))
    # h.apply_csys_obb()
    print()
    # print(h.anatomic_neck.points())
    # print(h.anatomic_neck.plane())
    h.radius_curvature()
    plane = h.anatomic_neck.plane()
    print(h.anatomic_neck._plane_sk_obb)
    print(np.mean(plane, axis=0))
    print(np.mean(h.anatomic_neck.plane(), axis=0))
    print(np.mean(h.anatomic_neck._plane, axis=0))
    print(np.mean(h.anatomic_neck._plane_ct, axis=0))

    print()
    print(np.mean(plane, axis=0))
    print(np.mean(h.anatomic_neck.plane(), axis=0))
    print(np.mean(h.anatomic_neck._plane, axis=0))
    print(np.mean(h.anatomic_neck._plane_ct, axis=0))

    print(plane.shape)

    p = shoulder.Plot(h, opacity=0.9)
    # p.figure.show()

    p.figure.write_html(
        "/home/gspangen/projects/shoulder_data/viz/" + stl_bone.stem + ".html"
    )
    break
