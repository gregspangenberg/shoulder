import shoulder
import pathlib

stls = pathlib.Path("/home/gspangen/projects/shoulder_data/bones/non_arthritic").glob(
    "*humerus*.stl"
)
for i, stl_bone in enumerate(stls):
    print(stl_bone.name)
    h = shoulder.Humerus(stl_bone)

    h.canal.axis()
    h.trans_epiconylar.axis()

    h.bicipital_groove.axis()
    print(h.anatomic_neck.points())
    print(h.anatomic_neck.plane())

    h.apply_csys_canal_transepiconylar()

    p = shoulder.Plot(h, opacity=0.9)
    # p.figure.show()

    break
    # p.figure.write_html(
    #     "/home/gspangen/projects/shoulder_data/viz/" + stl_bone.stem + ".html"
    # )
