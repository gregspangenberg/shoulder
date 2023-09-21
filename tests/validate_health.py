import shoulder
import pathlib

stls = pathlib.Path("/home/gspangen/projects/shoulder_data/bones/non_arthritic").glob(
    "*humerus*.stl"
)
for i, stl_bone in enumerate(stls):
    print(stl_bone.name)
    h = shoulder.Humerus(stl_bone)

    h.canal.axis([0.5, 0.8])
    h.trans_epiconylar.axis()

    h.bicipital_groove.axis(cutoff_pcts=(0.3, 0.75), deg_window=7)

    h.apply_csys_canal_transepiconylar()

    p = shoulder.Plot(h, opacity=0.9)
    p.figure.write_html(
        "/home/gspangen/projects/shoulder_data/viz/" + stl_bone.stem + ".html"
    )
