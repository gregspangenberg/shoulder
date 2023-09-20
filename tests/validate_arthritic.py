import shoulder
import pathlib
import numpy as np
import time

stls = pathlib.Path("/home/gspangen/projects/shoulder_data/bones/arthritic").glob(
    "*.stl"
)
for i, stl_bone in enumerate(stls):
    print(stl_bone.name)
    t0 = time.time()
    h = shoulder.ProximalHumerus(stl_bone)
    h.canal.axis()
    # h.bicipital_groove.axis(cutoff_pcts=[0.3, 0.75], deg_window=7)
    h.apply_csys_canal_articular(np.random.rand(2, 3))

    p = shoulder.Plot(h, opacity=0.9)
    p.figure.write_html(
        "/home/gspangen/projects/shoulder_data/viz/" + stl_bone.stem + ".html"
    )
    print(time.time() - t0)
    break
