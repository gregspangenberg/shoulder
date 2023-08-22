import shoulder
import pathlib
import numpy as np

stls = pathlib.Path("../shoulder_data/bones/arthritic").glob("*.stl")
for i, stl_bone in enumerate(stls):
    print(stl_bone.name)

    h = shoulder.ProximalHumerus(stl_bone)
    h.canal.axis()
    h.bicipital_groove.axis(cutoff_pcts=[0.3, 0.75], deg_window=7)
    h.apply_csys_canal_articular(np.random.rand(2, 3))

    p = shoulder.Plot(h, opacity=1.0)
    p.figure.write_html("../shoulder_data/viz/" + stl_bone.stem + ".html")
