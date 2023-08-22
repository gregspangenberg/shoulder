import shoulder
import pathlib
import numpy as np

stls = pathlib.Path("./validate/bones/arthritic").glob("*.stl")
for i, stl_bone in enumerate(stls):
    # not a thing anymore all are good!
    # bad_arthritic = [
    #     "CGP-R_humerus",
    #     "CJB-L_humerus",
    #     "JDH-R_humerus",
    #     "NIS-L_humerus",
    #     "WMM-R_humerus",
    # ]
    # if stl_bone.stem not in bad_arthritic:
    #     continue
    print(stl_bone.name)

    h = shoulder.ProximalHumerus(stl_bone)
    h.canal.axis()
    h.bicipital_groove.axis(cutoff_pcts=[0.3, 0.75], deg_window=7)
    h.apply_csys_canal_articular(np.random.rand(2, 3))

    p = shoulder.Plot(h, opacity=1.0)
    p.figure.write_html("./validate/viz/arthritic/" + stl_bone.stem + ".html")
