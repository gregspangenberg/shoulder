import shoulder
import pathlib

stls = pathlib.Path("./validate/bones/arthritic").glob("*.stl")
for i, stl_bone in enumerate(stls):
    # bad_arthritic = [
    #     "JAT-L_humerus",
    #     "JBW-R_humerus",
    #     "KS-L_humerus",
    #     "WGW-L_humerus",
    #     "CGP-R_humerus",
    #     "FTJ-R_humerus",
    #     "RWM-R_humerus",
    # ]
    # if stl_bone.stem not in bad_arthritic:
    #     continue
    print(stl_bone.name)

    # h = shoulder.Humerus(stl_bone)
    h = shoulder.ProximalHumerus(stl_bone)

    # h.canal.axis([0.5, 0.8])
    h.canal.axis()
    # h.trans_epiconylar.axis()
    # h.anatomic_neck.plane()
    h.bicipital_groove.axis(cutoff_pcts=[0.3, 0.75], deg_window=7)
    # h.apply_csys_canal_transepiconylar()
    # h.apply_csys_canal_articular(np.random.rand)

    p = shoulder.Plot(h, opacity=1.0)

    # p.figure.show()
    # break
    p.figure.write_html("./validate/viz/arthritic/" + stl_bone.stem + ".html")
    # break
