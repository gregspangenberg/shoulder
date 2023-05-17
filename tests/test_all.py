import shoulder
import pathlib

stls = pathlib.Path("./validate/bones/non_arthritic").glob("*.stl")
begin = 17
for i, stl_bone in enumerate(stls):
    if i < begin:
        continue
    print(stl_bone.name)

    h = shoulder.Humerus(stl_bone)

    # h.canal.axis([0.5, 0.8])
    h.canal.axis()
    h.trans_epiconylar.axis()
    h.anatomic_neck.plane()
    h.bicipital_groove.axis(cutoff_pcts=[0.2, 0.85])
    h.apply_csys_canal_transepiconylar()

    p = shoulder.Plot(h, opacity=1.0)

    p.figure.show()

    break
