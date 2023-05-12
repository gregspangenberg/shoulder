import shoulder
import pathlib

# print(pathlib.Path("./tests/test_bones").glob("*.stl"))
for stl_bone in pathlib.Path("./tests/test_bones").glob("*.stl"):
    print(stl_bone.name)

    h = shoulder.Humerus(stl_bone)

    # h.canal.axis([0.5, 0.8])
    h.canal.axis()
    h.trans_epiconylar.axis()
    h.anatomic_neck.plane()
    h.bicipital_groove.axis()
    # h.apply_csys_canal_transepiconylar()

    p = shoulder.Plot(h)

    p.figure.show()

    break
