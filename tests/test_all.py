import shoulder

import pathlib

print(pathlib.Path("./tests/test_bones").glob("*.stl"))
for stl_bone in pathlib.Path("./tests/test_bones").glob("*.stl"):
    print(stl_bone.name)
    h = shoulder.Humerus(stl_bone)
    print(dir(h))
    break

#     print(h.canal_axis)
#     shoulder.plot(str(stl_bone), h.canal_transepi_csys())
#     print("\n")
# #
