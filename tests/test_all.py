import shoulder

import pathlib

print(pathlib.Path("./tests/test_bones").glob("*.stl"))
for stl_bone in pathlib.Path("./tests/test_bones").glob("*.stl"):
    print(stl_bone.name)
    h = shoulder.Humerus(str(stl_bone))

    shoulder.plot(str(stl_bone), h.canal_transepi_csys())
    print("\n")
#
