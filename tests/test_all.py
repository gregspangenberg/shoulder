from shoulder import base

import pathlib

print(pathlib.Path("./tests/test_bones").glob("*.stl"))
for stl_bone in pathlib.Path("./tests/test_bones").glob("*.stl"):
    print(stl_bone.name)
    h = base.CsysBone(str(stl_bone), csys="transepi")


    h.line_plot().show()
    print("\n")

