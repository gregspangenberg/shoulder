from shoulder import base

import pathlib

if __name__ == "__main__":
    for stl_bone in pathlib.Path("test_bones").glob("*.stl"):
        print(stl_bone.name)
        h = base.CsysBone(str(stl_bone), csys="articular")


        h.line_plot()
        print("\n")

