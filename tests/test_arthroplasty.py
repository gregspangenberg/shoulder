import shoulder
import pathlib

stls = pathlib.Path("/home/gspangen/projects/shoulder_data/bones/arthritic").glob(
    "*.stl"
)
for i, stl_bone in enumerate(stls):
    # load into shoulder
    print(stl_bone.name)
    if stl_bone.name == "171265R_humerus.stl":
        h = shoulder.Humerus(stl_bone)
    else:
        h = shoulder.ProximalHumerus(stl_bone)

    print(h.retroversion)

    # # pass into arthroplasty
    # otmy = shoulder.HumeralHeadOsteotomy(h)
    # print(otmy.neckshaft_rel)
    # print(otmy.plane)

    # #
    # otmy.offest_neckshaft(10)
    # print(otmy.neckshaft_rel)
    # print(otmy.plane)

    # #
    # h.apply_csys_canal_articular()
    # print(otmy.neckshaft_rel)
    # print(otmy.plane)
    break
