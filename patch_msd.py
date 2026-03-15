import re

PATH = "medsam2_infer_3D_CT.py"

with open(PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
patched = False

for line in lines:
    if "range_suffix = re.findall" in line:
        indent = line[: len(line) - len(line.lstrip())]  # keep indentation
        out.append(f"{indent}m = re.findall(r'\\d{{3}}-\\d{{3}}', nii_fname)\n")
        out.append(f"{indent}range_suffix = m[0] if len(m) > 0 else ''\n")
        patched = True
        continue
    out.append(line)

with open(PATH, "w", encoding="utf-8") as f:
    f.writelines(out)

print("PATCHED range_suffix:", patched)
