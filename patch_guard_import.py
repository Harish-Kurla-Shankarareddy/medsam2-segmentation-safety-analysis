PATH = "medsam2_infer_3D_CT.py"

with open(PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# We will insert: if __name__ == "__main__": before the "DL_info =" line (start of script execution)
start_idx = None
for i, line in enumerate(lines):
    if line.lstrip().startswith("DL_info"):
        start_idx = i
        break

if start_idx is None:
    raise RuntimeError("Could not find 'DL_info' line to anchor the guard. Open the file and search for DL_info.")

# If it's already guarded, do nothing
already_guarded = any("__name__" in l and "__main__" in l for l in lines[max(0, start_idx-5): start_idx+5])
if already_guarded:
    print("Already guarded. No changes made.")
else:
    guarded = []
    guarded.extend(lines[:start_idx])
    guarded.append("\nif __name__ == \"__main__\":\n")
    # indent the rest of the file so it runs only when executed as a script
    for l in lines[start_idx:]:
        if l.strip() == "":
            guarded.append(l)
        else:
            guarded.append("    " + l)

    with open(PATH, "w", encoding="utf-8") as f:
        f.writelines(guarded)

    print("Patched: medsam2_infer_3D_CT.py is now safe to import.")
