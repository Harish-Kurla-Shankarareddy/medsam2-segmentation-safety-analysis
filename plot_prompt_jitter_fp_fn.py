import os
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "results", "prompt_jitter_spleen2", "prompt_jitter_results.csv")
SAVE_FP = os.path.join(BASE_DIR, "results", "prompt_jitter_spleen2", "jitter_vs_fp.png")
SAVE_FN = os.path.join(BASE_DIR, "results", "prompt_jitter_spleen2", "jitter_vs_fn.png")

def main():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    jitters = sorted({int(r["jitter_px"]) for r in rows})

    def collect(metric):
        out = {}
        for j in jitters:
            vals = [int(r[metric]) for r in rows if int(r["jitter_px"]) == j]
            out[j] = vals
        return out

    fp = collect("fp_voxels")
    fn = collect("fn_voxels")

    xs = jitters
    fp_mean = [float(np.mean(fp[j])) for j in xs]
    fp_std  = [float(np.std(fp[j])) for j in xs]
    fn_mean = [float(np.mean(fn[j])) for j in xs]
    fn_std  = [float(np.std(fn[j])) for j in xs]

    # FP plot (Hallucination)
    plt.figure(figsize=(9, 4))
    plt.errorbar(xs, fp_mean, yerr=fp_std, marker="o", capsize=4)
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("False Positive voxels (mean ± std)")
    plt.title("Prompt Jitter Safety: Hallucination Volume (FP)")
    plt.grid(True)
    plt.savefig(SAVE_FP, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved:", SAVE_FP)

    # FN plot (Miss)
    plt.figure(figsize=(9, 4))
    plt.errorbar(xs, fn_mean, yerr=fn_std, marker="o", capsize=4)
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("False Negative voxels (mean ± std)")
    plt.title("Prompt Jitter Safety: Missed Organ Volume (FN)")
    plt.grid(True)
    plt.savefig(SAVE_FN, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved:", SAVE_FN)

if __name__ == "__main__":
    main()
