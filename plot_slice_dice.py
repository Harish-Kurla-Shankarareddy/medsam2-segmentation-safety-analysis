import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GT_PATH   = os.path.join(BASE_DIR, "data", "Task09_Spleen", "labelsTr_5", "spleen_2.nii.gz")
PRED_PATH = os.path.join(BASE_DIR, "results", "msd_spleen_baseline_5", "spleen_2_pred_native.nii.gz")
SAVE_PATH = os.path.join(BASE_DIR, "results", "msd_spleen_baseline_5", "spleen_2_slice_dice.png")




def dice_2d(a, b, eps=1.0):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a & b).sum()
    return (2 * inter + eps) / (a.sum() + b.sum() + eps)


def main():

    print("Loading GT...")
    gt = sitk.GetArrayFromImage(sitk.ReadImage(GT_PATH)).astype(np.uint8)
    print("Loading Prediction...")
    pr = sitk.GetArrayFromImage(sitk.ReadImage(PRED_PATH)).astype(np.uint8)

    gt = (gt > 0).astype(np.uint8)
    pr = (pr > 0).astype(np.uint8)

    D = min(gt.shape[0], pr.shape[0])
    dices = []
    gt_area = []
    pr_area = []

    for z in range(D):
        dices.append(dice_2d(gt[z], pr[z]))
        gt_area.append(gt[z].sum())
        pr_area.append(pr[z].sum())

    dices = np.array(dices)
    gt_area = np.array(gt_area)
    pr_area = np.array(pr_area)

    # Ignore empty GT slices for meaningful evaluation
    valid_idx = np.where(gt_area > 0)[0]  # original slice numbers where GT exists
    valid_dice = dices[valid_idx]                # dice values at those slices

    worst_pos = int(np.argmin(valid_dice))        # position inside valid list
    best_pos  = int(np.argmax(valid_dice))

    worst_z = int(valid_idx[worst_pos])           # map back to original slice index
    best_z  = int(valid_idx[best_pos])

    print("Worst NON-EMPTY slice:", worst_z, "Dice:", float(valid_dice[worst_pos]))
    print("Best  NON-EMPTY slice:", best_z,  "Dice:", float(valid_dice[best_pos]))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(dices, label="Slice Dice")
    plt.plot(gt_area / (np.max(gt_area) + 1e-8), label="GT area (normalized)")
    plt.plot(pr_area / (np.max(pr_area) + 1e-8), label="Pred area (normalized)")
    plt.xlabel("Slice index (z)")
    plt.ylabel("Value")
    plt.title("Per-slice Dice + GT/Pred area (normalized)")
    plt.legend()
    plt.grid(True)
    plt.axvline(worst_z, linestyle="--", linewidth=1)
    plt.text(worst_z, 0.05, f"worst={worst_z}", rotation=90, va="bottom")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
    print("Saved plot:", SAVE_PATH)

    plt.show()


if __name__ == "__main__":
    main()
