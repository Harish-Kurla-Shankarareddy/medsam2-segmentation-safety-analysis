import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ============================================================
# 1) EDIT ONLY THESE PATHS
# ============================================================
CT_PATH   = r"data\Task09_Spleen\imagesTr_5\spleen_2.nii.gz"
GT_PATH   = r"data\Task09_Spleen\labelsTr_5\spleen_2.nii.gz"
PRED_PATH = r"results\msd_spleen_baseline_5\spleen_2_pred_native.nii.gz"

HU_MIN = -125
HU_MAX = 275
# ============================================================


def norm_ct(ct_zyx, hu_min=-125, hu_max=275):
    """Clip HU and normalize to 0..1 for display."""
    x = np.clip(ct_zyx, hu_min, hu_max)
    x = (x - hu_min) / (hu_max - hu_min + 1e-8)
    return x


def dice_2d(gt2d, pr2d, eps=1.0):
    """Dice for 1 slice (2D)."""
    gt2d = (gt2d > 0).astype(np.uint8)
    pr2d = (pr2d > 0).astype(np.uint8)
    inter = (gt2d & pr2d).sum()
    return float((2 * inter + eps) / (gt2d.sum() + pr2d.sum() + eps))


def color_overlay(mask2d, color_rgb, alpha=0.35):
    """Return a transparent RGBA overlay for a binary mask."""
    h, w = mask2d.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[..., :3] = np.array(color_rgb, dtype=np.float32)[None, None, :]
    out[..., 3] = (mask2d > 0).astype(np.float32) * alpha
    return out


def error_overlay(gt2d, pr2d, alpha=0.45):
    """
    False Positive (Pred=1 GT=0) -> red
    False Negative (Pred=0 GT=1) -> blue
    """
    h, w = gt2d.shape
    out = np.zeros((h, w, 4), dtype=np.float32)

    fp = (pr2d == 1) & (gt2d == 0)
    out[fp, :3] = np.array([1.0, 0.0, 0.0])  # red
    out[fp, 3] = alpha

    fn = (pr2d == 0) & (gt2d == 1)
    out[fn, :3] = np.array([0.0, 0.0, 1.0])  # blue
    out[fn, 3] = alpha

    return out


def main():
    print("Loading CT...")
    ct = sitk.GetArrayFromImage(sitk.ReadImage(CT_PATH)).astype(np.float32)  # (D,H,W)

    print("Loading GT...")
    gt = sitk.GetArrayFromImage(sitk.ReadImage(GT_PATH)).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    print("Loading Prediction...")
    pr = sitk.GetArrayFromImage(sitk.ReadImage(PRED_PATH)).astype(np.uint8)
    pr = (pr > 0).astype(np.uint8)

    # Normalize CT for display
    ct = norm_ct(ct, HU_MIN, HU_MAX)

    # Safety: make sure shapes match in depth
    D = min(ct.shape[0], gt.shape[0], pr.shape[0])
    ct, gt, pr = ct[:D], gt[:D], pr[:D]

    # Compute Dice per slice and choose worst slice
    slice_dice = [dice_2d(gt[z], pr[z]) for z in range(D)]
    worst_z = int(np.argmin(slice_dice))
    print("Worst slice:", worst_z, "Dice:", slice_dice[worst_z])

    # Start viewing at worst slice
    z = worst_z

    # Toggle switches
    show_gt = True
    show_pr = True

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("CT + GT(green) + Pred(red) | j/k scroll | g GT | p Pred")

    def draw():
        ax.clear()
        ax.imshow(ct[z], cmap="gray")

        # FP/FN errors always shown
        ax.imshow(error_overlay(gt[z], pr[z]))

        if show_gt:
            ax.imshow(color_overlay(gt[z], (0.0, 1.0, 0.0), alpha=0.30))
        if show_pr:
            ax.imshow(color_overlay(pr[z], (1.0, 0.0, 0.0), alpha=0.30))

        ax.set_title(f"z={z}/{D-1}  Dice={slice_dice[z]:.3f}  (j/k scroll, g GT, p Pred)")
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal z, show_gt, show_pr
        if event.key in ["j", "left"]:
            z = max(0, z - 1)
            draw()
        elif event.key in ["k", "right"]:
            z = min(D - 1, z + 1)
            draw()
        elif event.key == "g":
            show_gt = not show_gt
            draw()
        elif event.key == "p":
            show_pr = not show_pr
            draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
