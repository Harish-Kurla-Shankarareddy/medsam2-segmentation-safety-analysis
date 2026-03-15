import os
from os.path import join
import csv
import random
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt

from medsam2_infer_3D_CT import build_sam2_video_predictor_npz


# -------------------------
# Utilities
# -------------------------
def window_ct_hu(vol: np.ndarray, hu_min=-125, hu_max=275) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-8)
    return (vol * 255.0).astype(np.uint8)


def resize_grayscale_to_rgb_and_resize(array_zyx_uint8: np.ndarray, image_size: int = 512) -> np.ndarray:
    d, h, w = array_zyx_uint8.shape
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(array_zyx_uint8[i]).convert("RGB").resize((image_size, image_size))
        out[i] = np.array(img).transpose(2, 0, 1)
    return out


def dice_3d(pred: np.ndarray, gt: np.ndarray, eps=1.0) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def dice_2d(a: np.ndarray, b: np.ndarray, eps=1.0) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a & b).sum()
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def worst_slice_dice(pred_zyx: np.ndarray, gt_zyx: np.ndarray):
    """
    Tail-risk metric:
    - compute Dice per slice (2D)
    - ignore slices where BOTH GT and Pred are empty
    - return minimum Dice and its slice index (z)
    """
    D = min(pred_zyx.shape[0], gt_zyx.shape[0])

    dices = []
    zs = []
    for z in range(D):
        if (gt_zyx[z].sum() == 0) and (pred_zyx[z].sum() == 0):
            continue
        dices.append(dice_2d(pred_zyx[z], gt_zyx[z]))
        zs.append(z)

    if len(dices) == 0:
        return 1.0, 0

    idx = int(np.argmin(dices))
    return float(dices[idx]), int(zs[idx])


def getLargestCC(segmentation: np.ndarray) -> np.ndarray:
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation.astype(bool)
    largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
    return largestCC


def pick_key_slice(gt3d: np.ndarray) -> int:
    areas = gt3d.reshape(gt3d.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def bbox2d_from_mask(mask2d: np.ndarray):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def jitter_box(box_xyxy, jitter_px=5, scale_jitter=0.10, W=512, H=512, rng=None):
    """
    box_xyxy = [x1,y1,x2,y2]
    jitter_px: random shift in pixels (+/- jitter_px)
    scale_jitter: random expand/shrink as a fraction of box size
    """
    if rng is None:
        rng = random.Random()

    x1, y1, x2, y2 = map(int, box_xyxy)
    bw = x2 - x1
    bh = y2 - y1

    dx = rng.randint(-jitter_px, jitter_px)
    dy = rng.randint(-jitter_px, jitter_px)

    sx = int(rng.uniform(-scale_jitter, scale_jitter) * bw)
    sy = int(rng.uniform(-scale_jitter, scale_jitter) * bh)

    nx1 = x1 + dx - sx
    ny1 = y1 + dy - sy
    nx2 = x2 + dx + sx
    ny2 = y2 + dy + sy

    nx1 = max(0, min(W - 1, nx1))
    ny1 = max(0, min(H - 1, ny1))
    nx2 = max(0, min(W - 1, nx2))
    ny2 = max(0, min(H - 1, ny2))

    if nx2 <= nx1:
        nx2 = min(W - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H - 1, ny1 + 1)

    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


# -------------------------
# Hausdorff (mm) helpers
# -------------------------
def sitk_from_array_like(arr_zyx: np.ndarray, ref_img: sitk.Image, dtype=sitk.sitkUInt8) -> sitk.Image:
    img = sitk.GetImageFromArray(arr_zyx)
    img = sitk.Cast(img, dtype)
    img.CopyInformation(ref_img)
    return img


def hausdorff_max_mm(pred_zyx: np.ndarray, gt_zyx: np.ndarray, ref_img: sitk.Image) -> float:
    pred_itk = sitk_from_array_like((pred_zyx > 0).astype(np.uint8), ref_img, dtype=sitk.sitkUInt8)
    gt_itk = sitk_from_array_like((gt_zyx > 0).astype(np.uint8), ref_img, dtype=sitk.sitkUInt8)
    f = sitk.HausdorffDistanceImageFilter()
    f.Execute(pred_itk, gt_itk)
    return float(f.GetHausdorffDistance())


def hausdorff95_mm(pred_zyx: np.ndarray, gt_zyx: np.ndarray, ref_img: sitk.Image) -> float:
    pred_itk = sitk_from_array_like((pred_zyx > 0).astype(np.uint8), ref_img, dtype=sitk.sitkUInt8) > 0
    gt_itk = sitk_from_array_like((gt_zyx > 0).astype(np.uint8), ref_img, dtype=sitk.sitkUInt8) > 0

    pred_surf = sitk.LabelContour(pred_itk)
    gt_surf = sitk.LabelContour(gt_itk)

    dt_gt = sitk.SignedMaurerDistanceMap(gt_itk, squaredDistance=False, useImageSpacing=True, insideIsPositive=False)
    dt_pr = sitk.SignedMaurerDistanceMap(pred_itk, squaredDistance=False, useImageSpacing=True, insideIsPositive=False)

    d_pred_to_gt = sitk.GetArrayFromImage(sitk.Abs(dt_gt) * sitk.Cast(pred_surf, sitk.sitkFloat32))
    d_gt_to_pred = sitk.GetArrayFromImage(sitk.Abs(dt_pr) * sitk.Cast(gt_surf, sitk.sitkFloat32))

    d1 = d_pred_to_gt[d_pred_to_gt > 0]
    d2 = d_gt_to_pred[d_gt_to_pred > 0]

    if len(d1) == 0 and len(d2) == 0:
        return 0.0

    all_d = np.concatenate([d1, d2]) if (len(d1) and len(d2)) else (d1 if len(d1) else d2)
    return float(np.percentile(all_d, 95))


# -------------------------
# Uncertainty helpers (NEW)
# -------------------------
def compute_prob_and_uncertainty(masks_list):
    """
    masks_list: list of (D,H,W) uint8 {0,1}
    returns:
      prob: (D,H,W) float32 in [0,1]
      var:  (D,H,W) float32, p*(1-p)
    """
    stack = np.stack(masks_list, axis=0).astype(np.float32)  # (N,D,H,W)
    prob = stack.mean(axis=0)
    var = prob * (1.0 - prob)
    return prob, var


def summarize_uncertainty(var_map, pred_mask=None):
    """
    var_map: (D,H,W) float32
    pred_mask: optional (D,H,W) uint8 {0,1}
    returns: mean_all, max_all, mean_inside, max_inside
    """
    mean_all = float(var_map.mean())
    max_all = float(var_map.max())

    if pred_mask is None:
        return mean_all, max_all, None, None

    inside = var_map[pred_mask > 0]
    mean_inside = float(inside.mean()) if inside.size > 0 else 0.0
    max_inside = float(inside.max()) if inside.size > 0 else 0.0

    return mean_all, max_all, mean_inside, max_inside


def slice_uncertainty(var_map):
    """Return per-slice mean uncertainty and worst slice index."""
    D = var_map.shape[0]
    per_slice = np.array([var_map[z].mean() for z in range(D)], dtype=np.float32)
    worst_z = int(np.argmax(per_slice))
    return per_slice, worst_z, float(per_slice[worst_z])


# -------------------------
# Inference per trial
# -------------------------
def run_one_trial(predictor, vol_t, key_idx, box_xyxy, propagate_reverse=True):
    D = vol_t.shape[0]
    seg = np.zeros((D, 512, 512), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(vol_t, 512, 512)

        _, _, logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=key_idx,
            obj_id=1,
            box=box_xyxy,
        )
        seg[key_idx, (logits[0] > 0).cpu().numpy()[0]] = 1

        for fidx, _, logits in predictor.propagate_in_video(state):
            seg[fidx, (logits[0] > 0).cpu().numpy()[0]] = 1

        if propagate_reverse:
            for fidx, _, logits in predictor.propagate_in_video(state, reverse=True):
                seg[fidx, (logits[0] > 0).cpu().numpy()[0]] = 1

    seg = getLargestCC(seg).astype(np.uint8)
    return seg


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)  # Hydra fix

    CASE = "spleen_2.nii.gz"

    img_path = join(BASE_DIR, "data", "Task09_Spleen", "imagesTr_5", CASE)
    gt_path = join(BASE_DIR, "data", "Task09_Spleen", "labelsTr_5", CASE)
    out_dir = join(BASE_DIR, "results", "prompt_jitter_spleen2_uncertainty")
    os.makedirs(out_dir, exist_ok=True)

    jitter_levels = [0, 2, 5, 10, 20]
    trials_per_level = 5
    scale_jitter = 0.10

    ct_img = sitk.ReadImage(img_path)
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    print("CT shape:", ct.shape, "GT shape:", gt.shape)
    print("CT spacing (mm):", ct_img.GetSpacing())

    ct_u8 = window_ct_hu(ct)
    vol = resize_grayscale_to_rgb_and_resize(ct_u8, 512) / 255.0
    vol_t = torch.from_numpy(vol).cuda()

    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].cuda()
    vol_t = (vol_t - img_mean) / img_std

    cfg_rel = "configs/sam2.1_hiera_t512.yaml"
    ckpt_abs = join(BASE_DIR, "checkpoints", "MedSAM2_latest.pt")
    predictor = build_sam2_video_predictor_npz(cfg_rel, ckpt_abs)

    key_idx = pick_key_slice(gt)
    base_box = bbox2d_from_mask(gt[key_idx])
    if base_box is None:
        raise RuntimeError("GT is empty; cannot create bbox prompt.")
    print("Key slice:", key_idx, "Base bbox:", base_box.tolist())

    csv_path = join(out_dir, "prompt_jitter_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "jitter_px", "trial", "key_slice", "box_xyxy",
            "dice", "fp_voxels", "fn_voxels", "hd_max_mm", "hd95_mm",
            "worst_slice_dice", "worst_slice_z",
            # NEW uncertainty columns (per jitter level duplicated per trial for convenience)
            "mean_unc_all", "max_unc_all", "mean_unc_in_pred", "max_unc_in_pred",
            "worst_unc_slice_z", "worst_unc_slice_mean"
        ])

    all_rows = []

    # for per-jitter plots of uncertainty
    unc_summary_by_jitter = {}  # jpx -> dict of metrics

    for jpx in jitter_levels:
        # collect masks for uncertainty at this jitter level
        masks_for_unc = []

        # store per-trial rows temporarily; we’ll fill uncertainty after we compute it
        rows_this_jitter = []

        for t in range(trials_per_level):
            rng = random.Random(1000 + jpx * 10 + t)
            box = jitter_box(base_box, jitter_px=jpx, scale_jitter=scale_jitter, W=512, H=512, rng=rng)

            seg = run_one_trial(predictor, vol_t, key_idx, box, propagate_reverse=True)

            D = min(seg.shape[0], gt.shape[0])
            pred = seg[:D]
            gtD = gt[:D]

            masks_for_unc.append(pred)

            dsc = dice_3d(pred, gtD)
            fp = int(((pred == 1) & (gtD == 0)).sum())
            fn = int(((pred == 0) & (gtD == 1)).sum())

            hd_max = hausdorff_max_mm(pred, gtD, ct_img)
            hd95 = hausdorff95_mm(pred, gtD, ct_img)

            w_dice, w_z = worst_slice_dice(pred, gtD)

            rows_this_jitter.append([CASE, jpx, t, key_idx, box.tolist(),
                                     dsc, fp, fn, hd_max, hd95,
                                     w_dice, w_z])

            print(
                f"jitter={jpx:>2}px trial={t} "
                f"dice={dsc:.4f} worst={w_dice:.4f}(z={w_z}) "
                f"fp={fp} fn={fn} hd={hd_max:.2f}mm hd95={hd95:.2f}mm"
            )
            torch.cuda.empty_cache()

        # ---- compute uncertainty for this jitter level ----
        prob, var_map = compute_prob_and_uncertainty(masks_for_unc)
        final_mask = (prob >= 0.5).astype(np.uint8)  # majority vote

        mean_u_all, max_u_all, mean_u_in, max_u_in = summarize_uncertainty(var_map, final_mask)
        per_slice_u, worst_u_z, worst_u_val = slice_uncertainty(var_map)

        # save uncertainty + prob maps as NIfTI (so you can view in viewer)
        prob_img = sitk_from_array_like(prob.astype(np.float32), ct_img, dtype=sitk.sitkFloat32)
        unc_img = sitk_from_array_like(var_map.astype(np.float32), ct_img, dtype=sitk.sitkFloat32)
        final_img = sitk_from_array_like(final_mask.astype(np.uint8), ct_img, dtype=sitk.sitkUInt8)

        prob_path = join(out_dir, f"{CASE.replace('.nii.gz','')}_prob_jitter_{jpx}px.nii.gz")
        unc_path = join(out_dir, f"{CASE.replace('.nii.gz','')}_uncertainty_jitter_{jpx}px.nii.gz")
        final_path = join(out_dir, f"{CASE.replace('.nii.gz','')}_finalmask_jitter_{jpx}px.nii.gz")

        sitk.WriteImage(prob_img, prob_path)
        sitk.WriteImage(unc_img, unc_path)
        sitk.WriteImage(final_img, final_path)

        print(f"[UNC] jitter={jpx}px mean_all={mean_u_all:.6f} mean_in={mean_u_in:.6f} worst_slice={worst_u_z} val={worst_u_val:.6f}")
        print("Saved:", prob_path)
        print("Saved:", unc_path)
        print("Saved:", final_path)

        unc_summary_by_jitter[jpx] = dict(
            mean_u_all=mean_u_all, max_u_all=max_u_all,
            mean_u_in=mean_u_in, max_u_in=max_u_in,
            worst_u_z=worst_u_z, worst_u_val=worst_u_val
        )

        # write CSV rows (append uncertainty columns to every row of this jitter)
        for row in rows_this_jitter:
            row_with_unc = row + [mean_u_all, max_u_all, mean_u_in, max_u_in, worst_u_z, worst_u_val]
            all_rows.append(row_with_unc)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row_with_unc)

    print("\nSaved CSV:", csv_path)

    # ---- aggregate helper ----
    def group_metric(col_index):
        d = {}
        for row in all_rows:
            jpx = row[1]
            d.setdefault(jpx, []).append(float(row[col_index]))
        return d

    xs = sorted({r[1] for r in all_rows})

    # Dice plot
    dice_by = group_metric(5)
    mean_d = [float(np.mean(dice_by[x])) for x in xs]
    std_d = [float(np.std(dice_by[x])) for x in xs]

    plt.figure(figsize=(9, 4))
    plt.errorbar(xs, mean_d, yerr=std_d, marker="o", capsize=4)
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("Dice (mean ± std)")
    plt.title("Prompt Jitter Robustness (Dice)")
    plt.grid(True)
    dice_plot = join(out_dir, "jitter_vs_dice.png")
    plt.savefig(dice_plot, dpi=200, bbox_inches="tight")
    plt.show()

    # FP plot
    fp_by = group_metric(6)
    mean_fp = [float(np.mean(fp_by[x])) for x in xs]
    std_fp = [float(np.std(fp_by[x])) for x in xs]

    plt.figure(figsize=(9, 4))
    plt.errorbar(xs, mean_fp, yerr=std_fp, marker="o", capsize=4)
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("False Positive voxels (mean ± std)")
    plt.title("Prompt Jitter Safety: Hallucination Volume (FP)")
    plt.grid(True)
    fp_plot = join(out_dir, "jitter_vs_fp.png")
    plt.savefig(fp_plot, dpi=200, bbox_inches="tight")
    plt.show()

    # FN plot
    fn_by = group_metric(7)
    mean_fn = [float(np.mean(fn_by[x])) for x in xs]
    std_fn = [float(np.std(fn_by[x])) for x in xs]

    plt.figure(figsize=(9, 4))
    plt.errorbar(xs, mean_fn, yerr=std_fn, marker="o", capsize=4)
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("False Negative voxels (mean ± std)")
    plt.title("Prompt Jitter Safety: Missed Organ Volume (FN)")
    plt.grid(True)
    fn_plot = join(out_dir, "jitter_vs_fn.png")
    plt.savefig(fn_plot, dpi=200, bbox_inches="tight")
    plt.show()

    # HD95 plot
    hd95_by = group_metric(9)
    mean_hd95 = [float(np.mean(hd95_by[x])) for x in xs]
    std_hd95 = [float(np.std(hd95_by[x])) for x in xs]

    plt.figure(figsize=(9, 4))
    plt.errorbar(xs, mean_hd95, yerr=std_hd95, marker="o", capsize=4)
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("HD95 (mm)")
    plt.title("Prompt Jitter Safety: Robust Boundary Error (HD95)")
    plt.grid(True)
    hd95_plot = join(out_dir, "jitter_vs_hd95.png")
    plt.savefig(hd95_plot, dpi=200, bbox_inches="tight")
    plt.show()

    # Worst-slice Dice plot
    worst_by = group_metric(10)
    mean_w = [float(np.mean(worst_by[x])) for x in xs]
    std_w = [float(np.std(worst_by[x])) for x in xs]

    plt.figure(figsize=(9, 4))
    plt.errorbar(xs, mean_w, yerr=std_w, marker="o", capsize=4)
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("Worst-slice Dice (mean ± std)")
    plt.title("Tail Risk: Worst Slice Dice vs Prompt Jitter")
    plt.grid(True)
    worst_plot = join(out_dir, "jitter_vs_worst_slice_dice.png")
    plt.savefig(worst_plot, dpi=200, bbox_inches="tight")
    plt.show()

    # Step A: Worst-slice Z distribution per jitter
    worst_z_by = {}
    for row in all_rows:
        jpx = row[1]
        wz = int(row[11])
        worst_z_by.setdefault(jpx, []).append(wz)

    for jpx in xs:
        zs = worst_z_by[jpx]
        plt.figure(figsize=(8, 4))
        plt.hist(zs, bins=15)
        plt.xlabel("Slice index (z)")
        plt.ylabel("Count (trials)")
        plt.title(f"Worst-slice Z Distribution (jitter={jpx}px)")
        plt.grid(True)
        hist_path = join(out_dir, f"worst_z_hist_jitter_{jpx}px.png")
        plt.savefig(hist_path, dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved:", hist_path)

    # NEW: uncertainty summary plots (mean uncertainty vs jitter)
    mean_u = [unc_summary_by_jitter[x]["mean_u_all"] for x in xs]
    mean_u_in = [unc_summary_by_jitter[x]["mean_u_in"] for x in xs]
    worst_u = [unc_summary_by_jitter[x]["worst_u_val"] for x in xs]

    plt.figure(figsize=(9, 4))
    plt.plot(xs, mean_u, marker="o", label="Mean uncertainty (all voxels)")
    plt.plot(xs, mean_u_in, marker="o", label="Mean uncertainty (inside final mask)")
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("Uncertainty (variance p(1-p))")
    plt.title("Uncertainty vs Prompt Jitter")
    plt.grid(True)
    plt.legend()
    unc_plot = join(out_dir, "uncertainty_vs_jitter.png")
    plt.savefig(unc_plot, dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(xs, worst_u, marker="o")
    plt.xlabel("BBox jitter (pixels)")
    plt.ylabel("Worst-slice mean uncertainty")
    plt.title("Tail Uncertainty: Worst Slice Mean Uncertainty vs Jitter")
    plt.grid(True)
    unc_tail_plot = join(out_dir, "worst_slice_uncertainty_vs_jitter.png")
    plt.savefig(unc_tail_plot, dpi=200, bbox_inches="tight")
    plt.show()

    print(
        "Saved plots:\n",
        dice_plot, "\n",
        fp_plot, "\n",
        fn_plot, "\n",
        hd95_plot, "\n",
        worst_plot, "\n",
        unc_plot, "\n",
        unc_tail_plot
    )


if __name__ == "__main__":
    main()
