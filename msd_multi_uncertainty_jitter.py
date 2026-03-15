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
# Hausdorff 95 (mm)
# -------------------------
def sitk_from_array_like(arr_zyx: np.ndarray, ref_img: sitk.Image, dtype=sitk.sitkUInt8) -> sitk.Image:
    img = sitk.GetImageFromArray(arr_zyx)
    img = sitk.Cast(img, dtype)
    img.CopyInformation(ref_img)
    return img


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
# Uncertainty
# -------------------------
def compute_prob_and_var(masks_list):
    stack = np.stack(masks_list, axis=0).astype(np.float32)  # (N,D,H,W)
    prob = stack.mean(axis=0)
    var = prob * (1.0 - prob)
    return prob, var


def summarize_uncertainty(var_map, final_mask=None):
    mean_all = float(var_map.mean())
    max_all = float(var_map.max())
    if final_mask is None:
        return mean_all, max_all, 0.0, 0.0
    inside = var_map[final_mask > 0]
    mean_in = float(inside.mean()) if inside.size > 0 else 0.0
    max_in = float(inside.max()) if inside.size > 0 else 0.0
    return mean_all, max_all, mean_in, max_in


def slice_uncertainty(var_map):
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


# -------------------------
# Main
# -------------------------
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)  # Hydra fix

    # ---- MSD folders ----
    IM_DIR = join(BASE_DIR, "data", "Task09_Spleen", "imagesTr_5")
    GT_DIR = join(BASE_DIR, "data", "Task09_Spleen", "labelsTr_5")

    OUT_DIR = join(BASE_DIR, "results", "multi_uncertainty")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- experiment settings ----
    jitter_levels = [0, 2, 5, 10, 20]
    trials_per_level = 5
    scale_jitter = 0.10

    # ---- predictor ----
    cfg_rel = "configs/sam2.1_hiera_t512.yaml"
    ckpt_abs = join(BASE_DIR, "checkpoints", "MedSAM2_latest.pt")
    predictor = build_sam2_video_predictor_npz(cfg_rel, ckpt_abs)

    # ---- list cases ----
    cases = sorted([f for f in os.listdir(IM_DIR) if f.endswith(".nii.gz") and not f.startswith("._")])
    print("Found cases:", len(cases))

    # ---- output CSVs ----
    results_csv = join(OUT_DIR, "results_multi_unc.csv")
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "jitter_px", "trial",
            "dice", "fp_voxels", "fn_voxels", "hd95_mm",
            "worst_slice_dice", "worst_slice_z",
            "unc_mean_all", "unc_max_all", "unc_mean_inmask", "unc_max_inmask",
            "unc_worst_slice_z", "unc_worst_slice_mean"
        ])

    # ---- store rows in memory for plotting summaries ----
    rows = []

    # ---- run per case ----
    for ci, case in enumerate(cases):
        img_path = join(IM_DIR, case)
        gt_path = join(GT_DIR, case)
        if not os.path.exists(gt_path):
            print("Skipping (no GT):", case)
            continue

        ct_img = sitk.ReadImage(img_path)
        ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        ct_u8 = window_ct_hu(ct)
        vol = resize_grayscale_to_rgb_and_resize(ct_u8, 512) / 255.0
        vol_t = torch.from_numpy(vol).cuda()

        img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].cuda()
        img_std  = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].cuda()
        vol_t = (vol_t - img_mean) / img_std

        key_idx = pick_key_slice(gt)
        base_box = bbox2d_from_mask(gt[key_idx])
        if base_box is None:
            print("Skipping (empty GT):", case)
            continue

        print(f"\n[{ci+1}/{len(cases)}] {case}  key={key_idx}  bbox={base_box.tolist()}")

        # ---- per jitter ----
        for jpx in jitter_levels:
            # collect predictions for uncertainty ensemble at THIS jitter
            pred_list = []
            metrics_this = []

            for t in range(trials_per_level):
                rng = random.Random(777 + (ci * 10000) + (jpx * 100) + t)
                box = jitter_box(base_box, jitter_px=jpx, scale_jitter=scale_jitter, W=512, H=512, rng=rng)

                seg = run_one_trial(predictor, vol_t, key_idx, box, propagate_reverse=True)

                D = min(seg.shape[0], gt.shape[0])
                pred = seg[:D]
                gtD  = gt[:D]

                pred_list.append(pred)

                dsc = dice_3d(pred, gtD)
                fp = int(((pred == 1) & (gtD == 0)).sum())
                fn = int(((pred == 0) & (gtD == 1)).sum())
                hd95 = hausdorff95_mm(pred, gtD, ct_img)

                w_dice, w_z = worst_slice_dice(pred, gtD)

                metrics_this.append((dsc, fp, fn, hd95, w_dice, w_z))

                torch.cuda.empty_cache()

            # ---- uncertainty for this jitter level (ensemble over trials) ----
            prob, var = compute_prob_and_var(pred_list)
            final_mask = (prob >= 0.5).astype(np.uint8)

            unc_mean_all, unc_max_all, unc_mean_in, unc_max_in = summarize_uncertainty(var, final_mask)
            _, unc_worst_z, unc_worst_val = slice_uncertainty(var)

            # write each trial row (unc repeated per jitter level)
            for t, (dsc, fp, fn, hd95, w_dice, w_z) in enumerate(metrics_this):
                row = [
                    case, jpx, t,
                    float(dsc), int(fp), int(fn), float(hd95),
                    float(w_dice), int(w_z),
                    float(unc_mean_all), float(unc_max_all), float(unc_mean_in), float(unc_max_in),
                    int(unc_worst_z), float(unc_worst_val)
                ]
                rows.append(row)
                with open(results_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

            print(f"  jitter={jpx:>2}px  meanDice={np.mean([m[0] for m in metrics_this]):.4f}  meanUncIn={unc_mean_in:.6f}  worstUncSlice={unc_worst_z}")

    print("\nSaved:", results_csv)

    # -------------------------
    # Summaries
    # -------------------------
    # by jitter
    summary_jitter = join(OUT_DIR, "summary_by_jitter_unc.csv")
    with open(summary_jitter, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jitter_px", "dice_mean", "dice_std", "worst_slice_dice_mean", "hd95_mean",
                    "unc_mean_inmask_mean", "unc_worst_slice_mean_mean"])
        for jpx in jitter_levels:
            r = [x for x in rows if x[1] == jpx]
            dice_vals = [x[3] for x in r]
            worst_vals = [x[7] for x in r]
            hd95_vals = [x[6] for x in r]
            unc_in_vals = [x[11] for x in r]
            unc_worst_vals = [x[14] for x in r]
            w.writerow([
                jpx,
                float(np.mean(dice_vals)), float(np.std(dice_vals)),
                float(np.mean(worst_vals)),
                float(np.mean(hd95_vals)),
                float(np.mean(unc_in_vals)),
                float(np.mean(unc_worst_vals))
            ])

    # by case (average across all jitters/trials)
    summary_case = join(OUT_DIR, "summary_by_case_unc.csv")
    with open(summary_case, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "dice_mean", "worst_slice_dice_mean", "hd95_mean", "unc_mean_inmask_mean"])
        for case in sorted(set([x[0] for x in rows])):
            r = [x for x in rows if x[0] == case]
            w.writerow([
                case,
                float(np.mean([x[3] for x in r])),
                float(np.mean([x[7] for x in r])),
                float(np.mean([x[6] for x in r])),
                float(np.mean([x[11] for x in r])),
            ])

    print("Saved:", summary_jitter)
    print("Saved:", summary_case)

    # -------------------------
    # Plots (simple, clean)
    # -------------------------
    def plot_metric(metric_index, title, ylabel, out_name):
        xs = jitter_levels
        ys = []
        for jpx in xs:
            r = [x for x in rows if x[1] == jpx]
            ys.append(float(np.mean([x[metric_index] for x in r])))
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("BBox jitter (pixels)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        out_path = join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved plot:", out_path)

    plot_metric(3,  "Dice vs Jitter (multi-case)", "Dice", "dice_vs_jitter.png")
    plot_metric(6,  "HD95 vs Jitter (multi-case)", "HD95 (mm)", "hd95_vs_jitter.png")
    plot_metric(7,  "Worst-slice Dice vs Jitter (multi-case)", "Worst-slice Dice", "worst_slice_dice_vs_jitter.png")
    plot_metric(11, "Mean Uncertainty (inside mask) vs Jitter", "Mean uncertainty (p(1-p))", "unc_vs_jitter.png")
    plot_metric(14, "Worst-slice Uncertainty vs Jitter", "Worst-slice mean uncertainty", "worst_slice_unc_vs_jitter.png")


if __name__ == "__main__":
    main()
