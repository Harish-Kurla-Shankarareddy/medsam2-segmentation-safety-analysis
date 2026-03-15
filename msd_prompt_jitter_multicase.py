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
    - Dice per slice (2D)
    - ignore slices where BOTH GT and Pred are empty
    - return minimum Dice and slice index z
    """
    D = min(pred_zyx.shape[0], gt_zyx.shape[0])
    dices, zs = [], []
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
# Hausdorff (mm)
# -------------------------
def sitk_from_array_like(arr_zyx: np.ndarray, ref_img: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    img.CopyInformation(ref_img)
    return img


def hausdorff_max_mm(pred_zyx: np.ndarray, gt_zyx: np.ndarray, ref_img: sitk.Image) -> float:
    pred_itk = sitk_from_array_like(pred_zyx, ref_img)
    gt_itk = sitk_from_array_like(gt_zyx, ref_img)
    f = sitk.HausdorffDistanceImageFilter()
    f.Execute(pred_itk > 0, gt_itk > 0)
    return float(f.GetHausdorffDistance())


def hausdorff95_mm(pred_zyx: np.ndarray, gt_zyx: np.ndarray, ref_img: sitk.Image) -> float:
    pred_itk = sitk_from_array_like(pred_zyx, ref_img) > 0
    gt_itk = sitk_from_array_like(gt_zyx, ref_img) > 0

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

    # DATA folders (you already created imagesTr_5 / labelsTr_5)
    imgs_dir = join(BASE_DIR, "data", "Task09_Spleen", "imagesTr_5")
    gts_dir = join(BASE_DIR, "data", "Task09_Spleen", "labelsTr_5")

    out_dir = join(BASE_DIR, "results", "prompt_jitter_multicase")
    os.makedirs(out_dir, exist_ok=True)

    # SPEED SETTINGS (good for 8GB GPU)
    MAX_CASES = 10              # increase later to 41 full training set
    jitter_levels = [0, 2, 5, 10, 20]
    trials_per_level = 5
    scale_jitter = 0.10

    # Build predictor ONCE
    cfg_rel = "configs/sam2.1_hiera_t512.yaml"
    ckpt_abs = join(BASE_DIR, "checkpoints", "MedSAM2_latest.pt")
    predictor = build_sam2_video_predictor_npz(cfg_rel, ckpt_abs)

    # Output CSVs
    all_csv = join(out_dir, "results_multi.csv")
    jitter_csv = join(out_dir, "summary_by_jitter.csv")
    case_csv = join(out_dir, "summary_by_case.csv")

    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "jitter_px", "trial", "key_slice", "box_xyxy",
            "dice", "fp_voxels", "fn_voxels", "hd_max_mm", "hd95_mm",
            "worst_slice_dice", "worst_slice_z"
        ])

    # pick cases
    cases = sorted([f for f in os.listdir(imgs_dir) if f.endswith(".nii.gz") and not f.startswith("._")])
    cases = cases[:MAX_CASES]
    print(f"Running multi-case prompt jitter on {len(cases)} cases")
    print("Output:", out_dir)

    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].cuda()

    rows = []

    for case in cases:
        img_path = join(imgs_dir, case)
        gt_path = join(gts_dir, case)
        if not os.path.exists(gt_path):
            print("Missing GT:", case)
            continue

        # Load
        ct_img = sitk.ReadImage(img_path)
        ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        # Preprocess volume -> tensor
        ct_u8 = window_ct_hu(ct)
        vol = resize_grayscale_to_rgb_and_resize(ct_u8, 512) / 255.0
        vol_t = torch.from_numpy(vol).cuda()
        vol_t = (vol_t - img_mean) / img_std

        # Prompt from GT
        key_idx = pick_key_slice(gt)
        base_box = bbox2d_from_mask(gt[key_idx])
        if base_box is None:
            print("Empty GT??", case)
            continue

        for jpx in jitter_levels:
            for t in range(trials_per_level):
                rng = random.Random(hash(case) % 100000 + jpx * 1000 + t)
                box = jitter_box(base_box, jitter_px=jpx, scale_jitter=scale_jitter, W=512, H=512, rng=rng)

                seg = run_one_trial(predictor, vol_t, key_idx, box, propagate_reverse=True)

                D = min(seg.shape[0], gt.shape[0])
                pred = seg[:D]
                gtD = gt[:D]

                dsc = dice_3d(pred, gtD)
                fp = int(((pred == 1) & (gtD == 0)).sum())
                fn = int(((pred == 0) & (gtD == 1)).sum())
                hd_max = hausdorff_max_mm(pred, gtD, ct_img)
                hd95 = hausdorff95_mm(pred, gtD, ct_img)
                w_dice, w_z = worst_slice_dice(pred, gtD)

                row = [
                    case, jpx, t, key_idx, box.tolist(),
                    dsc, fp, fn, hd_max, hd95,
                    w_dice, w_z
                ]
                rows.append(row)

                with open(all_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

                print(f"{case} jitter={jpx:>2} t={t} dice={dsc:.4f} worst={w_dice:.4f} hd95={hd95:.2f} fp={fp} fn={fn}")

                torch.cuda.empty_cache()

    print("\nSaved all trials:", all_csv)

    # -------------------------
    # Summaries
    # -------------------------
    def to_float(x): 
        return float(x)

    # summary by jitter
    jitter_levels_found = sorted(set(r[1] for r in rows))
    with open(jitter_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jitter_px", "n", "dice_mean", "dice_std", "worst_mean", "worst_std", "fp_mean", "fn_mean", "hd95_mean", "hdmax_mean"])
        for jpx in jitter_levels_found:
            sub = [r for r in rows if r[1] == jpx]
            dice_vals = [to_float(r[5]) for r in sub]
            worst_vals = [to_float(r[10]) for r in sub]
            fp_vals = [to_float(r[6]) for r in sub]
            fn_vals = [to_float(r[7]) for r in sub]
            hd95_vals = [to_float(r[9]) for r in sub]
            hdmax_vals = [to_float(r[8]) for r in sub]
            w.writerow([
                jpx, len(sub),
                np.mean(dice_vals), np.std(dice_vals),
                np.mean(worst_vals), np.std(worst_vals),
                np.mean(fp_vals), np.mean(fn_vals),
                np.mean(hd95_vals), np.mean(hdmax_vals)
            ])
    print("Saved:", jitter_csv)

    # summary by case (averaged across all jitters+trials)
    case_names = sorted(set(r[0] for r in rows))
    with open(case_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "n", "dice_mean", "worst_mean", "fp_mean", "fn_mean", "hd95_mean"])
        for case in case_names:
            sub = [r for r in rows if r[0] == case]
            w.writerow([
                case, len(sub),
                np.mean([to_float(r[5]) for r in sub]),
                np.mean([to_float(r[10]) for r in sub]),
                np.mean([to_float(r[6]) for r in sub]),
                np.mean([to_float(r[7]) for r in sub]),
                np.mean([to_float(r[9]) for r in sub]),
            ])
    print("Saved:", case_csv)

    # -------------------------
    # Plots by jitter
    # -------------------------
    def plot_metric(yvals, ylabel, title, fname):
        plt.figure(figsize=(9,4))
        plt.plot(jitter_levels_found, yvals, marker="o")
        plt.xlabel("BBox jitter (pixels)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        outp = join(out_dir, fname)
        plt.savefig(outp, dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved:", outp)

    # read jitter summary back (so plotting is simple)
    jit = []
    dice_m = []
    worst_m = []
    fp_m = []
    fn_m = []
    hd95_m = []
    with open(jitter_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            jit.append(int(float(row["jitter_px"])))
            dice_m.append(float(row["dice_mean"]))
            worst_m.append(float(row["worst_mean"]))
            fp_m.append(float(row["fp_mean"]))
            fn_m.append(float(row["fn_mean"]))
            hd95_m.append(float(row["hd95_mean"]))

    plot_metric(dice_m, "Mean Dice", "Multi-case: Dice vs Prompt Jitter", "dice_vs_jitter.png")
    plot_metric(worst_m, "Mean Worst-slice Dice", "Multi-case Tail Risk: Worst-slice Dice vs Jitter", "worst_slice_dice_vs_jitter.png")
    plot_metric(fp_m, "Mean FP voxels", "Multi-case Safety: Hallucination (FP) vs Jitter", "fp_vs_jitter.png")
    plot_metric(fn_m, "Mean FN voxels", "Multi-case Safety: Missed Organ (FN) vs Jitter", "fn_vs_jitter.png")
    plot_metric(hd95_m, "Mean HD95 (mm)", "Multi-case Boundary Robustness: HD95 vs Jitter", "hd95_vs_jitter.png")

    print("\nDONE.")


if __name__ == "__main__":
    main()
