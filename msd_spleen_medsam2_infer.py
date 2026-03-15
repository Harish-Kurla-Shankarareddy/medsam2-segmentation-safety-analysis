import os
from os.path import join
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import torch
from PIL import Image
from skimage import measure

# Reuse the predictor builder from the repo
# NOTE: medsam2_infer_3D_CT.py must be safe to import (guarded with if __name__ == "__main__":)
from medsam2_infer_3D_CT import build_sam2_video_predictor_npz


def getLargestCC(segmentation: np.ndarray) -> np.ndarray:
    """Keep largest connected component in a 3D binary mask."""
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation.astype(bool)
    largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
    return largestCC


def dice(preds: np.ndarray, targets: np.ndarray, smooth: float = 1.0) -> float:
    preds = (preds > 0).astype(np.uint8)
    targets = (targets > 0).astype(np.uint8)
    inter = (preds * targets).sum()
    return (2.0 * inter + smooth) / (preds.sum() + targets.sum() + smooth)


def window_ct_hu(vol: np.ndarray, hu_min: float = -125, hu_max: float = 275) -> np.ndarray:
    """Clip HU range and map to uint8 [0,255]."""
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-8)
    vol = (vol * 255.0).astype(np.uint8)
    return vol


def resize_grayscale_to_rgb_and_resize(array_zyx_uint8: np.ndarray, image_size: int = 512) -> np.ndarray:
    """(D,H,W) uint8 -> (D,3,S,S) float32 in [0,255]."""
    d, h, w = array_zyx_uint8.shape
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img_pil = Image.fromarray(array_zyx_uint8[i])
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3,S,S)
        out[i] = img_array
    return out


def bbox2d_from_mask(mask2d: np.ndarray):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def pick_key_slice(gt3d: np.ndarray) -> int:
    """Pick slice with maximal foreground area."""
    areas = gt3d.reshape(gt3d.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def resize_mask_slice_nn(mask2d_512: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize for binary mask slice."""
    im = Image.fromarray((mask2d_512 > 0).astype(np.uint8) * 255)
    im = im.resize((out_w, out_h), resample=Image.NEAREST)
    return (np.array(im) > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/MedSAM2_latest.pt", help="checkpoint path")
    ap.add_argument("--cfg", type=str, default="configs/sam2.1_hiera_t512.yaml", help="model config")
    ap.add_argument("-i", "--imgs_path", type=str, required=True, help="path to images folder (nii.gz)")
    ap.add_argument("--gts_path", type=str, required=True, help="path to labels folder (nii.gz)")
    ap.add_argument("-o", "--pred_save_dir", type=str, required=True, help="output folder")
    ap.add_argument("--propagate_with_box", action="store_true", help="also propagate reverse direction")
    ap.add_argument("--hu_min", type=float, default=-125, help="CT window min HU")
    ap.add_argument("--hu_max", type=float, default=275, help="CT window max HU")
    args = ap.parse_args()

    os.makedirs(args.pred_save_dir, exist_ok=True)
    log_path = join(args.pred_save_dir, "dice_log.txt")

    nii_fnames = sorted(
        [f for f in os.listdir(args.imgs_path) if f.endswith(".nii.gz") and not f.startswith("._")]
    )
    print(f"Processing {len(nii_fnames)} nii files")

    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)

    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].cuda()

    # (Optional) clear log each run
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("case\tkey_slice\tdice_native\tpred_native_path\tpred_512_path\n")

    for nii_fname in tqdm(nii_fnames, desc="cases"):
        img_path = join(args.imgs_path, nii_fname)
        gt_path = join(args.gts_path, nii_fname)

        if not os.path.exists(gt_path):
            print("Missing GT for:", nii_fname, "->", gt_path)
            continue

        nii_img = sitk.ReadImage(img_path)
        img_zyx = sitk.GetArrayFromImage(nii_img).astype(np.float32)  # (D,H,W)
        D_native, H_native, W_native = img_zyx.shape

        nii_gt = sitk.ReadImage(gt_path)
        gt_zyx = sitk.GetArrayFromImage(nii_gt).astype(np.uint8)
        gt_zyx = (gt_zyx > 0).astype(np.uint8)

        # preprocess CT -> uint8 volume
        img_u8 = window_ct_hu(img_zyx, hu_min=args.hu_min, hu_max=args.hu_max)

        # pick key slice and bbox prompt (bbox on key slice)
        key_idx = pick_key_slice(gt_zyx)
        bbox = bbox2d_from_mask(gt_zyx[key_idx])
        if bbox is None:
            print("No GT foreground in:", nii_fname)
            continue

        # resize volume to (D,3,512,512), normalize
        vol_resized = resize_grayscale_to_rgb_and_resize(img_u8, 512) / 255.0
        vol_t = torch.from_numpy(vol_resized).cuda()
        vol_t = (vol_t - img_mean) / img_std

        D_model = vol_t.shape[0]
        segs_zyx_512 = np.zeros((D_model, 512, 512), dtype=np.uint8)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(vol_t, 512, 512)

            # prompt at key slice
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_idx,
                obj_id=1,
                box=bbox,
            )
            segs_zyx_512[key_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

            # forward propagation
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
                segs_zyx_512[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

            # optional reverse propagation
            if args.propagate_with_box:
                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                    segs_zyx_512[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

        # keep largest component in 512-space
        segs_zyx_512 = getLargestCC(segs_zyx_512).astype(np.uint8)

        # --- Convert pred 512 -> native HxW ---
        D = min(D_native, segs_zyx_512.shape[0], gt_zyx.shape[0])
        pred_native = np.zeros((D, H_native, W_native), dtype=np.uint8)
        for z in range(D):
            pred_native[z] = resize_mask_slice_nn(segs_zyx_512[z], H_native, W_native)

        # native dice
        dsc_native = dice(pred_native, gt_zyx[:D])

        # save native pred with correct metadata
        out_native_img = sitk.GetImageFromArray(pred_native.astype(np.uint8))
        out_native_img.CopyInformation(nii_img)
        out_native_path = join(args.pred_save_dir, nii_fname.replace(".nii.gz", "_pred_native.nii.gz"))
        sitk.WriteImage(out_native_img, out_native_path)

        # save 512 debug mask (no CopyInformation here to avoid misleading physical space)
        out_512_img = sitk.GetImageFromArray(segs_zyx_512.astype(np.uint8))
        out_512_path = join(args.pred_save_dir, nii_fname.replace(".nii.gz", "_pred_512.nii.gz"))
        sitk.WriteImage(out_512_img, out_512_path)

        # log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{nii_fname}\t{key_idx}\t{dsc_native:.4f}\t{out_native_path}\t{out_512_path}\n")

        torch.cuda.empty_cache()

    print("Done. Results in:", args.pred_save_dir)


if __name__ == "__main__":
    main()
P