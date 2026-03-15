import os
from os.path import join
import numpy as np
import SimpleITK as sitk
from PIL import Image

def dice(a, b, eps=1e-8):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a * b).sum()
    return (2 * inter + eps) / (a.sum() + b.sum() + eps)

def resize_mask_slice_nn(mask2d, out_h, out_w):
    # mask2d: HxW uint8/bool
    im = Image.fromarray((mask2d > 0).astype(np.uint8) * 255)
    im = im.resize((out_w, out_h), resample=Image.NEAREST)
    return (np.array(im) > 0).astype(np.uint8)

def main():
    imgs_path = r"data\Task09_Spleen\imagesTr_5"
    gts_path  = r"data\Task09_Spleen\labelsTr_5"
    pred_dir  = r"results\msd_spleen_baseline_5"

    out_native_dir = join(pred_dir, "native_space")
    os.makedirs(out_native_dir, exist_ok=True)

    fnames = sorted([f for f in os.listdir(imgs_path) if f.endswith(".nii.gz") and not f.startswith("._")])
    lines = []

    for f in fnames:
        img_path = join(imgs_path, f)
        gt_path  = join(gts_path, f)

        pred_path = join(pred_dir, f.replace(".nii.gz", "_medsam2_mask.nii.gz"))
        if not os.path.exists(pred_path):
            print("Missing pred:", pred_path)
            continue

        # Read original reference (for native size + metadata)
        ref_img = sitk.ReadImage(img_path)
        ref_arr = sitk.GetArrayFromImage(ref_img)  # (D,H,W)
        D_ref, H_ref, W_ref = ref_arr.shape

        # Read GT in native space
        gt_img = sitk.ReadImage(gt_path)
        gt_arr = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)

        # Read predicted mask (currently 512-space)
        pred_img = sitk.ReadImage(pred_path)
        pred_arr = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)  # (D,512,512) expected

        # Safety: align depth if needed
        D = min(D_ref, pred_arr.shape[0], gt_arr.shape[0])

        # Resize each slice to native HxW
        pred_native = np.zeros((D, H_ref, W_ref), dtype=np.uint8)
        for z in range(D):
            pred_native[z] = resize_mask_slice_nn(pred_arr[z], H_ref, W_ref)

        # Compute Dice in native voxel grid
        dsc = dice(pred_native, gt_arr[:D])

        # Save native-space prediction with correct metadata
        out_img = sitk.GetImageFromArray(pred_native.astype(np.uint8))
        out_img.CopyInformation(ref_img)  # spacing/origin/direction consistent with native CT

        out_path = join(out_native_dir, f.replace(".nii.gz", "_pred_native.nii.gz"))
        sitk.WriteImage(out_img, out_path)

        msg = f"{f}\tdice_native={dsc:.4f}\tsaved={out_path}"
        print(msg)
        lines.append(msg)

    out_log = join(out_native_dir, "dice_native_log.txt")
    with open(out_log, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    print("Saved log:", out_log)

if __name__ == "__main__":
    main()
