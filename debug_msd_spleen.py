import os
from os.path import join
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from skimage import measure

from medsam2_infer_3D_CT import build_sam2_video_predictor_npz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(BASE_DIR, "data", "Task09_Spleen", "imagesTr_5", "spleen_2.nii.gz")
gt_path  = os.path.join(BASE_DIR, "data", "Task09_Spleen", "labelsTr_5", "spleen_2.nii.gz")

def window_ct_hu(vol, hu_min=-125, hu_max=275):
    print("Original HU range:", vol.min(), "to", vol.max())
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-8)
    return (vol * 255.0).astype(np.uint8)


def pick_key_slice(gt3d):
    areas = gt3d.reshape(gt3d.shape[0], -1).sum(axis=1)
    print("Foreground pixels per slice:")
    print(areas)
    key = int(np.argmax(areas))
    print("Selected key slice:", key)
    return key


def bbox2d_from_mask(mask2d):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        print("No foreground found in this slice.")
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    print("Bounding box:", [x1, y1, x2, y2])
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def resize_to_model_input(vol):
    D, H, W = vol.shape
    print("Original shape:", vol.shape)
    out = np.zeros((D, 3, 512, 512), dtype=np.float32)
    for i in range(D):
        img = Image.fromarray(vol[i]).convert("RGB").resize((512, 512))
        out[i] = np.array(img).transpose(2, 0, 1)
    print("Resized shape:", out.shape)
    return out


def main():
    img_path = "data/Task09_Spleen/imagesTr_5/spleen_2.nii.gz"
    gt_path  = "data/Task09_Spleen/labelsTr_5/spleen_2.nii.gz"

    print("\nLoading CT...")
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)

    print("Loading GT...")
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    print("CT shape:", img.shape)
    print("GT shape:", gt.shape)

    img_u8 = window_ct_hu(img)

    key_idx = pick_key_slice(gt)
    bbox = bbox2d_from_mask(gt[key_idx])

    vol_resized = resize_to_model_input(img_u8) / 255.0
    vol_t = torch.from_numpy(vol_resized).cuda()

    predictor = build_sam2_video_predictor_npz(
        "configs/sam2.1_hiera_t512.yaml",
        "checkpoints/MedSAM2_latest.pt"
    )

    print("\nRunning model inference...")
    seg512 = np.zeros((vol_t.shape[0], 512, 512), dtype=np.uint8)

    with torch.inference_mode():
        state = predictor.init_state(vol_t, 512, 512)
        _, _, logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=key_idx,
            obj_id=1,
            box=bbox,
        )

        seg512[key_idx] = (logits[0] > 0).cpu().numpy()[0]

        for fidx, _, logits in predictor.propagate_in_video(state):
            seg512[fidx] = (logits[0] > 0).cpu().numpy()[0]

    print("\nPrediction foreground per slice:")
    pred_areas = seg512.reshape(seg512.shape[0], -1).sum(axis=1)
    print(pred_areas)

    print("\nGT foreground per slice:")
    gt_areas = gt.reshape(gt.shape[0], -1).sum(axis=1)
    print(gt_areas)

    print("\nDone. Now compare GT vs prediction slices.")

    gt_areas = gt.reshape(gt.shape[0], -1).sum(axis=1)
    nonzero = np.where(gt_areas > 0)[0]
    print("GT spleen exists from slice", nonzero.min(), "to", nonzero.max())

    pred_areas = seg512.reshape(seg512.shape[0], -1).sum(axis=1)
    nonzero_p = np.where(pred_areas > 0)[0]
    print("Pred exists from slice", nonzero_p.min(), "to", nonzero_p.max())


if __name__ == "__main__":
    main()
