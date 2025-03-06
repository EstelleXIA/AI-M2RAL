import numpy as np
import os
import SimpleITK as sitk
from typing import Tuple, Optional
from tqdm import tqdm
import shutil


def resize_img(dcm: sitk.Image, spacing: Optional[Tuple[float, float, float]] = (0.8, 0.8, 1.0),
               size: Optional[Tuple[int, int, int]] = (128, 128, 192)) -> sitk.Image:
    new_spacing = (np.array(dcm.GetSize()) / np.array(size) * np.array(spacing)).tolist()
    resized = sitk.Resample(image1=dcm, size=size,
                            transform=sitk.Transform(),
                            interpolator=sitk.sitkLinear,
                            outputOrigin=dcm.GetOrigin(),
                            outputSpacing=new_spacing,
                            outputDirection=dcm.GetDirection(),
                            defaultPixelValue=-200,
                            outputPixelType=dcm.GetPixelID())
    return resized


def resize_mask(dcm: sitk.Image, spacing: Optional[Tuple[float, float, float]] = (0.8, 0.8, 1.0),
                size: Optional[Tuple[int, int, int]] = (128, 128, 192)) -> sitk.Image:
    new_spacing = (np.array(dcm.GetSize()) / np.array(size) * np.array(spacing)).tolist()
    resized = sitk.Resample(image1=dcm, size=size,
                            transform=sitk.Transform(),
                            interpolator=sitk.sitkNearestNeighbor,
                            outputOrigin=dcm.GetOrigin(),
                            outputSpacing=new_spacing,
                            outputDirection=dcm.GetDirection(),
                            defaultPixelValue=0,
                            outputPixelType=dcm.GetPixelID())
    return resized


def get_img_from_arr(arr, ref: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(ref.GetOrigin())
    img.SetSpacing(ref.GetSpacing())
    img.SetDirection(ref.GetDirection())
    return img


ct_path = "./classification/data/internal_test_data/roi/roi_ct/"
mask_path = "./classification/data/internal_test_data/roi/roi_mask/"
save_path = "./classification/data/internal_test_data/pad/"

files = sorted(os.listdir(ct_path))
z_shape = []
for file in tqdm(files):
    try:
        ct_img = sitk.ReadImage(os.path.join(ct_path, file))
        ct_array = sitk.GetArrayFromImage(ct_img)
        z_shape.append(ct_array.shape[0])
        mask_img = sitk.ReadImage(os.path.join(mask_path, f"{file[:6]}_A-{file[9]}.nii.gz"))
        mask_array = sitk.GetArrayFromImage(mask_img)
        z, x, y = ct_array.shape
        w = max(128, x, y)
        z_min, z_max = (192 - z) // 2, 192 - z - (192 - z) // 2
        x_min, x_max = (w - x) // 2, w - x - (w - x) // 2
        y_min, y_max = (w - y) // 2, w - y - (w - y) // 2
        if z_min >= 0:
            ct_pad = np.pad(ct_array, ((z_min, z_max), (x_min, x_max), (y_min, y_max)), mode="constant", constant_values=-200)
            mask_pad = np.pad(mask_array, ((z_min, z_max), (x_min, x_max), (y_min, y_max)), mode="constant")
        else:
            ct_array = ct_array[-z_min: (-z_min + 192), :, :]
            ct_pad = np.pad(ct_array, ((0, 0), (x_min, x_max), (y_min, y_max)), mode="constant", constant_values=-200)
            mask_array = mask_array[-z_min: (-z_min + 192), :, :]
            mask_pad = np.pad(mask_array, ((0, 0), (x_min, x_max), (y_min, y_max)), mode="constant")
        # ct_pad = np.pad(ct_array, ((0, 0), (x_min, x_max), (y_min, y_max)), mode="constant")
        # mask_pad = np.pad(mask_array, ((0, 0), (x_min, x_max), (y_min, y_max)), mode="constant")
        ct_pad_img = get_img_from_arr(ct_pad, ct_img)
        mask_pad_img = get_img_from_arr(mask_pad, mask_img)
        os.makedirs(os.path.join(save_path, "ct"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "mask"), exist_ok=True)
        if w > 128:
            ct_resized = resize_img(ct_pad_img)
            mask_resized = resize_mask(mask_pad_img)
            sitk.WriteImage(ct_resized, os.path.join(save_path, "ct", file))
            sitk.WriteImage(mask_resized, os.path.join(save_path, "mask", f"{file[:6]}_A-{file[9]}.nii.gz"))
        if not os.path.exists(os.path.join(save_path, "ct", file)):
            sitk.WriteImage(ct_pad_img, os.path.join(save_path, "ct", file))
            sitk.WriteImage(mask_pad_img, os.path.join(save_path, "mask", f"{file[:6]}_A-{file[9]}.nii.gz"))
    except:
        continue
