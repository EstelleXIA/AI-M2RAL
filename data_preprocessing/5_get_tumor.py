import os
import sys
import SimpleITK as sitk
import numpy as np

mask_path = "./classification/data/internal_test_data/largest_3d/mask/"
save_path = "./classification/data/internal_test_data/largest_3d/tumor/"
os.makedirs(save_path, exist_ok=True)

files = os.listdir(mask_path)
for file in files:
    try:
        img = sitk.ReadImage(os.path.join(mask_path, file))
        img_np = sitk.GetArrayFromImage(img)
        tumor_np = (img_np == 2).astype(np.uint8)
        tumor = sitk.GetImageFromArray(tumor_np)
        tumor.SetOrigin(img.GetOrigin())
        tumor.SetDirection(img.GetDirection())
        tumor.SetSpacing(img.GetSpacing())
        sitk.WriteImage(tumor, os.path.join(save_path, file))

    except:
        continue
