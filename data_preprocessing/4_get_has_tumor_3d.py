import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import shutil

img_path = "./classification/data/internal_test_data/pad/ct/"
mask_path = "./classification/data/internal_test_data/pad/mask/"
save_path = "./classification/data/internal_test_data/largest_3d/"

os.makedirs(save_path, exist_ok=True)

files = os.listdir(mask_path)
patients = sorted(list(set([x[:8] for x in files])))

for patient in tqdm(patients):
    try:
        patient_list = list(filter(lambda x: x.startswith(patient), files))
        assert (len(patient_list) == 1) or (len(patient_list) == 2)
        idx = 999
        if len(patient_list) == 2:
            mask_1 = sitk.ReadImage(os.path.join(mask_path, f"{patient}-0.nii.gz"))
            mask_2 = sitk.ReadImage(os.path.join(mask_path, f"{patient}-1.nii.gz"))
            mask_1_np = sitk.GetArrayFromImage(mask_1)
            mask_2_np = sitk.GetArrayFromImage(mask_2)
            mask_num_1 = np.unique(mask_1_np)
            mask_num_2 = np.unique(mask_2_np)
            mask_1_np = (mask_1_np == 2).astype(int)
            mask_2_np = (mask_2_np == 2).astype(int)
            if len(mask_num_1) == 3 and len(mask_num_2) == 2:
                idx = 0
            elif len(mask_num_1) == 2 and len(mask_num_2) == 3:
                idx = 1
            else:
                idx = 999
        if len(patient_list) == 1:
            idx = 0

        assert idx != 999

        os.makedirs(os.path.join(save_path, "ct"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "mask"), exist_ok=True)
        shutil.copy(os.path.join(img_path, f"{patient}-{idx}.nii.gz"),
                    os.path.join(save_path, "ct"))
        shutil.copy(os.path.join(img_path, f"{patient.replace('A', 'N')}-{idx}.nii.gz"),
                    os.path.join(save_path, "ct"))
        shutil.copy(os.path.join(img_path, f"{patient.replace('A', 'V')}-{idx}.nii.gz"),
                    os.path.join(save_path, "ct"))
        shutil.copy(os.path.join(img_path, f"{patient.replace('A', 'D')}-{idx}.nii.gz"),
                    os.path.join(save_path, "ct"))
        shutil.copy(os.path.join(mask_path, f"{patient}-{idx}.nii.gz"),
                    os.path.join(save_path, "mask"))
    except:
        continue
