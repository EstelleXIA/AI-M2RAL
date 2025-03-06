import os
import numpy as np
import SimpleITK as sitk
import cc3d
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

ct_path = "./classification/data/internal_test_data/full_res/all_phases_registered/"
mask_path = "./classification/data/internal_test_data/full_res/mask_arterial/"
files = sorted(os.listdir(ct_path))
save_base = ["./classification/data/internal_test_data/roi/roi_ct/",
save_base = ["./classification/data/internal_test_data/roi/roi_mask/"]

os.makedirs(save_base[0], exist_ok=True)
os.makedirs(save_base[1], exist_ok=True)

roi_shape = []
for file in tqdm(files):
    try:
        ct = sitk.ReadImage(os.path.join(ct_path, file))
        ct_array = sitk.GetArrayFromImage(ct)
        mask = sitk.ReadImage(os.path.join(mask_path, f"{file[:6]}_A.nii.gz"))
        mask_array_raw = sitk.GetArrayFromImage(mask)
        mask_array = deepcopy(mask_array_raw)
        mask_array[mask_array > 0] = 1
        mask_cc3d, number = cc3d.connected_components(mask_array, connectivity=26, return_N=True)
        stats = cc3d.statistics(mask_cc3d)
        candidates_idx = [x + 1 for x in range(stats["voxel_counts"][1:].shape[0])]
        bboxes = [stats["bounding_boxes"][x] for x in candidates_idx]
        bbox_candidates = [[bboxes[i][0].start, bboxes[i][0].stop,
                            bboxes[i][1].start, bboxes[i][1].stop,
                            bboxes[i][2].start, bboxes[i][2].stop, ] for i in range(len(bboxes))]
        assert len(bbox_candidates) > 0 and len(bbox_candidates) <= 2
        for ids, box in enumerate(bbox_candidates):
            roi_ct = ct_array[max(0, int(box[0] - 5)):min(ct_array.shape[0], int(box[1] + 5)),
                              max(0, int(box[2] - 10)):min(512, int(box[3] + 10)),
                              max(0, int(box[4] - 10)):min(512, int(box[5] + 10))]
            roi_mask = mask_array_raw[max(0, int(box[0] - 5)):min(ct_array.shape[0], int(box[1] + 5)),
                                    max(0, int(box[2] - 10)):min(512, int(box[3] + 10)),
                                    max(0, int(box[4] - 10)):min(512, int(box[5] + 10))]
            for save_path_id, roi_selected in enumerate((roi_ct, roi_mask)):
                roi = sitk.GetImageFromArray(roi_selected)
                roi.SetOrigin(ct.GetOrigin())
                roi.SetDirection(ct.GetDirection())
                roi.SetSpacing(ct.GetSpacing())
                sitk.WriteImage(roi, os.path.join(save_base[save_path_id],
                                                  file.replace("_0000", "").split(".")[0] + "-" + str(ids) + ".nii.gz"))
    except:
        continue
