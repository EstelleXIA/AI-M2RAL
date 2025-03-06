import os
import SimpleITK as sitk
from tqdm import tqdm

path = "./classification/data/internal_test_data/largest_3d/"
save_path = "./classification/data/internal_test_data/largest_2d/"

ct_path = os.path.join(path, "ct")
mask_path = os.path.join(path, "mask")
os.makedirs(os.path.join(save_path, "ct"), exist_ok=True)
os.makedirs(os.path.join(save_path, "mask"), exist_ok=True)

files = sorted(os.listdir(os.path.join(path, "tumor")))
for file in tqdm(files):
    mask = sitk.ReadImage(os.path.join(mask_path, file))
    ct_a = sitk.ReadImage(os.path.join(ct_path, file))
    ct_n = sitk.ReadImage(os.path.join(ct_path, file.replace("A", "N")))
    ct_v = sitk.ReadImage(os.path.join(ct_path, file.replace("A", "V")))
    ct_d = sitk.ReadImage(os.path.join(ct_path, file.replace("A", "D")))
    mask_np = sitk.GetArrayFromImage(mask)
    mask_z = (mask_np == 2).sum(axis=(1, 2))
    max_slice = list(mask_z).index(max(list(mask_z)))
    mask_save = save_img(max_slice, mask)
    sitk.WriteImage(mask_save, os.path.join(save_path, "mask", file))
    img_save_a = save_img(max_slice, ct_a)
    sitk.WriteImage(img_save_a, os.path.join(save_path, "ct", file))
    img_save_n = save_img(max_slice, ct_n)
    sitk.WriteImage(img_save_n, os.path.join(save_path, "ct", file.replace("A", "N")))
    img_save_v = save_img(max_slice, ct_v)
    sitk.WriteImage(img_save_v, os.path.join(save_path, "ct", file.replace("A", "V")))
    img_save_d = save_img(max_slice, ct_d)
    sitk.WriteImage(img_save_d, os.path.join(save_path, "ct", file.replace("A", "D")))

