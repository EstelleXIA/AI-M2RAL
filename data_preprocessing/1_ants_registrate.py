import ants
import os
import shutil
from tqdm import tqdm

path = "./classification/data/internal_test_data/full_res/all_phases/"
save_path = "./classification/data/internal_test_data/full_res/all_phases_registered/"
os.makedirs(save_path, exist_ok=True)
files = os.listdir(path)
arterial_files = sorted(list(filter(lambda x: x.split("_")[1] == "A", files)))

for phase in ["N", "V", "D"]:
    for file in tqdm(arterial_files):
        f = os.path.join(path, file)
        m = os.path.join(path, file.replace("A", phase))

        f = ants.image_read(f)
        m = ants.image_read(m)
        m.set_origin(f.origin)
        mytx = ants.registration(fixed=f, moving=m, type_of_transform='SyN', verbose=False)
        m_wrap200 = ants.apply_transforms(fixed=f, moving=m,
                                          transformlist=mytx['fwdtransforms'],
                                          defaultvalue=-200)

        transformed_img = os.path.join(save_path, file.replace("A", phase))
        m_wrap200.to_file(transformed_img)
        if not os.path.exists(os.path.join(save_path, file)):
            shutil.copy(os.path.join(path, file), os.path.join(save_path, file))
