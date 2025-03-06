import pandas as pd
import radiomics.featureextractor as featureextractor
import SimpleITK as sitk
import os
from tqdm import tqdm

df = pd.DataFrame()
settings = {'binWidth': 25,
            'resampledPixelSpacing': [0.8, 0.8, 1],
            'interpolator': sitk.sitkLinear,
            'normalize': True}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

ct_path = "./data/internal_test_data/largest_3d/ct/"
mask_path = "./data/internal_test_data/largest_3d/tumor/"
save_path = "./data/internal_test_data/radiomics/single_file/"
os.makedirs(save_path, exist_ok=True)

files = sorted(os.listdir(ct_path))

for file in tqdm(files):
    try:
        imageFile = os.path.join(ct_path, file)
        maskFile = os.path.join(mask_path, f"{file[:6]}_A.nii.gz")

        featureVector = extractor.execute(imageFile, maskFile)
        df_new = pd.DataFrame.from_dict(featureVector.values()).T
        df_new.columns = featureVector.keys()
        df_new.insert(0, 'imageFile', imageFile)
        df_new["imageFile"] = df_new["imageFile"].apply(lambda x: os.path.basename(x)[:8])
        df = pd.concat([df, df_new])
    except:
        print(file)
        continue

df.to_csv("./classification/data/summary_tumor.csv", index=False)
