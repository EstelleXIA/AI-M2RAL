import torch
import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from itertools import chain

ss = StandardScaler()
ss_shape = StandardScaler()
ss_lab = StandardScaler()


#  status = training & validation & external
class RadDataset(Dataset):
    def __init__(self, rad_data, shape_data, status):
        self.data = rad_data

        all_columns = list(pd.read_csv("./classification/data/summary_tumor.csv", index_col="imageFile").columns)
        self.order_cols = sorted(list(filter(lambda x: "original_firstorder" in x, all_columns)))
        self.texture_cols = sorted(list(filter(lambda x: ("original_glcm" in x) or ("original_gldm" in x)
                                                         or ("original_glrlm" in x) or ("original_glszm" in x)
                                                         or ("original_ngtdm" in x), all_columns)))
        self.wavelet_cols = sorted(list(filter(lambda x: "wavelet" in x, all_columns)))

        self.shape = shape_data
        with open(os.path.join("./classification/code/ct_model/kidney_split_final.json"), "r") as f:
            split = json.load(f)

        self.info = pd.DataFrame(split[status])
        self.patients = self.info.iloc[:, 0].tolist()
        self.info.index = self.patients
        train_patients = pd.DataFrame(split["training"]).iloc[:, 0].tolist()
        train_indexes = list(chain(*[[f"{p}_N", f"{p}_A", f"{p}_V", f"{p}_D"] for p in train_patients]))
        train_data = self.data.loc[train_indexes]
        train_shape = self.shape.loc[train_indexes]
        _ = ss.fit_transform(train_data)
        _ = ss_shape.fit_transform(train_shape)
        self.cnn_data = pd.read_csv("./classification/data/summary_cnn.csv", index_col=0)

        self.lab_data = pd.read_csv("./classification/code/preprocess_clinical/"
                                    "final_clinical_patients_mice.csv", index_col=0)
        train_lab = self.lab_data.loc[train_patients]
        _ = ss_lab.fit_transform(train_lab)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        target_patient = self.patients[index]
        target_files = [f"{target_patient}_N", f"{target_patient}_A", f"{target_patient}_V", f"{target_patient}_D"]
        target_rad = pd.DataFrame(ss.transform(self.data.loc[target_files]), columns=self.data.columns)
        target_shape = ss_shape.transform(self.shape.loc[[f"{target_patient}_A"]])
        target_order = target_rad.loc[:, self.order_cols].values
        target_texture = target_rad.loc[:, self.texture_cols].values
        target_wavelet = target_rad.loc[:, self.wavelet_cols].values

        target_label = self.info.loc[target_patient, "label"]
        target_cnn = self.cnn_data.loc[target_files].values

        target_lab = ss_lab.transform(self.lab_data.loc[[target_patient]])

        return {"order": torch.from_numpy(target_order.astype(np.float32)),
                "texture": torch.from_numpy(target_texture.astype(np.float32)),
                "wavelet": torch.from_numpy(target_wavelet.astype(np.float32)),
                "shape": torch.from_numpy(target_shape.astype(np.float32))[0],
                "cnn": torch.from_numpy(target_cnn.astype(np.float32)),
                "lab": torch.from_numpy(target_lab.astype(np.float32))[0],
                "label": target_label,
                "name": target_patient}
