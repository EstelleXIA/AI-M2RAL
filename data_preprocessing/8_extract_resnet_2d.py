import pandas as pd
from torchvision.models import resnet34
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import SimpleITK as sitk
import json
from tqdm import tqdm
from torchvision import transforms


class KidneyDataset(Dataset):
    def __init__(self):
        super(KidneyDataset, self).__init__()
        self.path = "./classification/data/largest_2d/ct/"
        self.files = os.listdir(self.path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_path = os.path.join(self.path, self.files[item])
        trans_norm = transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
        img = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img_path))[0])
        img = (torch.stack((img, img, img), dim=0).float() + 200) / 400
        img_norm = trans_norm(img)
        return self.files[item], img_norm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet34(pretrained=True)
extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
extractor.to(device)
extractor.eval()

input_data = KidneyDataset()
input_loader = DataLoader(dataset=input_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
file_list = []
output_list = []
for idx, (file_name, data) in tqdm(enumerate(input_loader)):
    with torch.no_grad():
        output = extractor(data.to(device))
        output_list.append(output.squeeze().cpu().numpy())
        file_list.append(file_name[0].replace(".nii.gz", ""))

final_feature = pd.DataFrame(output_list, index=file_list)
final_feature.to_csv("./classification/data/summary_cnn.csv")
