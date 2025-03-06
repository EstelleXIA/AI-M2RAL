import sys
sys.path.append("./classification/code/ct_model/")
import torch
import os
from model import CNNOmicsLabRNNConcat, CNNOmicsRNNConcat, CNNConcat, CNNOmicsLabRNNConcatV2
from dataset import RadDataset
import pandas as pd
import numpy as np
import torch.nn as nn
from train_utils import eval_once
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.metrics import ROCAUCMetric
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import argparse

parser = argparse.ArgumentParser(description="Arguments for model training.")
parser.add_argument("--model_type", type=str, default="M2RAL", choices=["M2RAL", "M2RAD", "Dai"])
args = parser.parse_args()

model_type = args.model_type
batch_size = 32
max_epochs = 500
learning_rate = 5e-5
val_interval = 1
best_metric = -1
best_metric_test = -1
best_metric_epoch = -1
best_metric_epoch_test = -1
class_num = 3

save_ckpt_path = f"./classification/code/ct_model/ckpts/{model_type}/"
os.makedirs(save_ckpt_path, exist_ok=True)

radiomics_data = pd.read_csv("./classification/data/summary_tumor.csv", index_col="imageFile")
cols_fix = list(filter(lambda x: "original_shape" in x, radiomics_data.columns.tolist()))
cols_bin = list(filter(lambda x: x.startswith("diagnostics"), radiomics_data.columns.tolist()))
cols = sorted(list(set(radiomics_data.columns.tolist()) - set(cols_fix) - set(cols_bin)))
radiomics_prep = radiomics_data.loc[:, cols]
radiomics_shape = radiomics_data.loc[:, cols_fix]

train_dataset = RadDataset(radiomics_prep, radiomics_shape, status="training")
val_dataset = RadDataset(radiomics_prep, radiomics_shape, status="validation")
int_dataset = RadDataset(radiomics_prep, radiomics_shape, status="internal")
ext_dataset = RadDataset(radiomics_prep, radiomics_shape, status="external")

print(f"Train on {len(train_dataset)} patients.")
print(f"Validation on {len(val_dataset)} patients.")
print(f"Internal Test on {len(int_dataset)} patients.")
print(f"External Test on {len(ext_dataset)} patients.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
int_loader = DataLoader(int_dataset, batch_size=1, shuffle=False)
ext_loader = DataLoader(ext_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == "M2RAL":
    model = CNNOmicsLabRNNConcatV2(num_classes=class_num).to(device)
elif model_type == "M2RAD":
    model = CNNOmicsRNNConcat(num_classes=class_num).to(device)
elif model_type == "Dai":
    model = CNNConcat(num_classes=class_num).to(device)
else:
    raise NotImplementedError

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
auc_metric = ROCAUCMetric()
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=class_num)])

epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        order_data = batch_data["order"].to(device)
        texture_data = batch_data["texture"].to(device)
        wavelet_data = batch_data["wavelet"].to(device)
        shape_data = batch_data["shape"].to(device)
        lab_data = batch_data["lab"].to(device)
        cnn_data, labels = batch_data["cnn"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(order_data, texture_data, wavelet_data, cnn_data, shape_data, lab_data)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_loader.batch_size

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"[Training] loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        acc_val, micro_val, macro_val, loss_val = eval_once(loader=val_loader, model=model, loss_function=loss_function,
                                                            class_num=class_num, device=device)
        acc_int, micro_int, macro_int, loss_int = eval_once(loader=int_loader, model=model, loss_function=loss_function,
                                                            class_num=class_num, device=device)
        acc_ext, micro_ext, macro_ext, loss_ext = eval_once(loader=ext_loader, model=model, loss_function=loss_function,
                                                            class_num=class_num, device=device)

        ckpt_name = f"epoch_{epoch}_val_{acc_val:.4f}_{macro_val:.4f}_int_{acc_int:.4f}_{macro_int:.4f}" \
                    f"_ext_{acc_ext:.4f}_{macro_ext:.4f}.pth"
        torch.save(model.state_dict(), os.path.join(save_ckpt_path, ckpt_name))

        print(f"[Validation] loss: {loss_val / len(val_loader):.4f} accuracy: {acc_val:.4f}"
              f" micro-auc: {micro_val:.4f} macro-auc: {macro_val:.4f}")

        print(f"[Internal] loss: {loss_int / len(int_loader):.4f} accuracy: {acc_int:.4f}"
              f" micro-auc: {micro_int:.4f} macro-auc: {macro_int:.4f}")

        print(f"[External] loss: {loss_ext / len(ext_loader):.4f} accuracy: {acc_ext:.4f}"
              f" micro-auc: {micro_ext:.4f} macro-auc: {macro_ext:.4f}")
