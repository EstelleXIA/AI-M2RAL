import torch
from monai.metrics import ROCAUCMetric
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)


def eval_once(loader, model, loss_function, class_num, device):
    model.eval()
    auc_metric_micro = ROCAUCMetric("micro")
    auc_metric_macro = ROCAUCMetric("macro")
    with torch.no_grad():
        post_pred = Compose([Activations(softmax=True)])
        post_label = Compose([AsDiscrete(to_onehot=class_num)])
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        loss_all = 0
        for batch_data in loader:
            order_data = batch_data["order"].to(device)
            texture_data = batch_data["texture"].to(device)
            wavelet_data = batch_data["wavelet"].to(device)
            shape_data = batch_data["shape"].to(device)
            lab_data = batch_data["lab"].to(device)
            cnn_data, labels = batch_data["cnn"].to(device), batch_data["label"].to(device)

            output = model(order_data, texture_data, wavelet_data, cnn_data, shape_data, lab_data)

            loss = loss_function(output, labels)
            loss_all += loss.item()
            y_pred = torch.cat([y_pred, output], dim=0)
            y = torch.cat([y, labels], dim=0)

        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        auc_metric_micro(y_pred_act, y_onehot)
        auc_result_micro = auc_metric_micro.aggregate()
        auc_metric_micro.reset()

        auc_metric_macro(y_pred_act, y_onehot)
        auc_result_macro = auc_metric_macro.aggregate()
        auc_metric_macro.reset()

    return acc_metric, auc_result_micro, auc_result_macro, loss_all


def plot_auc(y_true_bin, y_scores, save_file_path):
    rcParams['font.size'] = 12.5
    # rcParams['font.family'] = 'san-serif'
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print("AUC for Class 0:", roc_auc[0])
    print("AUC for Class 1:", roc_auc[1])
    print("AUC for Class 2:", roc_auc[2])
    print("AUC-micro:", roc_auc["micro"])
    print("AUC-macro:", roc_auc["macro"])

    plt.figure(figsize=(6.5, 6.5))
    plt.plot(fpr[0], tpr[0], label='Benign (area = %0.3f)' % roc_auc[0], color="deepskyblue")
    plt.plot(fpr[1], tpr[1], label='nccRCC (area = %0.3f)' % roc_auc[1], color="royalblue")
    plt.plot(fpr[2], tpr[2], label='ccRCC (area = %0.3f)' % roc_auc[2], color="cadetblue")
    plt.plot(fpr["micro"], tpr["micro"], label='micro (area = %0.3f)' % roc_auc["micro"], linestyle=':',
             linewidth=2, color="orange")
    plt.plot(fpr["macro"], tpr["macro"], label='macro (area = %0.3f)' % roc_auc["macro"], linestyle=':',
             linewidth=2, color="gold")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    plt.savefig(save_file_path, dpi=600)
    # plt.show()


def plot_cm(y_true, y_scores, save_file_path):
    rcParams['font.size'] = 12.5
    # rcParams['font.family'] = 'san-serif'
    cm = confusion_matrix(y_true, np.argmax(y_scores, axis=1))
    fig, ax = plt.subplots(figsize=(6, 6))
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)
    proportions = cm / row_sums
    proportions = np.nan_to_num(proportions, nan=0)  # Handle division by zero
    im = ax.imshow(proportions, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.75)
    classes = ['Benign', 'nccRCC', 'ccRCC']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label',
           ylabel='True label',
           title='')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{proportions[i, j] * 100:.1f}%\n(n={cm[i, j]:.0f})"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if proportions[i, j] > proportions.max() / 2 else "black")

    fig.tight_layout()
    plt.savefig(save_file_path, dpi=600)


def save_prediction_result(y_true, y_scores, name_list, save_file_path):
    results = pd.DataFrame({"class_0": y_scores[:, 0],
                            "class_1": y_scores[:, 1],
                            "class_2": y_scores[:, 2],
                            "pred": list(np.argmax(y_scores, axis=1)),
                            "gt": y_true.cpu()}, index=name_list)
    results.to_csv(save_file_path)


def plot_eval(loader, model, class_num, device, save_fig_path):
    model.eval()
    name_list = []
    auc_metric_micro = ROCAUCMetric("micro")
    auc_metric_macro = ROCAUCMetric("macro")
    with torch.no_grad():
        post_pred = Compose([Activations(softmax=True)])
        post_label = Compose([AsDiscrete(to_onehot=class_num)])
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)

        for batch_data in loader:
            order_data = batch_data["order"].to(device)
            texture_data = batch_data["texture"].to(device)
            wavelet_data = batch_data["wavelet"].to(device)
            shape_data = batch_data["shape"].to(device)
            lab_data = batch_data["lab"].to(device)
            cnn_data, labels = batch_data["cnn"].to(device), batch_data["label"].to(device)
            name_list.append(batch_data["name"][0])

            output = model(order_data, texture_data, wavelet_data, cnn_data, shape_data, lab_data)

            y_pred = torch.cat([y_pred, output], dim=0)
            y = torch.cat([y, labels], dim=0)

        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

        y_scores = np.stack([post_pred(i).cpu().numpy() for i in decollate_batch(y_pred)])

        plot_auc(y_true_bin=label_binarize(y.cpu(), classes=[0, 1, 2]),
                 y_scores=y_scores,
                 save_file_path=os.path.join(save_fig_path, "roc.png"))

        plot_cm(y_true=y.cpu(), y_scores=y_scores,
                save_file_path=os.path.join(save_fig_path, "cm.png"))

        save_prediction_result(y_true=y.cpu(), y_scores=y_scores, name_list=name_list,
                               save_file_path=os.path.join(save_fig_path, "pred.csv"))

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        auc_metric_micro(y_pred_act, y_onehot)
        auc_result_micro = auc_metric_micro.aggregate()
        auc_metric_micro.reset()

        auc_metric_macro(y_pred_act, y_onehot)
        auc_result_macro = auc_metric_macro.aggregate()
        auc_metric_macro.reset()

    return acc_metric, auc_result_micro, auc_result_macro


