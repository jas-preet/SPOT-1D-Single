import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset.data_functions import SS3_CLASSES, SS8_CLASSES, get_angle_degree, get_unnorm_asa

CSV_COLUMNS = ["AA", "SS3", "SS8", "ASA", "HseU", "HseD", "CN", "Psi", "Phi", "Theta", "Tau",
               "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G", "P8I", "P8E", "P8B"]

# Output channels of the regression models. Each angle is a (sin, cos) pair.
PSI, PHI, THETA, TAU = 0, 2, 4, 6
HSE_UP, HSE_DOWN, CONTACT_NUMBER, REL_ASA = 8, 9, 10, 11

# The models predict these quantities scaled to [0, 1].
HSE_UP_SCALE, HSE_DOWN_SCALE, CONTACT_NUMBER_SCALE = 50, 65, 85


def prepare_models(models, device):
    models = [model.to(device) for model in models]
    for model in models:
        model.eval()
    return models


def predict(models, feats, length):
    """The second model of each ensemble was exported without the length argument."""
    model1, model2, model3 = models
    return model1(feats, length), model2(feats), model3(feats, length)


def prepare_batch(data, mean, std, device):
    feats, length, seq, name = data
    length = torch.tensor(length).to(device)
    feats = (feats - mean) / std
    feats = feats.to(device, dtype=torch.float)
    return feats, length, seq, name


def decode_classes(probabilities, classes):
    indices = np.argmax(probabilities, axis=1)
    return np.array([classes[index] for index in indices])[:, None]


@torch.no_grad()
def classification(data_loader, model1, model2, model3, mean, std, device):
    models = prepare_models((model1, model2, model3), device)
    ss3_pred_list = []
    ss8_pred_list = []
    ss3_prob_list = []
    ss8_prob_list = []
    names_list = []
    seq_list = []
    for data in data_loader:
        feats, length, seqs, names = prepare_batch(data, mean, std, device)

        pred1, pred2, pred3 = predict(models, feats, length)
        pred = (pred1 + pred2 + pred3) / 3

        ss3_pred = F.softmax(pred[:, :, 0:3], dim=2).cpu().numpy()
        ss8_pred = F.softmax(pred[:, :, 3:], dim=2).cpu().numpy()
        for i, prot_len in enumerate(length.tolist()):
            ss3_prob = ss3_pred[i, :prot_len, :]
            ss8_prob = ss8_pred[i, :prot_len, :]
            ss3_pred_list.append(decode_classes(ss3_prob, SS3_CLASSES))
            ss8_pred_list.append(decode_classes(ss8_prob, SS8_CLASSES))
            ss3_prob_list.append(ss3_prob)
            ss8_prob_list.append(ss8_prob)
            names_list.append(names[i])
            seq_list.append(np.array(list(seqs[i]))[:, None])

    return names_list, seq_list, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list


def median_angle(preds, channel):
    """Ensemble an angle by taking the median of its sin and cos, then convert to degrees."""
    stacked = torch.stack([pred[:, :, channel:channel + 2] for pred in preds], dim=3)
    median, _ = torch.median(stacked, dim=-1)
    return get_angle_degree(median.cpu().numpy())


def mean_channel(preds, channel):
    pred1, pred2, pred3 = preds
    return (pred1[:, :, channel] + pred2[:, :, channel] + pred3[:, :, channel]) / 3


def append_trimmed(target, values, length):
    """Remove the padding from each protein in the batch and add it to the target list."""
    for i, prot_len in enumerate(length.tolist()):
        target.append(values[i, :prot_len, None])


@torch.no_grad()
def regression(data_loader, model1, model2, model3, mean, std, device):
    models = prepare_models((model1, model2, model3), device)

    psi_list = []
    phi_list = []
    theta_list = []
    tau_list = []
    hseu_list = []
    hsed_list = []
    cn_list = []
    asa_list = []
    for data in data_loader:
        feats, length, seqs, _ = prepare_batch(data, mean, std, device)
        preds = predict(models, feats, length)

        append_trimmed(psi_list, median_angle(preds, PSI), length)
        append_trimmed(phi_list, median_angle(preds, PHI), length)
        append_trimmed(theta_list, median_angle(preds, THETA), length)
        append_trimmed(tau_list, median_angle(preds, TAU), length)

        append_trimmed(hseu_list, (mean_channel(preds, HSE_UP) * HSE_UP_SCALE).cpu().numpy(), length)
        append_trimmed(hsed_list, (mean_channel(preds, HSE_DOWN) * HSE_DOWN_SCALE).cpu().numpy(), length)
        append_trimmed(cn_list, (mean_channel(preds, CONTACT_NUMBER) * CONTACT_NUMBER_SCALE).cpu().numpy(), length)
        append_trimmed(asa_list, get_unnorm_asa(mean_channel(preds, REL_ASA), seqs), length)

    return psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list


def write_csv(class_out, reg_out, save_dir):
    names, seqs, ss3_preds, ss8_preds, ss3_probs, ss8_probs = class_out
    psis, phis, thetas, taus, hseus, hseds, cns, asas = reg_out

    for name, *columns in zip(names, seqs, ss3_preds, ss8_preds, asas, hseus, hseds, cns,
                              psis, phis, thetas, taus, ss3_probs, ss8_probs):
        data = np.concatenate(columns, axis=1)
        save_path = os.path.join(save_dir, name + ".csv")
        pd.DataFrame(data).to_csv(save_path, header=CSV_COLUMNS)
    print(f'Results saved to {save_dir}')


if __name__ == '__main__':
    print("Please run spot1d_single.py instead")
