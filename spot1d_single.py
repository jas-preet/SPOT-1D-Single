import torch
import numpy as np
from dataset.dataset_inference import ProteinDataset, text_collate_fn
from dataset.data_functions import pickle_load, read_list
from torch.utils.data import DataLoader
from main import classification, regression, write_csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--save_path', default='results/', type=str, help='save path')
parser.add_argument('--device', default='cpu', type=str,
                    help='"cuda:0", or "cpu" note wont run on other gpu then gpu0 due to limitations of jit trace')


args = parser.parse_args()

path_list = read_list(args.file_list)
dataset = ProteinDataset(path_list)
fm12_loader = DataLoader(dataset, batch_size=1, collate_fn=text_collate_fn, num_workers=4)
means = pickle_load("means_single.pkl")
stds = pickle_load("stds_single.pkl")
means = torch.tensor(means, dtype=torch.float32)
stds = torch.tensor(stds, dtype=torch.float32)

if args.device == "cpu":
    model1_class = torch.jit.load("jits/model1_class_cpu.pth")
    model2_class = torch.jit.load("jits/model2_class_cpu.pth")
    model3_class = torch.jit.load("jits/model3_class_cpu.pth")

    model1_reg = torch.jit.load("jits/model1_reg_cpu.pth")
    model2_reg = torch.jit.load("jits/model2_reg_cpu.pth")
    model3_reg = torch.jit.load("jits/model3_reg_cpu.pth")

elif args.device == "cuda:0":
    model1_class = torch.jit.load("jits/model1_class_gpu.pth")
    model2_class = torch.jit.load("jits/model2_class_gpu.pth")
    model3_class = torch.jit.load("jits/model3_class_gpu.pth")

    model1_reg = torch.jit.load("jits/model1_reg_gpu.pth")
    model2_reg = torch.jit.load("jits/model2_reg_gpu.pth")
    model3_reg = torch.jit.load("jits/model3_reg_gpu.pth")

else:
    print("please check the arguments passed and refer to help associated with the arguments")

class_out = classification(fm12_loader, model1_class, model2_class, model3_class, means, stds, args.device)
names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out

reg_out = regression(fm12_loader, model1_reg, model2_reg, model3_reg, means, stds, args.device)
psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list = reg_out
print(len(ss3_pred_list), len(psi_list))
write_csv(class_out, reg_out, args.save_path)

## conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
## conda install pandas
