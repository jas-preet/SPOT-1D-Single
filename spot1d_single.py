import torch
import numpy as np
from dataset.dataset_inference import ProteinDataset, text_collate_fn
from dataset.data_functions import pickle_load, read_list
from torch.utils.data import DataLoader
from main import classification, regression, write_csv_new
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--save_path', default='results/', type=str, help='save path')
parser.add_argument('--device', default='cpu', type=str,
                    help='"cuda:0", or "cpu" note wont run on other gpu then gpu0 due to limitations of jit trace')
parser.add_argument('--batch', default=1, type=int,
                    help='only two batch sizes are possible 1 or 10 and if you choose batch size 10 the number of proteins in the filelist should be a multiple of 10')

args = parser.parse_args()

path_list = read_list(args.file_list)
dataset = ProteinDataset(path_list)
fm12_loader = DataLoader(dataset, batch_size=args.batch, collate_fn=text_collate_fn, num_workers=1)
means = pickle_load("/home/jaspreet/jaspreet_data/stats/means_single.pkl")
stds = pickle_load("/home/jaspreet/jaspreet_data/stats/stds_single.pkl")
means = torch.tensor(means, dtype=torch.float32)
stds = torch.tensor(stds, dtype=torch.float32)

if args.device == "cpu" and args.batch == 1:
    model1_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_class.pth")
    model2_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_class.pth")
    model3_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_class.pth")

    model1_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_reg.pth")
    model2_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_reg.pth")
    model3_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_reg.pth")

elif args.device == "cpu" and args.batch == 10:
    model1_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_class_batch10.pth")
    model2_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_class_batch10.pth")
    model3_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_class_batch10.pth")

    model1_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_reg_batch10.pth")
    model2_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_reg_batch10.pth")
    model3_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_reg_batch10.pth")

elif args.device == "cuda:0" and args.batch == 1:
    model1_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_class_batch1_GPU.pth")
    model2_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_class_batch1_GPU.pth")
    model3_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_class_batch1_GPU.pth")

    model1_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_reg_batch1_GPU.pth")
    model2_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_reg_batch1_GPU.pth")
    model3_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_reg_batch1_GPU.pth")

elif args.device == "cuda:0" and args.batch == 10:
    model1_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_class_batch10_GPU.pth")
    model2_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_class_batch10_GPU.pth")
    model3_class = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_class_batch10_GPU.pth")

    model1_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model1_reg_batch10_GPU.pth")
    model2_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model2_reg_batch10_GPU.pth")
    model3_reg = torch.jit.load("/home/jaspreet/jaspreet_data/jits/model3_reg_batch10_GPU.pth")
else:
    print("please check the arguments passed and refer to help associated with the arguments")

class_out = classification(fm12_loader, model1_class, model2_class, model3_class, means, stds, args.device)
names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out

reg_out = regression(fm12_loader, model1_reg, model2_reg, model3_reg, means, stds, args.device)
psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list = reg_out
print(len(ss3_pred_list), len(psi_list))
write_csv_new(class_out, reg_out, args.save_path)

## conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
## conda install pandas
