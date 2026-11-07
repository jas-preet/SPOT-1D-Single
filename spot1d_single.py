import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.data_functions import pickle_load, read_fasta_file, read_list
from dataset.dataset_inference import ProteinDataset, text_collate_fn
from main import classification, regression, write_csv

REPO_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict protein 1D structural properties from single sequences.")
    parser.add_argument('--file_list', required=True, type=str,
                        help='text file listing one FASTA file path per line')
    parser.add_argument('--save_path', default='results/', type=str,
                        help='output directory, created if needed (default: %(default)s)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'],
                        help='the GPU checkpoints were traced on cuda:0 and only run there '
                             '(default: %(default)s)')
    parser.add_argument('--jits_dir', default=REPO_DIR / 'jits', type=Path,
                        help='directory holding the model checkpoints (default: jits/ next to this script)')
    return parser.parse_args()


def checkpoint_paths(jits_dir, kind, device):
    suffix = 'cpu' if device == 'cpu' else 'gpu'
    return [Path(jits_dir) / f'model{number}_{kind}_{suffix}.pth' for number in (1, 2, 3)]


def check_inputs(args):
    """Report anything that would stop the run before spending time on the models."""
    if args.device != 'cpu' and not torch.cuda.is_available():
        sys.exit(f'error: --device {args.device} was requested but CUDA is not available')

    missing = [str(path) for kind in ('class', 'reg')
               for path in checkpoint_paths(args.jits_dir, kind, args.device) if not path.is_file()]
    if missing:
        sys.exit('error: missing model checkpoints (see "Installation" in the README):\n  '
                 + '\n  '.join(missing))

    try:
        path_list = read_list(args.file_list)
        for path in path_list:
            read_fasta_file(path)
        os.makedirs(args.save_path, exist_ok=True)
    except (OSError, ValueError) as error:
        sys.exit(f'error: {error}')
    if not path_list:
        sys.exit(f'error: {args.file_list} does not list any FASTA files')
    return path_list


def load_ensemble(jits_dir, kind, device):
    return [torch.jit.load(str(path)) for path in checkpoint_paths(jits_dir, kind, device)]


def main():
    args = parse_args()
    path_list = check_inputs(args)

    # batch_size must stay 1: the traced checkpoints do not support larger batches
    loader = DataLoader(ProteinDataset(path_list), batch_size=1, collate_fn=text_collate_fn)
    means = torch.tensor(pickle_load(REPO_DIR / 'means_single.pkl'), dtype=torch.float32)
    stds = torch.tensor(pickle_load(REPO_DIR / 'stds_single.pkl'), dtype=torch.float32)

    # Each ensemble is released before the next is loaded to keep peak memory down
    class_out = classification(loader, *load_ensemble(args.jits_dir, 'class', args.device),
                               means, stds, args.device)
    reg_out = regression(loader, *load_ensemble(args.jits_dir, 'reg', args.device),
                         means, stds, args.device)
    write_csv(class_out, reg_out, args.save_path)


if __name__ == '__main__':
    main()
