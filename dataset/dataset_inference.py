import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset.data_functions import one_hot, read_fasta_file


class ProteinDataset(Dataset):
    def __init__(self, path_fasta):
        self.path = path_fasta

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        protein = self.path[idx]
        seq = read_fasta_file(protein)
        seq_hot = one_hot(seq)
        protein_len = len(seq)
        name = os.path.splitext(os.path.basename(protein))[0]

        return seq_hot, protein_len, seq, name


def text_collate_fn(data):
    """
    collate function for data read from text file
    """

    # sort data by caption length
    data.sort(key=lambda x: x[1], reverse=True)
    features, protein_len, seq, name = zip(*data)

    features = [torch.FloatTensor(x) for x in features]

    # Pad features
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    return padded_features, protein_len, seq, name  ### also return feats_lengths and label_lengths if using packpadd
