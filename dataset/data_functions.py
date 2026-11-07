import pickle

import numpy as np

SS3_CLASSES = 'CEH'
SS8_CLASSES = 'CSTHGIEB'

# Maximum accessible surface area per residue, used to convert relative ASA to absolute.
# Anything else (X, B, Z, U, O, J, ...) falls back to UNKNOWN_RESIDUE_ASA.
MAX_ASA = dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
     185, 160, 145, 180, 225, 115, 140, 155, 255, 230),
))
UNKNOWN_RESIDUE_ASA = 1


def read_list(file_name):
    """
    read a text file to get the list of elements, ignoring blank lines
    :param file_name: complete path to a file (string)
    :return: list of elements in the text file
    """
    with open(file_name, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def read_fasta_file(fname):
    """
    reads the single sequence from a fasta file
    :param fname: filename (string)
    :return: upper-case protein sequence, without whitespace or a trailing stop "*" (string)
    :raises ValueError: if the file does not hold exactly one sequence
    """
    with open(fname, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or not lines[0].startswith('>'):
        raise ValueError(f"{fname}: expected a FASTA header line starting with '>'")
    if any(line.startswith('>') for line in lines[1:]):
        raise ValueError(f"{fname}: expected one sequence per file, found several")

    sequence = ''.join(''.join(lines[1:]).split()).upper().rstrip('*')
    if not sequence:
        raise ValueError(f"{fname}: no sequence after the header")
    return sequence


def one_hot(seq):
    """
    converts a sequence to one hot encoding
    :param seq: amino acid sequence (string)
    :return: one hot encoding of the amino acid (array)[L,20]
    """
    prot_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in prot_seq])
    return feat


def pickle_load(file_address):
    with open(file_address, "rb") as f:
        data = pickle.load(f)
    return data


def get_angle_degree(preds):
    preds = preds * 2 - 1
    preds_sin = preds[:, :, 0]
    preds_cos = preds[:, :, 1]
    preds_angle_rad = np.arctan2(preds_sin, preds_cos)
    preds_angle = np.degrees(preds_angle_rad)
    return preds_angle


def get_unnorm_asa(rel_asa, seqs):
    """
    :param rel_asa: predicted relative ASA, padded to the longest sequence (tensor)[B,L]
    :param seqs: protein sequences in the batch, longest first
    :return: absolute ASA (array)[B,L]
    """
    rel_asa = rel_asa.cpu().detach().numpy()
    asa_max = np.full(rel_asa.shape, UNKNOWN_RESIDUE_ASA, dtype=np.float32)
    for i, seq in enumerate(seqs):
        asa_max[i, :len(seq)] = [MAX_ASA.get(residue.upper(), UNKNOWN_RESIDUE_ASA) for residue in seq]
    return np.multiply(rel_asa, asa_max)
