import numpy as np
import pytest
import torch

from dataset.data_functions import (
    get_angle_degree,
    get_unnorm_asa,
    one_hot,
    read_fasta_file,
    read_list,
)


def test_one_hot_marks_the_matching_residue():
    encoded = one_hot("AV")
    assert encoded.shape == (2, 20)
    assert encoded[0, 0] == 1 and encoded[0].sum() == 1
    assert encoded[1, 19] == 1 and encoded[1].sum() == 1


def test_one_hot_encodes_unknown_residues_as_minus_one():
    assert (one_hot("X") == -1).all()


def test_read_list_skips_blank_lines(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("a.fasta\n\n  b.fasta  \n   \n")
    assert read_list(path) == ["a.fasta", "b.fasta"]


def test_read_fasta_joins_wrapped_lines_and_ignores_whitespace(tmp_path):
    path = tmp_path / "x.fasta"
    path.write_text(">x description\nMKV \n\nLA\r\n\n")
    assert read_fasta_file(path) == "MKVLA"


def test_angle_is_recovered_from_scaled_sin_and_cos():
    angle = np.radians(30.0)
    scaled = (np.array([[[np.sin(angle), np.cos(angle)]]]) + 1) / 2
    assert get_angle_degree(scaled)[0, 0] == pytest.approx(30.0)


def test_asa_is_scaled_by_each_residues_maximum():
    rel_asa = torch.tensor([[0.5, 0.5]])
    assert get_unnorm_asa(rel_asa, ("AG",)).tolist() == [[57.5, 37.5]]


def test_asa_treats_lowercase_like_uppercase():
    rel_asa = torch.tensor([[0.5]])
    assert get_unnorm_asa(rel_asa, ("a",)).tolist() == [[57.5]]


def test_asa_treats_nonstandard_residues_like_x():
    rel_asa = torch.tensor([[0.5, 0.5]])
    assert get_unnorm_asa(rel_asa, ("BX",)).tolist() == [[0.5, 0.5]]


def test_asa_handles_a_padded_batch():
    rel_asa = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert get_unnorm_asa(rel_asa, ("AG", "A")).tolist() == [[57.5, 37.5], [57.5, 0.5]]
