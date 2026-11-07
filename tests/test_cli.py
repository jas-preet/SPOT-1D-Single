"""End-to-end tests of `python spot1d_single.py ...`."""
import shutil

import pandas as pd
import pytest

COLUMNS = [
    "AA", "SS3", "SS8", "ASA", "HseU", "HseD", "CN", "Psi", "Phi", "Theta", "Tau",
    "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G", "P8I", "P8E", "P8B",
]
SIMPLE = {"prot.fasta": ">prot\nMKVLA\n"}


def read_csv(path):
    return pd.read_csv(path, index_col=0, keep_default_na=False)


def test_writes_one_row_per_residue_with_documented_columns(run_cli, make_list, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    result = run_cli("--file_list", make_list(SIMPLE), "--save_path", out)
    assert result.returncode == 0, result.stderr
    table = read_csv(out / "prot.csv")
    assert list(table.columns) == COLUMNS
    assert "".join(table["AA"]) == "MKVLA"


def test_creates_missing_save_directory(run_cli, make_list, tmp_path):
    out = tmp_path / "does" / "not" / "exist"
    result = run_cli("--file_list", make_list(SIMPLE), "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


def test_runs_from_a_different_working_directory(run_cli, make_list, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    result = run_cli("--file_list", make_list(SIMPLE), "--save_path", out, cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


def test_jits_dir_option_finds_checkpoints_elsewhere(run_cli, make_list, install, tmp_path):
    moved = tmp_path / "weights"
    shutil.copytree(install / "jits", moved)
    out = tmp_path / "out"
    result = run_cli("--file_list", make_list(SIMPLE), "--save_path", out, "--jits_dir", moved)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


@pytest.mark.parametrize("sequence", ["MKBZUOJXA", "mkvla", "MKVLA*"])
def test_accepts_unusual_residues(run_cli, make_list, tmp_path, sequence):
    out = tmp_path / "out"
    result = run_cli(
        "--file_list", make_list({"odd.fasta": f">odd\n{sequence}\n"}), "--save_path", out
    )
    assert result.returncode == 0, result.stderr
    assert "".join(read_csv(out / "odd.csv")["AA"]) == sequence.rstrip("*").upper()


def test_keeps_dots_in_output_names(run_cli, make_list, tmp_path):
    records = {"P1.v1.fasta": ">a\nMKVLA\n", "P1.v2.fasta": ">b\nMKVLA\n"}
    out = tmp_path / "out"
    result = run_cli("--file_list", make_list(records), "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == ["P1.v1.csv", "P1.v2.csv"]


def test_ignores_blank_lines_in_file_list(run_cli, make_list, tmp_path):
    out = tmp_path / "out"
    result = run_cli("--file_list", make_list(SIMPLE, trailer="\n  \n"), "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


def assert_clean_failure(result, message):
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert message in result.stderr


def test_rejects_unsupported_device(run_cli, make_list, tmp_path):
    result = run_cli("--file_list", make_list(SIMPLE), "--device", "cuda:1")
    assert_clean_failure(result, "cuda:1")


def test_requires_file_list(run_cli):
    assert_clean_failure(run_cli(), "--file_list")


def test_reports_missing_checkpoints_with_pointer_to_readme(run_cli, make_list, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_cli("--file_list", make_list(SIMPLE), "--jits_dir", empty)
    assert_clean_failure(result, "README")


def test_reports_missing_fasta_file(run_cli, tmp_path):
    file_list = tmp_path / "list.txt"
    file_list.write_text("nope.fasta\n")
    assert_clean_failure(run_cli("--file_list", file_list), "nope.fasta")


@pytest.mark.parametrize(
    "text, reason",
    [
        ("MKVLA\n", "header"),
        (">only_a_header\n", "no sequence"),
        (">a\nMKV\n>b\nLAA\n", "one sequence"),
    ],
)
def test_reports_malformed_fasta_by_name(run_cli, make_list, tmp_path, text, reason):
    result = run_cli("--file_list", make_list({"bad.fasta": text}))
    assert_clean_failure(result, "bad.fasta")
    assert reason in result.stderr
