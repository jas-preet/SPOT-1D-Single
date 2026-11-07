"""Shared fixtures.

The real checkpoints are ~1.2 GB, so the end-to-end tests run the actual command
line against tiny scripted models with the same call signatures and output widths.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class _ModelWithLengths(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, feats, length):
        return torch.sigmoid(feats.sum(2, keepdim=True)).repeat(1, 1, self.width)


class _ModelWithoutLengths(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, feats):
        return torch.sigmoid(feats.sum(2, keepdim=True)).repeat(1, 1, self.width)


def _save_fake_models(jits_dir):
    jits_dir.mkdir()
    for kind, width in (("class", 11), ("reg", 12)):
        for number, model in (
            (1, _ModelWithLengths(width)),
            (2, _ModelWithoutLengths(width)),
            (3, _ModelWithLengths(width)),
        ):
            torch.jit.script(model).save(str(jits_dir / f"model{number}_{kind}_cpu.pth"))


@pytest.fixture(scope="session")
def install(tmp_path_factory):
    """A copy of the program with fake checkpoints in the usual jits/ location."""
    root = tmp_path_factory.mktemp("install")
    for name in ("spot1d_single.py", "main.py", "means_single.pkl", "stds_single.pkl"):
        shutil.copy(REPO_ROOT / name, root)
    shutil.copytree(
        REPO_ROOT / "dataset", root / "dataset", ignore=shutil.ignore_patterns("__pycache__")
    )
    _save_fake_models(root / "jits")
    return root


@pytest.fixture
def run_cli(install):
    def run(*args, cwd=install):
        return subprocess.run(
            [sys.executable, str(install / "spot1d_single.py"), *map(str, args)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )

    return run


@pytest.fixture
def make_list(tmp_path):
    """Write FASTA files and a file list that points at them."""

    def make(records, trailer=""):
        paths = []
        for filename, text in records.items():
            path = tmp_path / filename
            path.write_text(text)
            paths.append(str(path))
        file_list = tmp_path / "list.txt"
        file_list.write_text("\n".join(paths) + "\n" + trailer)
        return file_list

    return make
