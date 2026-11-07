# SPOT-1D-Single

*Improving the Single-Sequence-Based Prediction of Protein Secondary Structure, Backbone Angles, Solvent Accessibility and Half-Sphere Exposures using a Large Training Set and Ensembled Deep Learning.*

The standalone version of SPOT-1D-Single, available for public use for research purposes. It predicts per-residue structural properties from a protein sequence alone, without needing a multiple sequence alignment.

![SPOT-1D-Single](https://github.com/jas-preet/SPOT-1D-Single/assets/34640945/34ee403a-a8f9-40ab-ad1a-e1970452163c)

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Limitations](#limitations)
- [Datasets](#datasets)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)

## Requirements

- About 1 GB of free RAM on CPU (measured peak: 0.7 GB for a 3,000-residue protein)
- Python 3 with PyTorch, NumPy and pandas
- Optional: an NVIDIA GPU with CUDA

Tested on CPU with Python 3.12, PyTorch 2.15, NumPy 2.5 and pandas 3.1. The checkpoints were exported with PyTorch 1.6 on Python 3.7, and `requirements.txt` does not pin versions. If your PyTorch cannot load the checkpoints, use the [original environment](#original-environment).

## Installation

```bash
git clone https://github.com/jas-preet/SPOT-1D-Single.git
cd SPOT-1D-Single
pip install -r requirements.txt
```

Download the model checkpoints (about 1.2 GB) and check the download. The server only offers plain HTTP, so don't skip the checksum.

```bash
curl -LO http://183.36.5.251:8080/sparks_downloads/servers/downloads/Resource/Protein/2_Protein_local_structure_prediction/jits.tar.xz
echo "8fef1e8091866f43f56f6c9e02f719607e0825a64c37d7fe8df7c2e3a07d1ea0  jits.tar.xz" | shasum -a 256 -c
tar -xf jits.tar.xz
```

If the link stops working, look for SPOT-1D-Single on the [Zhou lab download page](https://zhouyq-lab.szbl.ac.cn/download/).

### Original environment

```bash
conda create -n spot_single_env python=3.7
conda activate spot_single_env
conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch   # GPU: replace cpuonly with cudatoolkit=10.1
conda install pandas=1.1.1
```

## Usage

```bash
python spot1d_single.py --file_list file_lists/file_list_casp13.txt --save_path results/ --device cpu
```

| Option | Description |
| --- | --- |
| `--file_list` | Required. Text file with one FASTA path per line. Relative paths are resolved from the current directory. |
| `--save_path` | Output directory, created if needed. Default: `results/` |
| `--device` | `cpu` (default) or `cuda:0` |
| `--jits_dir` | Directory holding the checkpoints. Default: `jits/` next to the script |

Each FASTA file must contain one sequence. Lower-case letters are accepted, a trailing `*` is removed, and residues other than the 20 standard amino acids are treated as unknown. Unknown residues have no reference surface area, so their `ASA` value (between 0 and 1) is a placeholder and does not mean the residue is buried.

## Output

Each input produces `<save_path>/<fasta file name>.csv`, with one row per residue.

| Column | Meaning |
| --- | --- |
| `AA` | Amino acid |
| `SS3`, `SS8` | Predicted 3-state and 8-state secondary structure |
| `ASA` | Accessible surface area |
| `HseU`, `HseD` | Half-sphere exposure, up and down |
| `CN` | Contact number |
| `Psi`, `Phi`, `Theta`, `Tau` | Backbone angles in degrees |
| `P3C`, `P3E`, `P3H` | 3-state probabilities |
| `P8C`, `P8S`, `P8T`, `P8H`, `P8G`, `P8I`, `P8E`, `P8B` | 8-state probabilities |

## Limitations

These come from how the checkpoints were exported with `torch.jit.trace`:

- Proteins are processed one at a time, because the checkpoints do not support batches larger than 1 ([#1](https://github.com/jas-preet/SPOT-1D-Single/issues/1)).
- The GPU checkpoints only run on `cuda:0`.

## Datasets

```bash
curl -L -o datasets.tar.xz "https://www.dropbox.com/scl/fi/ek78e18wq80jvet0ewf9a/datasets.tar.xz?rlkey=u04bs6ocqieoswrryotdt3d1r&dl=1"
```

## Tests

The tests use small stand-in models, so they don't need the checkpoints.

```bash
pip install pytest
pytest
```

## Citation

Singh J, Litfin T, Paliwal K, Singh J, Hanumanthappa AK, Zhou Y. SPOT-1D-Single: improving the single-sequence-based prediction of protein secondary structure, backbone angles, solvent accessibility and half-sphere exposures using a large training set and ensembled deep learning. *Bioinformatics*. 2021;37(20):3464-3472. [doi:10.1093/bioinformatics/btab316](https://doi.org/10.1093/bioinformatics/btab316)

## License

[MIT](LICENSE)
