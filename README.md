SPOT-1D-Single: *Improving theSingle-Sequence-Based Prediction of ProteinSecondary Structure, Backbone Angles, SolventAccessibility and Half-Sphere Exposures using aLarge Training Set and Ensembled Deep Learning.*
====
The standalone version of SPOT-1D-Single available for public use for research purposes. 

Contents
----
  * [Introduction](#introduction)
  * [Results](#results)
  * [System Requirments](#system-requirments)

# SPOT-1D-Single

Introduction
----

Results
----

System Requirments
----

**Hardware Requirments:**
SPOT-1D-single predictor requires only a standard computer with approximately 16 GB RAM to support the in-memory operations.

* [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive) (Optional if using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional if using GPU)

Installation
----

To install SPOT-1D-Single and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/jas-preet/SPOT-1D-Single.git`
2. `cd SPOT-1D-Single`

To download the model check points from the dropbox use the following commands in the terminal:

3. `wget https://apisz.sparks-lab.org:8443/downloads/Resource/Protein/2_Protein_local_structure_prediction/jits.tar.xz`
4. `tar -xvf jits.tar.xz`

To install the dependencies and create a conda environment use the following commands

5. `conda create -n spot_single_env python=3.7`
6. `conda activate spot_single_env`

if GPU computer:
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`

for CPU only 
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch`

8. `conda install pandas=1.1.1`

Execute
----
To run SPOT-1D-Single use the following command

`python spot1d_single.py --file_list file_lists/file_list_casp13.txt --save_path results/ --device cuda:0`

or 

`python spot1d_single.py --file_list file_lists/file_list_casp13.txt --save_path results/ --device cpu` 

Datasets
----
`wget https://www.dropbox.com/s/2k4l6u82pwbadgl/datasets.tar.xz`

**License**
----
MIT LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
