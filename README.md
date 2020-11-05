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
Knowing protein secondary and other one-dimensional structural properties is essential for accurate protein structure and function prediction. As a result, many methods have been developed for predicting these one-dimensional structural properties.  However,  most methods relied on evolutionary information  that  may  not  exist  for  many  proteins  due  to  a  lack  of  sequence  homologs.  Moreover,  it is  computationally  intensive  for  obtaining  evolutionary  information  as  the  library  of  protein  sequences continues to expand exponentially. Here we developed a new single-sequence method called SPOT-1D-Single based on a large training dataset of 46790 proteins deposited prior to 2016 and an ensemble of hybrid Long-Short-Term-Memory bidirectional neural network and convolutional neural network.

Results
----
We  showed  that  SPOT-1D-Single  consistently  improve  over  SPIDER3-Single  for  secondary structure, solvent accessibility, contact numbers, and backbone angles for all four independent test sets (TEST2018,  TEST2020,  CASP12 and CASP13 free-modelling targets).  For example,  the accuracy of predicted secondary structure ranges from 72.0-74.4% by SPOT-1D-single, compared to 69.2-72.6% by SPIDER3-Single. The improvement is the combination of the larger training set with ensembled learning.

System Requirments
----

**Hardware Requirments:**
SPOT-1D-single predictor requires only a standard computer with approximately 16 GB RAM to support the in-memory operations.

* [Python3](https://docs.python-guide.org/starting/install3/linux/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive) (Optional If using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional If using GPU)

Installation
----

To install SPOT-1D-Single and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/jas-preet/SPOT-1D-Single.git`
2. `cd SPOT-1D-Single`

To download the model check points from the dropbox use the following commands in the terminal:

3. `wget https://www.dropbox.com/s/gree7whebuovz0w/jits.tar.xz`
4. `tar -xz jits.tar.xz`

To install the dependencies and create a conda environment use the following commands

5. `conda create -n spot_single_env python3.7`
6. `conda activate spot_single_env`

if GPU computer:
7. `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

for CPU only 
7. `conda install pytorch torchvision cpuonly -c pytorch`

8. `conda install pandas=1.1.1`

To run SPOT-1D-Single use the following command

9. `python spot1d_single.py --file_list file_lists/file_list_casp13.txt --save_path results/ --device cuda:0 --batch 10`

or 

9. `python spot1d_single.py --file_list file_lists/file_list_casp13.txt --save_path results/ --device cpu --batch 1` 


