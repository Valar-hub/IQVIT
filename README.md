[TOC]

# photometric vision quasar network (PVQNET)
## Ussage
PVQNet is a neural network for finding high-redshift quasars (z>5) through photometric images. 

## How to Install
The python version of our code is python 3.9.13. Install the relevant package dependencies by 'pip install requirements.txt'. 



# Repository Structure
This github repository releases the following files:
- relevant model source code of PVQNet.
- pre-trained model weight files.
- relevant datasets used in the training and testing processes.
- newly identified quasar candidates.

# Datasets Introduction
The images of the five channels u, g, r, i, and z are cropped into the form of 64 $\times$ 64 according to the ra and dec coordinates. Pictures of five channels of a single quasar are saved in the form of a matrix as the 'specObjId_z_zErr.mat' file. But for the 307 quasar samples obtained by cross-matching in the DESI and PANSSTARS Quasar surveys, our rule for naming the mat file is 'objId_z_SDSSNAME.mat'.
