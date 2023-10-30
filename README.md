- [photometric vision quasar network (PVQNET)](#photometric-vision-quasar-network--pvqnet-)
  * [Ussage](#ussage)
  * [How to Install](#how-to-install)
- [Repository Structure](#repository-structure)
- [Datasets Introduction](#datasets-introduction)
  * [Available Datasets](#available-datasets)
  * [DatasetsCatalog](#datasetscatalog)
    + [OutlierTestI.csv (307 Quasars from DESI and PANSSTARS quasar surveys)](#outliertesticsv--307-quasars-from-desi-and-pansstars-quasar-surveys-)
- [How To Run The code](#how-to-run-the-code)



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

## Available Datasets

The images of the five channels u, g, r, i, and z are cropped into the form of $64 \times 64$  according to the ra and dec coordinates. Pictures of five channels of a single quasar are saved in the form of a matrix as the 'specObjId_z_zErr.mat' file. But for the 307 quasar samples obtained by cross-matching in the DESI and PANSSTARS Quasar surveys, our rule for naming the mat file is 'objId_z_SDSSNAME.mat'.

1. DataSets1 (4644 Training Datasets of our model)
2. OutlierTestDataset1 (307 Quasars from DESI and PANSSTARS quasar surveys)
3. OutlierTestDataset2 ()

## DatasetsCatalog

1. TrainingDatasets.csv (4644 Training Datasets of our model)

2. ### OutlierTestI.csv (307 Quasars from DESI and PANSSTARS quasar surveys)

3. OutlierTestII:  OutlierTestDatasetII(Mag).csv: 82,415 quasar candidates with complete magnitude in u,g,r,i,z.

   ​                         OutlierTestDatasetII(NOMag).csv: 841 quasar candidates with the lack of magnitude in u,g,r,i,z.



# How To Run The code











