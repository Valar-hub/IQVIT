- [photometric vision quasar network (PVQNet)](#photometric-vision-quasar-network--pvqnet-)
  * [Ussage](#ussage)
  * [How to Install](#how-to-install)
- [Repository Structure](#repository-structure)
- [Datasets Introduction](#datasets-introduction)
  * [Available Datasets](#available-datasets)
  * [DatasetsCatalog](#datasetscatalog)
- [How To Run The code](#how-to-run-the-code)
  * [Training the model](#training-the-model)
  * [Evaluate the model](#evaluate-the-model)
- [Additional Information](#additional-information)








# photometric vision quasar network (PVQNet)

## Ussage

PVQNet is a neural network for finding high-redshift quasars (z>5) through photometric images. 

## How to Install

The python version of our code is python 3.9.13. Install the relevant package dependencies by:

```
pip install requirements.txt
```

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
3. OutlierTestDataset2 (It will be available and uploaded in public cloud)

## DatasetsCatalog

1. TrainingDatasets.csv (4644 Training Datasets of our model)

2. OutlierTestI.csv (307 Quasars from DESI and PANSSTARS quasar surveys)

3. OutlierTestII:  OutlierTestDatasetII(Mag).csv: 82,415 quasar candidates with complete magnitude in u,g,r,i,z.

   ​                         OutlierTestDatasetII(NOMag).csv: 841 quasar candidates with the lack of magnitude in u,g,r,i,z.



# How To Run The code



## Training the model

```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env gpu_train.py --model=VIT --data-path=../DataSets/DataSets1/
#use PVQNet model to train the model. More specific parameters can be found in detailed in gpu_train.py. ‘CUDA_VISIBLE_DEVICE=[id]’ means to specific GPU ID, nproc_per_node means to the number of GPUS that you want to use. './weights' will reserve the trained model weights, while './log' output the logs during the PVQNet training.

```

## Evaluate the model

```python
python evaluate_model.py --model=VIT --weights=PVQNet.pth --data-path=../DataSets/OutlierTestDataset1/
```

 

# Additional Information

For the 83,256 candidate source images contained in OutlierTestDatasetII, we only provide catalog, not all *.mat images. This is due to their large size and we are considering uploading them to a public cloud later.

Operating system: Linux version 5.15.0-86-generic (buildd@lcy02-amd64-086) (gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, GNU ld (GNU Binutils for Ubuntu) 2.38)

GPU: GeForce RTX™ 4080















