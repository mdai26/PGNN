# Microstructure generation based on the experimental EBSD data

Instructions to generate the microstructures based on the experimenal EBSD data. The related results can be found in the paper [Graph Neural Network for Predicting the Effective Properties of Polycrystalline Materials:
A Comprehensive Analysis](https://arxiv.org/ftp/arxiv/papers/2209/2209.05583.pdf).

## Dataset download

The dataset described in the paper [Multi-modal Dataset of a Polycrystalline Metallic Material: 3D Microstructure and Deformation Fields](https://www.nature.com/articles/s41597-022-01525-w) is utilized for the microstructure generation.
The dataset can be downloaded via [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.83bk3j9sj). 

## Preprocessing 

1) Extract the Euler Angle and Feature id of each grid via the Dream3D file **718_final_yield.dream3d** in the downloaded zipped foler **3D_EBSD_dataset_Voxelized.zip**. 
An example Dream3D pipeline named **readdata.json** is provided in this folder, which can be run using [Dream3D](http://dream3d.bluequartz.net/). 
Two files named "EulerAngles.txt" and "FeatureIds.txt" will be output. 

2) Convert the whole microstructure with the size of 549 x 420 x 526 to a list of small cubes with the size of 70 x 70 x 70.
```
python preprocessing.py
```
Note that the two output files should be in the same folder with the code.

## microstructure generation

The microstructures are generated via the following steps.

1) Remark the grains
2) Flip the grains for plotting purpose. Note the corresponding graphs and images are almost the same before and after flipping.
The only difference is the grain center position, but the relative position of different grains is kept considering the periodic boundary condition. 
3) Add grain boundaries
4) Remove grains with zero areas
5) Generate random Euler Angles for grain boundary grids
6) Extract grain features and output

All the above steps can be done via
```
python datageneration.py
```


