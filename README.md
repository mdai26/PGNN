# Polycrystal Graph Neural Network 

This repo contains the code base for the paper [***"Polycrystal Graph Neural Network"***]()
by [Minyi Dai](https://www.linkedin.com/in/minyi-dai-7bb82b197/), 
[Mehmet F. Demirel](http://cs.wisc.edu/~demirel), 
[Xuanhan Liu](https://www.linkedin.com/in/xuanhan-liu-2976b3218/),
[Yingyu Liang](http://cs.wisc.edu/~yliang), 
[Jiamian Hu](https://mesomod.weebly.com/people.html).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Generation](#data-generation)
  - [Polycrystal Microstructure Generation](#polycrystal-microstructure-generation)
  - [Materials Property calculation](#materials-property-calculation)
  - [Available dataset](#available-dataset)
- [Polycrystal Graph Neural Network](#polycrystal-graph-neural-network)
  - [PGNN architecture](#pgnn-architecture)
  - [Train a PGNN model](#train-a-pgnn-model)
  - [Available trained model weight](#available-trained-model-weight)
- [License](license)

## Prerequisites
- Python == 3
- [Pytorch](https://pytorch.org/) == 1.10.1
- [Scikit-Learn](https://scikit-learn.org/stable/) 

Those packages can be installed in a new anaconda environment named pgnn through
```
conda create --name pgnn
conda activate pgnn
conda install python=3 scikit-learn pytorch==1.10.1 torchvision -c pytorch -c conda-forge
```



### 
## Data Generation

![Data Generation](https://github.com/mdai26/PGNN/blob/main/Images/figure1.png)

### Polycrystal Microstructure Generation

The 3D polygrain microstructure is generated through [Voronoi Tessellation](https://en.wikipedia.org/wiki/Voronoi_diagram). The code can be found in the folder [VoronoiGrain](https://github.com/mdai26/PGNN/tree/main/Voronoigrain).

- By default setting, 100 microstructures with the size of 64 x 64 x 64 can be generated through
  ```
  python datageneration.py
  ```

- The parameters used for microstructure generation can be changed through revising Line #122-#138 of datageneration.py
  ```
  # set random seed
  random_seed = 10
  np.random.seed(random_seed)
  # specify the limit of number of grains
  lowerlimit = 10
  upperlimit = 400
  # specify number of microstructures
  ndata = 100
  # get the number of grain array
  ngrain = np.random.randint(lowerlimit, upperlimit, ndata, dtype=int)
  # define number of grain thickness
  gbwidth = 1
  # define dimension of the microstruture
  Nx = 64; Ny = 64; Nz = 64
  # define randome walk for grain center
  nrwalk = 10;
  # define use periodic boundary condition or not
  periodic = True
  ```
- Introduction of output file

  - **struct_${number}.in**: document the structure id of each grid (0 for grain and 1 for grain boundary)

  - **eulerAng_${number}.in**: document the three euler angles of each grid

  - **feature_${number}.txt**: document the features of each grains

  - **neighbor_${number}.txt**: document the adjacency matrix of the microstructure

  - **mark_and_newmark_${number}.vts**: vtk file for plotting through [ParaView](https://www.paraview.org/), mark is the grain label and newmark is the structure id. 

### Materials Property calculation

Both the ionic conductivity and the Young's modulus are calculated through [MuPro](https://www.mupro.co/). The parameters can be found in the paper. Some scripts that help to run the simulation and collect the simulation results on [Euler](https://wacc.wisc.edu/resources/docs/faqs.html) can be found in folder [conductivity](https://github.com/mdai26/PGNN/tree/main/conductivity) and [elastic](https://github.com/mdai26/PGNN/tree/main/elastic).

### Available dataset

The datasets can be downloaded through this [Link](https://drive.google.com/drive/folders/1ZxbRhB0Q5BLh89LYblG_GZGJsqtsiMuq?usp=sharing).

- **Microstructure - Conductivity Dataset with graph representation (suitable for Graph Neural Network)**: 
  - GNNtraindata_unscaled.npz (training dataset with 4000 data points)
  - GNNvaliddata_unscaled.npz (validation dataset with 500 data points)
  - GNNtestdata_unscaled.npz (testing dataset with 500 data points)
  - Import the data
    ```
    GNNdata = np.load(filename)
    # node feature matrix
    nfeature = GNNdata['nfeature']
    # adjacency matrix
    neighblist = GNNdata['neighblist']
    # edge feature matrix
    efeature = GNNdata['efeature']
    # Target
    targetlist = GNNdata['targetlist']
    ```
- **Microstructure - Conductivity Dataset with image representation (suitable for Convolutional Neural Network)**
  - CNNtraindata_unscaled.npz (training dataset with 4000 data points)
  - CNNvaliddata_unscaled.npz (validation dataset with 500 data points)
  - CNNtestdata_unscaled.npz (testing dataset with 500 data points)
  - Import the data
    ```
    CNNdata = np.load(filename)
    # 3D image 
    image = np.asarray(CNNdata['imagelist'])
    # target
    target = np.asarray(CNNdata['targetlist']
    ```
    Note that the CNN and GNN dataset are for the same 5000 raw data points. The data splits are also same for the two datasets
- **Microstructure - Young's Modulus dataset with graph representation**
  - elasticdata.npz (the whole dataset with 604 data points)
  - Import the data
    ```
    GNNdata = np.load(filename)
    # node feature matrix
    nfeature = GNNdata['nfeature']
    # adjacency matrix
    neighblist = GNNdata['neighblist']
    # edge feature matrix
    efeature = GNNdata['efeature']
    # Target
    targetlist = GNNdata['targetlist']
    ```
Note that the targets of all the three datasets are not scaled. 

## Polycrystal Graph Neural Network

The code of the PGNN model is developed based on the code of [CGCNN model](https://github.com/txie-93/cgcnn).

### PGNN architecture

![PGNN](https://github.com/mdai26/PGNN/blob/main/Images/figure2.png)

### Train a PGNN model

- Put the following data and code in the same folder.
  - GNNtraindata_unscaled.npz
  - GNNvaliddata_unscaled.npz
  - GNNtestdata_unscaled.npz
  - main.py
  - model.py
  - data.py
- Train the model from scratch
  - activate conda envirnoment if not
    ```
    conda activate pgnn
    ```
  - train the model from scratch (the default parameters are the optimized hyperparameters stated in the paper)
    ```
    python main.py
    ```

### Available trained model weight

The available trained model weight using the microstructure-conductivity training dataset can be found through this [link](https://drive.google.com/drive/folders/1ZxbRhB0Q5BLh89LYblG_GZGJsqtsiMuq?usp=sharing). Note that the targets of the dataset are scaled to get the trainable weights. 

To load the model weight
```
python main.py --load_model=checkpoint.pth.tar
```
## License

PGNN is released under the MIT License.



