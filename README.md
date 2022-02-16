# Prediction the conductivity of polycrystalline materials
### 1. Generate three-dimensional polycrystalline structure with different parameters in the grain growth model
```
python changepara.py
```
### 2. Convert and collect the data of polycrystalline structure
```
python dataconvert.py
```
### 3. Calculate the conductivity of the polycrystalline microstructures with different conductivity of Grain and Grain boundary
```
python changecond.py
```
### 4. Collect the conductivity data
```
python readcond.py
```
### 5. Perform GNN training
```
python main.py
```
