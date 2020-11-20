# ViscNet, a model to predict viscosity
[![arXiv](https://img.shields.io/badge/arXiv-2007.03719-b31b1b.svg)](https://arxiv.org/abs/2007.03719)

ViscNet is a machine learning model that can predict the temperature-dependency of viscosity of oxide liquids and the fragility index and the glass transition temperature.

## How to use
Python 3.6+ is required to run the code. The recommended procedure is to create a new virtual environment and install the necessary modules by running

``` sh
pip install -r requirements.txt
```
## Brief description of the script files
- [models.py](./viscnet/models.py): class to build the models and function to train them.
- [data.py](./viscnet/data.py): reads the data and splits it.
- [train.py](./viscnet/train.py): train ViscNet, ViscNet-Huber, and ViscNet-VFT models.
- [cross-validation.py](./viscnet/cross-validation.py): computes the cross-validation metrics.
- [metrics.py](./viscnet/metrics.py): compute the metrics of the ViscNet, ViscNet-Huber, and ViscNet-VFT models.
- [plots.py](./viscnet/plots.py): generate the plots to check the performance of the models.

## Issues and how to contribute
If you find bugs or have questions, please open an issue. PRs are most welcome.

## How to cite
Cassar, D. R. Reproducible gray-box neural network for predicting the fragility index and the temperature-dependency of viscosity. arXiv:2007.03719 [cond-mat, physics:physics] (2020).

## Database licences
Portions of the data from these databases are used and available in this repository:
- [SciGlass](https://github.com/epam/SciGlass/blob/master/LICENSE) Copyright (c) 2019 EPAM Systems
- [matminer](https://github.com/hackingmaterials/matminer/blob/master/LICENSE) Copyright (c) 2015, The Regents of the University of California
- [mendeleev](https://github.com/lmmentel/mendeleev/blob/master/LICENSE) Copyright (c) 2015 Lukasz Mentel

## ViscNet license
[GPL](https://github.com/drcassar/viscnet/blob/master/LICENSE)

ViscNet, a machine learning model to predict viscosity. Copyright (C) 2020 Daniel Roberto Cassar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
