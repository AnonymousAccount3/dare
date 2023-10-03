Deep Anti-Regularized Ensemble
==============================

Source code of the Deep Anti-Regularized Ensemble (DARE) experiments

This repositories contains the following python files:

DARE
----
`methods/dare.py` : code for the DARE algorithm

Competitors
-----------
- `methods/de.py` : code for Deep Ensemble
- `methods/anchor_net.py` : code for Anchored-Networks
- `methods/mod.py` : code for MOD
- `methods/negcorr.py` : code for NegCorr
- `methods/rde.py` : code for RDE

Utilities
---------
- `utils.py` : utility functions
- `preprocessing_citycam.py` : preprocessing the Citycam dataset
- `script.py` : main file to run
- `train.py` : training file
- `datasets` : load function for datasets

Experiments
-----------
  - Two-Moons Classification : `configs/2d_classif.yml`
  - 1D Regression : `configs/1d_reg.ym`
  - Citycam Weather-Shift : `configs/citycam_weather.yml`
  - CityCam BigBus-Shift : `configs/citycam_bigbus.yml`
  - CityCam Camera-Shift : `configs/citycam_camera.yml`
  - OOD Detection CIFAR10 : `configs/cifar10_resnet.yml`
  - OOD Detection Fashion-MNIST : `configs/fashionmnist.yml`


To launch the experiemnts, first pip install the requirements in a conda environment with Python version 3.8.12.
```
conda create -n dare python=3.8.12
pip install -r requirements.txt
```

The experiments can be run with the following command:
```
python script.py
```

The models are saved in the `results` folder and the scores in the `logs/results` folder.

**Note** : To run the CityCam experiments, please download the dataset at this [url](https://www.citycam-cmu.com/) and then preprocessed it with the following command line:
```
python preprocessing_citycam.py <path_to_the_citycam_dataset_on_your_labtop>
```
To run the OOD detction experiments, please download the test file of the SVHN dataset [here](http://ufldl.stanford.edu/housenumbers/test_32x32.mat), and store the file in the `datasets/svhn/` folder. For the CIFAR10 experiments, ResNet32 networks should first be pre-trained using the `configs/cifar10.yml` config file.
