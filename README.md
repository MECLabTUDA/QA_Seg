# Quality monitoring of federated Covid-19 lesion quantification
Using this repository, a linear classification model can be trained to predict the quality of a predicted Covid-19 lesion segmentations. The model uses 4 features for its prediction, namely
1. "dice_scores" : Also called segmentation smoothness. Smoothness is defined as the average dice score between two consecutive slices of a connected component, which is again     averaged over all connected components in the segmentation.
2. "connected components" : The number of connected components in the segmentation
3. "gauss params" : The mean of a normal distribution fitted on sampled intensity values 
    of the 4 biggest components.
4. "seg in lung" : The percentage of the segmented (as infectious tissue) area, that lies 
    within the segmentation of the lung 
    
The classes range from 1 to 5, getting displayed as 0.2(corresponding to 1 and being the worst) to 1.0(corresponding to 5 and being the best) during inference. In our experiments we declared segmentations with a score of less or equal to 0.6 as 'failed'. 

## Installation
The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):

1. Create a Python3.8 environment as follows: `conda create -n <your_anaconda_env> python=3.8` and activate the environment.
2. Install CUDA and PyTorch through conda with the command specified by [PyTorch](https://pytorch.org/). The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`. The code was developed and last tested with the PyTorch version 1.6.
3. Navigate to the project root (where setup.py lives).
4. Execute `pip install -r requirements.txt` to install all required packages.
5. Set your paths in mp.paths.py.
6. Execute `git update-index --assume-unchanged mp/paths.py` so that changes in the paths file are not tracked in the repository.


## Training and Inference using the model

## JIP Datastructure
The whole preprocessing, training and inference is based on the data stored in the following structure:

    ../JIP/
    ├── data_dirs/
    │   ├── input/
    |   │   ├── patient_00
    |   │   |    ├── img
    |   │   |       ├── img.nii.gz
    |   |   ├── ...
    │   ├── ...
    ├── preprocessed_dirs/
    │   ├── ...
    ├── train_dirs/
    │   ├── input/
    |   │   ├── patient_00
    |   │   |    ├── img
    |   │   |       ├── img.nii.gz
    |   │   |    ├── seg
    |   │   |       ├── 001.nii.gz
    |   │   |    ├── pred 
    |   │   |       ├── 001.nii.gz
    |   |   ├── ...
    |   ├── ...
   

The corresponding paths need to be set in [paths.py](../mp/paths.py) before starting any process. For instance, only the `storage_path` variable needs to be set -- in the example above it would be `../JIP`.

The data for inference *-- data_dirs/input --* and training *-- train_dirs/input --* needs to be provided by the user with respect to the previously introduced structure before starting any process. If this is not done properly, neither one of the later presented methods will work properly, since the data will not be found, thus resulting in an error during runtime. The preprocessed folder will be automatically generated during preprocessing and should not be changed by the user. Note that the folders (`patient_00`, etc.) can be named differently, however the name of the corresponding scan needs to be `img.nii.gz`, a Nifti file located in `img/` folder. The corresponding segmentation needs to be named `001.nii.gz`, also a Nifti file but located in the `seg/` folder.


## Data Pipeline
The preprocessing and feature extraction of the data is done in the background for inference as well as for training and does not need to be done manually. The data pipeline consists of :
1. Copying the data from the input directory into the preprocess directory and resizing it. 
2. Masking out the segmented tissue with label unequal the given number (default value is 1) 
3. Computation of the lung segmentations using https://github.com/JoHof/lungmask
4. Extracion of the 4 features for inference/training data
The implementation of all preprocessing methods can be found in /mp/utils/preprocess_utility_functions' 

## Training classifiers
After the data has been put into the train_dir as described above, the model can be trained by executing:
```bash
<your_anaconda_env> $ python seg_qual_JIP.py --mode train --label <label_in_segmentation_mask>
```
Here the '--label' argument specifies which label in the segmentation mask is used for the training. The default value is 1. 
The model can be found at under '*--data_dirs/persistent' 

## Performing inference
Performing inference on data is also very straight-forward. To start the inference, the following command needs to be executed:
```bash
<your_anaconda_env> $ python seg_qual_JIP.py --mode inference --label <label_in_segmentation_mask>
```

After sucessfull termination, the `metrics.json` file is generated under '*--data_dirs/output'  and has the following structure:
```
{	
    "patient_00":	{
                        TODO
                    },
    ...
}
```
