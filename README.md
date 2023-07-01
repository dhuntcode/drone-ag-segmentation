# Example Segmentation

This repository contains code and resources for training a segmentation model.  

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

## Introduction

The goal of this project is to develop a segmentation model that can accurately classify different agricultural regions in drone images. The model is based on the U-Net architecture, which has shown promising results in various segmentation tasks.  This is a simple implimentation of it and will require further improvements.

## Installation

Python setup follows the Hyper Mordern Python blog suggestions:
https://cjolowicz.github.io/posts/hypermodern-python-01-setup/

Dependency management uses poetry:
https://python-poetry.org/docs/

Install pytorch with the correct version of cuda:
pip3 install torch torchvision torchaudio

To set up the environment and install the required dependencies, follow these steps:

1. Clone the repository:
```shell
git clone https://github.com/dhuntcode/segmentation-example.git
```
```shell
cd segmentation-example
```
```shell
poetry install
```

## Usage
To train and evaluate the segmentation model, follow these steps:

Prepare your dataset by following the instructions in the Dataset Preparation section.

Add your params to the params.yaml file.

Train the segmentation model using the train_model.py script:
shell
```
make train
```

Evaluate the trained model using the evaluate_model.py script:
shell
```
make evaluate
```

Make predictions on new images using the predict.py script:
shell
```
make predict
```

## Dataset Preparation

To prepare the dataset, 

```
make create_dataset
```
the train/test split is 0.2

## Model Training
To train the segmentation model, use the train.py script. This script takes care of loading the prepared dataset, defining the model architecture, setting up training parameters, and saving the trained model.  Model weights are saved in src/segmentation/models/saved_checkpoints

Adjust the training parameters in the script according to your specific requirements.

## Evaluation
To evaluate the trained segmentation model, use the evaluate.py script. This script loads the trained model, evaluates it on the validation set, and prints the evaluation metrics.

Adjust the evaluation parameters and metrics in the script as needed.
