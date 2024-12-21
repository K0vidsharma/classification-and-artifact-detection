# AI-Generated Image classification and Artifact Detection

This repository contains team-57's implementation of Problem Statement-7 of InterIIT Tech Meet 13.

Dataset for task-1: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Training Notebook for task-1: https://www.kaggle.com/code/k0vidsharma/kovid-adobe-ensemble

Model for task-1(Kaggle Link): https://www.kaggle.com/models/k0vidsharma/resnet50-freqnet-waveletattention/

#### Team Members:
1. Kovid Sharma (Team Lead)
2. Bathula Satwik
3. Gouri Verma
4. Sai Nikhita Palisett
5. Aditya Giri
6. Unmilan Das

## Installation and Downloads
#### Installing Pytorch Wavelets for task-1
```bash
$ git clone https://github.com/fbcotter/pytorch_wavelets
$ cd pytorch_wavelets
$ pip install .
```
#### Installing the remaining libraries

```bash
$ pip install -r requirements.txt
```

## Repository Structure
```
.
├── AI Image Artifacts.txt
├── Adobe Additional PS Info.pdf
├── LICENSE
├── PS-7.pdf
├── PS-7_Team-57
│   ├── 57_task1.json
│   ├── 57_task2.json
│   ├── Task-1
│   │   ├── freqnet.py
│   │   ├── inference.py
│   │   ├── resnet.py
│   │   ├── train.ipynb
│   │   └── wavelet.py
│   ├── Task-2
│   │   ├── CLIP.ipynb
│   │   ├── Explaination.py
│   │   ├── VLM_inference.py
│   │   ├── VLM_train.py
│   │   ├── classification_results.json
│   │   └── raw_explaination.json
│   ├── presentation.pdf
│   ├── report.pdf
│   └── requriments.txt
└── README.md
```

<!-- ## Task 1
- **Dataset**: Contains the data used for Task 1.
- **Model**: Includes the model and associated scripts for training and evaluation.
- **Results**: 
  - JSON file with explanations generated for Task 1.

## Task 2
- **Dataset**: Contains the data used for Task 2.
- **Model**: Includes the model and associated scripts for training and evaluation.
- **Results**: 
  - JSON file with explanations generated for Task 2. -->

# Pipeline Of Our project:

## Task 1:
# FreqNet and ResNet Architecture

This repository implements a hybrid architecture combining spatial and frequency domain processing for image forgery detection. It integrates *FreqNet* for frequency-domain feature extraction, *Wavelet Attention* mechanisms, and a *ResNet-inspired backbone* for robust spatial feature extraction.

## Architecture Overview

1. *Input Processing*
   - Transforms the RGB input image into the frequency domain using *FFT*.
   - Suppresses low-frequency components, enhancing high-frequency details.
   - Converts the frequency domain back to the spatial domain using *iFFT*.

2. *FreqNet*
   - Alternates between *HFRFSBlock* and *FCLBlock* for frequency-based feature extraction.
   - Uses *stride-2 convolutions* to reduce dimensions.

3. *Wavelet Attention*
   - Applies *Discrete Wavelet Transform (DWT)* for multi-scale feature extraction.
   - Enhances important high-frequency regions using a *softmax-based attention* mechanism.

4. *ResNet-Inspired Backbone*
   - Uses residual blocks with convolution layers (1x1, 3x3) for spatial feature extraction.
   - Applies *wavelet attention* in deeper layers for enriched multi-scale representation.

5. *Feature Aggregation & Classification*
   - *Adaptive average pooling* reduces spatial dimensions.
   - A *fully connected layer* performs binary classification.
# Task 2: Vision-Language Model (VLM) for Explanation Generation

The Task 2 pipeline generates explanations for images detected as fake by the Task 1 pipeline. Below is an overview of the pipeline steps:

## Pipeline Overview

1. *Input Image*: An image identified as fake by Task 1 is passed into Task 2.
2. *CLIP Model: The image is processed through the **CLIP* model to extract relevant artifacts associated with the image.
3. *Idefics2 Model: The extracted artifacts and the image are input into the pretrained **Idefics2* model.
4. *Explanation Generation: **Idefics2* generates detailed explanations, outlining why the image is detected as fake.

## Pre-trained Models and Setup

- *CLIP Model*: Used for extracting relevant artifacts associated with the input image.
- *Idefics2 Model*: A pretrained model for generating explanations based on the image and its corresponding artifacts.

## Evaluation and Logging

- *Evaluation Steps*: Evaluations are conducted every 10 steps.
- *Logging Steps*: Logs are generated every 5 steps to monitor the process.
- *Checkpoint Management*: Checkpoint saving is limited to ensure efficiency.

# References
1. ResNet code inspired from: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py

2. Pytorch Wavelets: Cotter, F. (2019). Uses of Complex Wavelets in Deep Convolutional Neural Networks [Apollo - University of Cambridge Repository]. https://doi.org/10.17863/CAM.53748




