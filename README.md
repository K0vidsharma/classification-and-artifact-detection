# AI-Generated Image classification and Artifact Detection

This repository contains team-57's implementation of Problem Statement-7 of InterIIT Tech Meet 13.

# Repository Structure
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
│   ├── report.pdf
│   └── requriments.txt
└── README.md
```

## Task 1
- **Dataset**: Contains the data used for Task 1.
- **Model**: Includes the model and associated scripts for training and evaluation.
- **Results**: 
  - JSON file with explanations generated for Task 1.

## Task 2
- **Dataset**: Contains the data used for Task 2.
- **Model**: Includes the model and associated scripts for training and evaluation.
- **Results**: 
  - JSON file with explanations generated for Task 2.


# Directory  Overview:
## Task 1:
- model_checkpoint.pth.tar: The checkpoint of the final classification model.
- train.ipynb: Code for training
- resnet.py: The Code of our custom resnet50 included with freqnet and wavelet attention.
- wavelet.py: Code of wavelet attention.
- freqnet.py: Code of Freqnet Attention.
- pytorch_wavelets: Library used to create wavelet models.
- inference.py: code for infference
## Task 2: 
- CLIP.ipynb: Code for Identification of Artifacts
- Explanation.py: Code for Explanation of Artifacts
- Dataset.zip: Dataset created by us for fine tuning the VLM
- VLM_train.py: The training code for Idefics8B
- VLM_inference.py: The inference code for Idefics8B
- classification_results.json
- raw_explanation.json: this file contain the output given by the model

57_task1.json

57_task2.json

requirements.txt

report.pdf

README.md


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

## Installation

```bash
pip install -r requirements.txt 
