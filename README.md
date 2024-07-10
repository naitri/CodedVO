# CodedVO: Coded Visual Odometry

Official implementation of "CodedVO: Coded Visual Odometry" accepted in IEEE Robotics and Automation Letters,2024.  

[[Project page](http://prg.cs.umd.edu/CodedVO)], [[CodedVO arxiv](https://ieeexplore.ieee.org/document/DOI_NUMBER_HERE)]  

![Example of coded aperture setup](assets/coded_setup.jpg)



```bibtex
@ARTICLE{10564186,
  author={Shah, Sachin and Rajyaguru, Naitri and Singh, Chahat Deep and Metzler, Christopher and Aloimonos, Yiannis},
  journal={IEEE Robotics and Automation Letters}, 
  title={CodedVO: Coded Visual Odometry}, 
  year={2024},
  doi={10.1109/LRA.2024.3416788}}
```
## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
   - [Clone Repository](#clone-repository)
   - [Environment Setup](#environment-setup)
3. [Usage](#usage)
   - [Run Visual Odometry](#run-visual-odometry)
4. [Models](#models)
   - [Download Pre-trained Models](#download-pre-trained-models)
5. [Dataset](#dataset)
   - [Download and Setup](#download-and-setup)
6. [Training](#training)
   - [Train from Scratch](#train-from-scratch)
7. [Bibtex](#bibtex)
8. [License](#license)

## Introduction
- A novel method for estimating monocular visual odometry that leverages RGB and metric depth estimates obtained through a phase mask on a standard 1-inch camera sensor.
- A depth-weighted loss function designed to prioritize learning depth maps at closer distances.
- Evaluation in zero-shot indoor scenes without requiring a scale for evaluation.

## Installation

### Clone Repository

```bash
git clone https://github.com/naitri/CodedVO
cd CodedVO
```

## Environment Setup

```bash
conda create -n codedvo python=3.8
conda activate codedvo
pip install -r requirements.txt
