# BS-Diff: Effective Bone Suppression in CXRs via Conditional Diffusion Models

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Awesome-FC60A8?style=flat-square&logo=Awesome&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)

<div align=center><img width="500" height="500" src="https://github.com/Benny0323/BS-Diff/assets/104205136/e8edb3b0-559d-4a61-90ac-9a6ea53e7a4e)"/></div>

### 🧨 Congratulations! Our paper has been accepted by ISBI 2024(Oral Presentation)!

## Proposed method 

We spend a lot of time collecting and summarizing relevant papers and datasets, where you can find them at https://github.com/diaoquesang/A-detailed-summarization-about-bone-suppression-in-Chest-X-rays

This code is a PyTorch implementation of our paper "BS-Diff: Effective Bone Suppression in CXRs via Conditional Diffusion Models".

Our proposed framework comprises two stages: **a conditional diffusion model (CDM) equipped with a U-Net architecture and a simple enhancement module** that incorporates an autoencoder. It can not only generate soft tissue images with **a high bone suppression ratio** but also possess the capability to **capture fine image information and spatial features,
while preserving overall structures**. The figure below shows our proposed network.

![image](https://github.com/Benny0323/BS/blob/main/framework.png)

## The bone-suppressed images generated by our method
![image](https://github.com/Benny0323/BS/blob/main/contrast.png)

## Comparison performance with previous works (visualization)
![image](https://github.com/Benny0323/BS/blob/main/Comparison.png)

## Clinical evaluation
The results below demonstrated that our soft-tissues can **clearly preserve the visibility of pulmonary vessels and central airways and greatly suppress bones**, significantly improving the clinician’s performance in finding lung lesions. Each criterion has a maximum score of 3.
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Centered Table</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f4f4f4;
        }
        table {
            border-collapse: collapse;
            width: 80%;
            max-width: 800px;
            background: white;
            text-align: center;
        }
        th, td {
            border: 1px solid black;
            padding: 10px;
        }
        th {
            background: #ddd;
        }
        td[rowspan] {
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th colspan="2">Clinical Evaluation Criteria</th>
                <th>Junior Clinician</th>
                <th>Intermediate Clinician</th>
                <th>Senior Clinician</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">Pulmonary vessels visibility</td>
                <td>Clearly displayed (3)</td>
                <td rowspan="3">2</td>
                <td rowspan="3">3</td>
                <td rowspan="3">3</td>
            </tr>
            <tr><td>Displayed (2)</td></tr>
            <tr><td>Not displayed (1)</td></tr>
            <tr>
                <td rowspan="3">Central airway visibility</td>
                <td>Lobar and intermediate bronchi (3)</td>
                <td rowspan="3">2</td>
                <td rowspan="3">3</td>
                <td rowspan="3">2</td>
            </tr>
            <tr><td>Main bronchus and rump (2)</td></tr>
            <tr><td>Trachea (1)</td></tr>
            <tr>
                <td rowspan="3">Degree of bone suppression</td>
                <td>Nearly perfect suppression (3)</td>
                <td rowspan="3">2</td>
                <td rowspan="3">3</td>
                <td rowspan="3">2</td>
            </tr>
            <tr><td>Unsuppressed bones less than 5 (2)</td></tr>
            <tr><td>5 or more bones unsuppressed (1)</td></tr>
        </tbody>
    </table>
</body>
</html>

## Pre-requisties
* Linux

* Python>=3.7

* NVIDIA GPU (memory>=6G) + CUDA cuDNN

### Download the dataset
Now, we only provide three paired images with CXRs and soft-tissues via pre-processing. Soon, we will make them available to the public after data usage permission. Three paired images are located at
```
├─ Data
│    ├─ BS_Aug
│    │    ├─ 0.png
│    │    ├─ 1.png
│    │    └─ 2.png
│    ├─ CXR_Aug
│    │    ├─ 0.png
│    │    ├─ 1.png
│    │    └─ 2.png
```

## Getting started to evaluate
### Install dependencies
```
pip install -r requirements.txt
```
### Download the checkpoint
Due to the fact that our proposed model comprises two stages, you need to download both stages' checkpoints to successfully run the codes!
These two files can be found in the following link : 

https://drive.google.com/drive/folders/1cDlXJ7Sh4k05aM_tvzor9_F_TPCeIGMN?usp=sharing

### Evaluation
To do the evaluation process, first run the following command in stage 1 (the conditional diffusion model):
```
python Test.py
```      
Then, you will get a series of images generated by the conditional diffusion model. After that, run the following command in stage 2 with these images as inputs.
```
python Hybrid_autoencodereval.py
```
## Train by yourself
If you want to train our model by yourself, you are primarily expected to split the whole dataset into training, validation, and testing. You can find the codes in **Data Spliting** directory and run the following commands one by one:
```
python txt.py
python split.py
```
Then, you can run the following command in stage 1:
```
python Train.py
```
Then after finishing stage 1, you can use the generated output of stage 1 to train our stage (enhancement module) by running the following command:
```
python Hybridloss_autoencoder.py
```
These two files are located at
```
├─ Stage1
│    └─ Train.py
├─ Stage2
│    ├─ Hybridloss_autoencoder.py
│    └─ pytorch_msssim.py
```

## Evaluation metrics
You can also run the following commands about evaluation metrics in our experiment including PSNR, SSIM, MSE and BSR:
```
python metrics.py
```
## Citation
```
@inproceedings{chen2024bs,
  title={BS-Diff: Effective Bone Suppression Using Conditional Diffusion Models From Chest X-Ray Images},
  author={Chen, Zhanghao and Sun, Yifei and Ge, Ruiquan and Qin, Wenjian and Pan, Cheng and Deng, Wenming and Liu, Zhou and Min, Wenwen and Elazab, Ahmed and Wan, Xiang and others},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Benny0323/BS-Diff&type=Timeline)](https://star-history.com/#Benny0323/BS-Diff&Timeline)
