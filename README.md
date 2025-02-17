# DFG-DDM
Code implementation of the DFG-DDM: Deep Frequency-Guided Denoising Diffusion Model for remote sensing image dehazing.

## Requirement

- ubuntu, torch==1.10.0+cu113, torchvision==0.11.0+cu113

## Data preparation

Put the training and testing data to corresponding folders
The final file path should be the same as the following:

   data
    ├─ ... (dataset name)
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │   │   └─ ... (corresponds to the former)
    │   │   └─train.txt
    │   └─ test
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │   │   └─ ... (corresponds to the former)
    │   │   └─test.txt


## Train

1. Hyperparameter(./configs/haze.yml)
2. Execute the training command
```bash```
python train.py --config "haze.yml"

## Test

1. Hyperparameter(./configs/haze.yml)
2. Execute the testing command
```bash```
python eval_diffusion.py --config "haze.yml" --test_set 'rsid'


## Random Haze Distribution Dataset for Remote Sensing dehazing (RHDRS)

Download from Baidu Cloud:  
[Download Link](https://pan.baidu.com/s/137xO7BbPMtMrC3NCrnRAgg?pwd=98dk)

## Contact Us

If you have any questions, please contact us:  
[zhikichan@mail.wpu.edu.cn](mailto:zhikichan@mail.wpu.edu.cn)