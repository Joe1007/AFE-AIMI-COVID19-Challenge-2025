# ACVLab AFE-AIMI-COVID19-Challenge-2025

This repository is the code of our lab [ACVLAB] used in AFE-AIMI-COVID19-Challenge-2025 (https://ai-medical-image-analysis.github.io/5th/#features).
We provide all the processed files and necessary codes in this repository.


## Environment Installation
Use the command as below:
```
git clone https://github.com/Joe1007/AFE-AIMI-COVID19-Challenge-2025.git
conda create --name e2d python=3.8 -y
conda activate e2d 
pip install -r requirements.txt
# CUDA 11.6
```
Then you can get the virtual enviroment and packages we use

## Data Structure
For the format transformation problem(mac to window), you can conduct `./preproceesing/0-data_setup.ipynb` & `./preproceesing/inference/1-spatial removal.ipynb` to get the `*_fixed` dataset (Not necessary)
```
track1/
# Traingset
├── train/
│   ├── annotations/          # CSV files
│   ├── covid/
│   │   ├── ct_scan_0/        # *.jpg
│   │   ├── ct_scan_1/
│   │   └── ...
│   └── non-covid/
│       ├── ct_scan_0/
│       ├── ct_scan_1/
│       └── ...
└── val/
    ├── annotations/          # CSV files
    ├── covid/
    │   ├── ct_scan_0/
    │   └── ...
    └── non-covid/
        ├── ct_scan_0/
        └── ...

# Testingset
test/
  ├── ct_scan_0/        # *.jpg
  ├── ct_scan_1/
  ├── ct_scan_2/
  └── ...
  ├── ct_scan_1487/

precessing_test/
  ├── filter_slice_test.csv/        
  ├── filter_slice_test_dic1_05_.pickle/
  ├── test_area_df.csv/
  ├── test_dic1_05_.pickle/
```

## How To Train & Inference

Train
---
- Run `./preproceesing/*` folder step by step (**Note: you need to change directory & path within all file.**)
- Then run the command below:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py (for efficientNet)

CUDA_VISIBLE_DEVICES=0,1 python train_swin.py (for swin_transformer)
```

Inference
---
- Run `./preproceesing/inference/*` step by step (**Note:you need to change directory & path within all file.**)
- Get and the weights(.bin) after training, and put it in `weights_path_list` in `inference.py`
- Then run the command below:
```
CUDA_VISIBLE_DEVICES=0,1 python inference.py
```

Quick Demo
---
- Unzip the `preprocessing_test` folder, and modify the paths in files to fit your directory(you can reference the data structure we mention above)
- Run `./preproceesing/inference/1-spatial removal.ipynb`, and the get `test_cropped` folder
- Access the link(https://drive.google.com/drive/folders/1izXmN-rRdZIiSpaZsozlLbZxah7u8L0T?usp=sharing), and put .bin in `weights_path_list` in `inference.py`
- Run `CUDA_VISIBLE_DEVICES=0,1 python inference.py`

## Reminder
- Note that the original data files are too large, we didn't put them into our repository.

## Copyright
- Author: Chih-Chung Hsu
e-mail: chihchung@nycu.edu.tw

- Author: Chia-Ming Lee
e-mail: zuw408421476@gmail.com

- Author: Bo-Cheng Qiu
e-mail: a36492183@gmail.com

- Author: Ting-Yao Chen
e-mail: xpple413208@gmail.com

- Author: Ming-Han Sun
e-mail: harris910815@gmail.com

- Author: Jun-Lin Chen
e-mail: u109029067@gmail.com

- Author: Jung-Tse Tsai
e-mail: jungtsetsai@gmail.com

- Author: I-An Tsai
e-mail: iantsai@gmail.com

- Author: Yu-Fan Lin
e-mail: aas12as12as12tw@gmail.com
