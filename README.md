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


## How To Test

- Run `./preproceesing/inference/*` step by step and then (you need to change directory within all file.)
```
CUDA_VISIBLE_DEVICES=0,1 python inference.py
```

## How To Train
- Run `./preproceesing/*` step by step and then (you need to change directory within all file.)
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
