## Environment
### Installation
```
git clone https://github.com/ming053l/E2D.git
conda create --name e2d python=3.8 -y
conda activate e2d
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
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
