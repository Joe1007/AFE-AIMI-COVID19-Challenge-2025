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


## How To Inference



## How To Train & Inference
- Run `./preproceesing/*` folder step by step (**Note: you need to change directory & path within all file.**)
- then the command below:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

- Run `./preproceesing/inference/*` step by step (**Note:you need to change directory & path within all file.**)
- get and the weights(.bin) after training in `weights_path_list`
- then the command below:
```
CUDA_VISIBLE_DEVICES=0,1 python inference.py
```
