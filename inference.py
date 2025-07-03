import os
import gc
import cv2
import math
import copy
import time
import random
import timm
import torch
from timm.models.efficientnet import efficientnet_b3a, tf_efficientnet_b4_ns, tf_efficientnetv2_s, tf_efficientnetv2_m,efficientnet_b7

from timm.models.convnext import *
# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score,roc_auc_score
import timm
from timm.models.efficientnet import *

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import glob

import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CONFIG = {"seed": 2022,
          "img_size": 256,  # 更新為訓練時的尺寸
          "valid_batch_size": 1,
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          "train_batch": 8,          
          }
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()

# 更新為你的新資料路徑
test_ct_list = list(glob.glob(os.path.join("/ssd7/ICCV2025_COVID19/test_cropped", "*")))  # 修改為你的測試資料路徑
df = pd.DataFrame(test_ct_list, columns=["path"])

# 更新字典檔案路徑
with open("/ssd7/ICCV2025_COVID19/processing_test/filter_slice_test_dic1_05_.pickle", 'rb') as f:  # 修改為你的測試字典路徑
    test_dic = pickle.load(f)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 使用 timm 創建模型，更穩定
        e = timm.create_model('efficientnet_b3a', pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        
        # 檢查並獲取正確的activation層
        try:
            act1 = e.act1
        except AttributeError:
            act1 = getattr(e, 'activation', nn.SiLU())
            
        try:
            act2 = e.act2
        except AttributeError:
            act2 = getattr(e, 'activation', nn.SiLU())
        
        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]
        self.b8 = nn.Sequential(
            e.conv_head, #384, 1536
            e.bn2,
            act2,
        )

        self.emb = nn.Linear(1536, 224)
        self.logit = nn.Linear(224, 1)
        
    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1     

        x = self.b0(x) 
        x = self.b1(x) 
        x = self.b2(x)
        x = self.b3(x) 
        x = self.b4(x) 
        x = self.b5(x) 
        x = self.b6(x) 
        x = self.b7(x) 
        x = self.b8(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        x = self.emb(x)
        logit = self.logit(x)
     
        return logit
        
class Covid19Dataset_valid(Dataset):
    def __init__(self, df, valid_dic, train_batch=10, img_size = 512, transforms=None):
        self.df = df
        self.valid_dic = valid_dic
        self.path = df['path'].values
        self.transforms = transforms
        self.img_batch = train_batch
        self.img_size = img_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.path[index]
        img_path_l = os.listdir(img_path)
        img_path_l_ = [file[2:] if file.startswith("._") else file for file in img_path_l]
        
        # 過濾掉非jpg檔案
        img_path_l_ = [f for f in img_path_l_ if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(img_path_l_) == 0:
            print(f"警告：路徑 {img_path} 沒有有效圖片")
            img_sample = torch.zeros((self.img_batch, 3, self.img_size, self.img_size))
            return {
                'image': img_sample,
                'id': img_path
            }
        
        img_list = [int(i.split('.')[0]) for i in img_path_l_]
        index_sort = sorted(range(len(img_list)), key=lambda k: img_list[k])
        ct_len = len(img_list)

        # 安全獲取字典值
        try:
            if img_path in self.valid_dic:
                dict_data = self.valid_dic[img_path]
                if isinstance(dict_data, (list, tuple)) and len(dict_data) >= 2:
                    start_idx = dict_data[0]
                    end_idx = dict_data[1]
                    if len(dict_data) > 3:
                        sample_idx = dict_data[3]
                    else:
                        sample_idx = None
                else:
                    start_idx = 0
                    end_idx = ct_len
                    sample_idx = None
            else:
                start_idx = 0
                end_idx = ct_len
                sample_idx = None
        except Exception as e:
            print(f"字典讀取錯誤 {img_path}: {e}")
            start_idx = 0
            end_idx = ct_len
            sample_idx = None
        
        img_sample = torch.zeros((self.img_batch, 3, self.img_size, self.img_size))
        
        # 處理sample_idx
        if sample_idx is None:
            if ct_len == 1:
                sample_idx = [0] * self.img_batch
            elif (end_idx - start_idx) >= self.img_batch:
                # 為了inference的一致性，使用固定間隔採樣而非隨機
                step = (end_idx - start_idx) // self.img_batch
                sample_idx = [start_idx + i * step for i in range(self.img_batch)]
                # 確保不超出範圍
                sample_idx = [min(idx, end_idx - 1) for idx in sample_idx]
            else:
                available_range = range(start_idx, min(end_idx, ct_len))
                if len(available_range) == 0:
                    available_range = range(ct_len)
                # 重複採樣填滿batch
                sample_idx = []
                for i in range(self.img_batch):
                    sample_idx.append(list(available_range)[i % len(available_range)])
        
        # 確保 sample_idx 格式正確
        if not isinstance(sample_idx, list):
            sample_idx = [sample_idx] * self.img_batch
        if len(sample_idx) < self.img_batch:
            sample_idx.extend([sample_idx[-1]] * (self.img_batch - len(sample_idx)))
        elif len(sample_idx) > self.img_batch:
            sample_idx = sample_idx[:self.img_batch]

        for count, idx in enumerate(sample_idx):
            try:
                if idx >= len(index_sort):
                    idx = len(index_sort) - 1
                if idx < 0:
                    idx = 0
                    
                img_path_ = os.path.join(img_path, img_path_l_[index_sort[idx]])
                
                img = cv2.imread(img_path_)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transforms(image=img)['image']
                img_sample[count] = img[:]
                
            except Exception as e:
                print(f"處理圖片時出錯: {e}")
                continue
                
        return {
            'image': img_sample,
            'id': img_path
        }   
        
        
def prepare_loaders(CONFIG, test_df, test_dlc, data_transforms, world_seed = None, rank=None):
    valid_dataset = Covid19Dataset_valid(test_df, test_dlc, CONFIG['train_batch'], 
                                        img_size=CONFIG['img_size'],
                                        transforms=data_transforms["valid"])
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["valid_batch_size"], 
                              num_workers=8, shuffle=False, pin_memory=True)  # 降低num_workers避免問題
    return valid_loader

data_transforms = {
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),  # 使用CONFIG中的img_size
        A.Normalize(),
        ToTensorV2()], p=1.)
}

@torch.inference_mode()
def inference(model, dataloader, device):
    model.eval()
    dataset_size = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    IDS = []
    pred_y = []
    for step, data in bar:
        ids = data["id"]
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        images = data_img.to(device, dtype=torch.float)
        batch_size = images.size(0)
        outputs = model(images)
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())
        IDS.append(ids)
        
    pred_y = np.concatenate(pred_y)
    IDS = np.concatenate(IDS)
    gc.collect()
    
    pred_y = np.array(pred_y).reshape(-1, 1)
    pred_y = np.array(pred_y).reshape(-1, img_b)
    pred_y = pred_y.mean(axis=1)
    
    return pred_y, IDS

job = 60  # 更新為你的job編號

# 更新權重路徑 - 你可以在這裡指定你已訓練好的模型路徑
weights_path_list = [
    '/ssd7/ICCV2025_COVID19/track1_hospital_0-3_model_effb3a_with_pos_weight/f1/job_60_effb7_size256_challenge[DataParallel]-fold1.bin0.9468443059936927',  # 請更新為你實際的模型路徑
    # 如果你有多個fold的模型，可以在這裡添加更多路徑
]

for j in range(len(weights_path_list)):
    bin_save_path = "/ssd7/ICCV2025_COVID19/track1_hospital_all_model_test"  # 更新模型保存路徑
    weights_path = weights_path_list[j]
    
    print("="*10, "loading *model*", "="*10)
    model = Net()
    
    # 載入模型權重
    try:
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # 處理DataParallel的權重
        if any(key.startswith('module.') for key in state_dict.keys()):
            # 權重已經有module前綴，直接載入
            model = nn.DataParallel(model)
            model.load_state_dict(state_dict)
        else:
            # 權重沒有module前綴，需要添加
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k
                new_state_dict[name] = v
            model = nn.DataParallel(model)
            model.load_state_dict(new_state_dict)
            
        print(f"成功載入模型: {weights_path}")
        
    except Exception as e:
        print(f"載入模型失敗: {e}")
        continue
    
    model = model.cuda()
    
    # 準備資料載入器
    test_loader = prepare_loaders(CONFIG, df, test_dic, data_transforms)
    
    # 單次預測
    total_pred = []
    pred_y, name = inference(model, test_loader, device=CONFIG['device'])
    total_pred.append(pred_y)
    
    final_pred = np.mean(total_pred, axis=0)
    dict_all = dict(zip(name, final_pred))
    cnn_one_pred_df = pd.DataFrame(list(dict_all.items()), columns=['path', 'pred'])
    
    # 更新輸出路徑
    output_dir = "/ssd7/ICCV2025_COVID19/output"
    os.makedirs(output_dir, exist_ok=True)
    cnn_one_pred_df.to_csv(f"{output_dir}/3_cnn_one_pred_{j+1}df.csv", index=False)
     
    # 多次預測平均（TTA）
    times_list = [10]
    for times in times_list:
        total_pred = []
        for i in range(times):
            pred_y, name = inference(model, test_loader, device=CONFIG['device'])
            total_pred.append(pred_y)
        final_pred = np.mean(total_pred, axis=0)
        dict_all = dict(zip(name, final_pred))
    
        cnn_times_pred_df = pd.DataFrame(list(dict_all.items()), columns=['path', 'pred'])
        cnn_times_pred_df.to_csv(f"{output_dir}/3_cnn_{times}_pred_{j+1}df.csv", index=False)
        print(f"save to {output_dir}/3_cnn_{times}_pred_{j+1}df.csv")