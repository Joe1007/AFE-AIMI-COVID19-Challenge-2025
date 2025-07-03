import os, random, cv2, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class Covid19Dataset(Dataset):
    def __init__(self, df, train_dict, train_batch=10, img_size = 512, transforms=None):
        self.df = df
        self.train_dict = train_dict
        self.img_size = img_size
        # self.file_names = df['filename'].values
        
        self.transforms = transforms
        self.img_batch=train_batch
        self.not_allow=['/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/covid/ct_scan_106',
                        '/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/covid/ct_scan_55',
                        '/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/covid/ct_scan_79',
                        '/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/non-covid/ct_scan_12',
                        '/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/non-covid/ct_scan_136',
                        '/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/non-covid/ct_scan_160',
                        '/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/non-covid/ct_scan_45',
                        '/ssd7/ICCV2025_COVID19/track1_fixed/train_pure_crop_challenge/non-covid/ct_scan_498']
        
        self.df = df[~df['path'].isin(self.not_allow)]
        self.path = df['path'].values
        self.labels = df['label'].values
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        label = self.labels[index]
        img_path = self.path[index]

        img_path_l = os.listdir(img_path)
        img_path_l_ = [file[2:] if file.startswith("._") else file for file in img_path_l]
        # 過濾掉非jpg檔案
        img_path_l_ = [f for f in img_path_l_ if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(img_path_l_) == 0:
            # 如果沒有有效圖片，返回空的樣本
            print(f"警告：路徑 {img_path} 沒有有效圖片")
            img_sample = torch.zeros((self.img_batch, 3, self.img_size, self.img_size))
            label_sample = torch.zeros((self.img_batch, 1))
            return {
                'image': img_sample,
                'label': torch.tensor(label_sample, dtype=torch.long)
            }
        
        img_list = [int(i.split('.')[0]) for i in img_path_l_]
        index_sort = sorted(range(len(img_list)), key=lambda k: img_list[k])
        
        ct_len = len(img_list)
        
        # 安全獲取字典值
        try:
            if img_path in self.train_dict:
                dict_data = self.train_dict[img_path]
                if isinstance(dict_data, (list, tuple)) and len(dict_data) >= 2:
                    start_idx = dict_data[0]
                    end_idx = dict_data[1]
                    # 檢查是否有預定義的sample_idx
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
        label_sample = torch.zeros((self.img_batch, 1))
        
        # 重新設計 sample_idx 邏輯
        if sample_idx is None:
            if ct_len == 1:
                sample_idx = [0] * self.img_batch  # 重複使用唯一的圖片
            elif (end_idx - start_idx) >= self.img_batch:
                # 從範圍內隨機選擇
                sample_idx = random.sample(range(start_idx, min(end_idx, ct_len)), 
                                         min(self.img_batch, end_idx - start_idx))
                # 如果樣本不足，重複最後一個
                while len(sample_idx) < self.img_batch:
                    sample_idx.append(sample_idx[-1])
            else:
                # 從整個範圍隨機選擇，允許重複
                available_range = range(start_idx, min(end_idx, ct_len))
                if len(available_range) == 0:
                    available_range = range(ct_len)
                sample_idx = [random.choice(available_range) for _ in range(self.img_batch)]
        
        # 確保 sample_idx 是列表且長度正確
        if not isinstance(sample_idx, list):
            sample_idx = [sample_idx] * self.img_batch
        if len(sample_idx) < self.img_batch:
            # 補齊到所需長度
            sample_idx.extend([sample_idx[-1]] * (self.img_batch - len(sample_idx)))
        elif len(sample_idx) > self.img_batch:
            # 截斷到所需長度
            sample_idx = sample_idx[:self.img_batch]
            
        # 處理圖片載入
        successful_loads = 0
        for count, idx in enumerate(sample_idx):
            if count >= self.img_batch:
                break
                
            try:
                # 確保索引在有效範圍內
                if idx >= len(index_sort):
                    idx = len(index_sort) - 1
                if idx < 0:
                    idx = 0
                    
                img_path_ = os.path.join(img_path, img_path_l_[index_sort[idx]])
                
                if not os.path.exists(img_path_):
                    print(f"圖片不存在: {img_path_}")
                    continue
                    
                img = cv2.imread(img_path_)
                if img is None:
                    print(f"無法讀取圖片: {img_path_}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transforms(image=img)['image']
                
                img_sample[count] = img[:]
                label_sample[count] = label
                successful_loads += 1
                
            except Exception as e:
                print(f"處理圖片時出錯 {img_path_}: {e}")
                print(f"count: {count}, idx: {idx}, len(index_sort): {len(index_sort)}")
                # 使用零填充或跳過
                continue
        
        if successful_loads == 0:
            print(f"警告：沒有成功載入任何圖片 from {img_path}")
            
        return {
            'image': img_sample,
            'label': torch.tensor(label_sample, dtype=torch.long)
        }

class Covid19Dataset_valid(Dataset):
    def __init__(self, df, valid_dic, train_batch=10, img_size = 512, transforms=None):
        self.df = df
        self.valid_dic = valid_dic
        # self.file_names = df['filename'].values
        self.path = df['path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        self.img_batch=train_batch
        self.img_size = img_size
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        label = self.labels[index]
        img_path = self.path[index]
        img_path_l = os.listdir(img_path)
        img_path_l_ = [file[2:] if file.startswith("._") else file for file in img_path_l]
        # 過濾掉非jpg檔案
        img_path_l_ = [f for f in img_path_l_ if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(img_path_l_) == 0:
            print(f"警告：路徑 {img_path} 沒有有效圖片")
            img_sample = torch.zeros((self.img_batch, 3, self.img_size, self.img_size))
            label_sample = torch.zeros((self.img_batch, 1))
            return {
                'image': img_sample,
                'label': torch.tensor(label_sample, dtype=torch.long)
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
        label_sample = torch.zeros((self.img_batch, 1))
        
        # 重新設計 sample_idx 邏輯（與訓練集相同）
        if sample_idx is None:
            if ct_len == 1:
                sample_idx = [0] * self.img_batch
            elif (end_idx - start_idx) >= self.img_batch:
                sample_idx = list(range(start_idx, min(start_idx + self.img_batch, end_idx, ct_len)))
                while len(sample_idx) < self.img_batch:
                    sample_idx.append(sample_idx[-1])
            else:
                available_range = range(start_idx, min(end_idx, ct_len))
                if len(available_range) == 0:
                    available_range = range(ct_len)
                sample_idx = [random.choice(available_range) for _ in range(self.img_batch)]
        
        # 確保 sample_idx 格式正確
        if not isinstance(sample_idx, list):
            sample_idx = [sample_idx] * self.img_batch
        if len(sample_idx) < self.img_batch:
            sample_idx.extend([sample_idx[-1]] * (self.img_batch - len(sample_idx)))
        elif len(sample_idx) > self.img_batch:
            sample_idx = sample_idx[:self.img_batch]

        for count, idx in enumerate(sample_idx):
            if count >= self.img_batch:
                break
                
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
                label_sample[count] = label
                
            except Exception as e:
                print(f"處理圖片時出錯: {e}")
                continue
                
        return {
            'image': img_sample,
            'label': torch.tensor(label_sample, dtype=torch.long)
        }

        
def prepare_loaders(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms, world_seed = None, rank=None):
    train_dataset = Covid19Dataset(train_df, train_dic, CONFIG['train_batch'], 
                                    img_size = CONFIG["img_size"] , transforms=data_transforms["train"])
    valid_dataset = Covid19Dataset_valid(valid_df, valid_dlc, CONFIG['train_batch'], 
                                img_size = CONFIG["img_size"] , transforms=data_transforms["valid"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["train_batch_size"],
                            num_workers=8, shuffle=True, pin_memory=True, drop_last=True)  # 降低 num_workers
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["valid_batch_size"],
                            num_workers=8, shuffle=False, pin_memory=True)  # 降低 num_workers

    return train_loader, valid_loader

def prepare_loaders_eval(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms):
    train_dataset = Covid19Dataset(train_df, train_dic, CONFIG['train_batch'], transforms=data_transforms["valid"])
    valid_dataset = Covid19Dataset_valid(valid_df, valid_dlc, CONFIG['train_batch'], transforms=data_transforms["valid"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["train_batch_size"], 
                              num_workers=5, shuffle=False, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["valid_batch_size"], 
                              num_workers=5, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader