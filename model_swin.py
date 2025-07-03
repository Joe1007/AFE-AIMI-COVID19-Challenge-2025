import torch.nn as nn, torch.nn.functional as F
import timm
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 使用 Swin Transformer 替代 EfficientNet
        # 選擇 swin_base_patch4_window7_224 作為backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224', 
            pretrained=True, 
            drop_rate=0.3, 
            drop_path_rate=0.2,
            num_classes=0  # 移除預設的分類頭
        )
        
        # 獲取backbone的輸出特徵維度
        # Swin Base的輸出維度通常是1024
        self.feature_dim = self.backbone.num_features
        
        # 保持原有的分類頭設計
        self.emb = nn.Linear(self.feature_dim, 224)
        self.logit = nn.Linear(224, 1)
        

    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1     # 保持原有的normalization

        # 使用Swin Transformer提取特徵
        x = self.backbone(x)  # 直接通過backbone獲取特徵
        
        # 原有的分類頭
        x = self.emb(x)
        logit = self.logit(x)
     
        return logit


def criterion(outputs, labels, gpu=None, pos_weight=None):
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight]).cuda(gpu)
        if gpu is not None:
            pos_weight = pos_weight.cuda(gpu)

    if gpu is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda(gpu)(outputs, labels)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, labels)