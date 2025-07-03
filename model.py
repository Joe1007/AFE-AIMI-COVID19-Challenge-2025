import torch.nn as nn, torch.nn.functional as F
import timm
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 使用 timm 創建模型，更穩定
        e = timm.create_model('efficientnet_b3a', pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        
        # 檢查並獲取正確的activation層
        # 新版本可能叫 activation 而不是 act1
        try:
            act1 = e.act1
        except AttributeError:
            act1 = getattr(e, 'activation', nn.SiLU())  # 備用activation
            
        try:
            act2 = e.act2
        except AttributeError:
            act2 = getattr(e, 'activation', nn.SiLU())  # 備用activation
        
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


def criterion(outputs, labels, gpu=None, pos_weight=None):
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight]).cuda(gpu)
        if gpu is not None:
            pos_weight = pos_weight.cuda(gpu)

    if gpu is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda(gpu)(outputs, labels)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, labels)