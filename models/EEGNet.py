import torch
import torch.nn as nn
from models.layers import Conv2dWithConstraint

class backbone(nn.Module):
    def __init__(self,
                num_channels: int,
                num_bands: int,
                F1=8, D=2, F2= 'auto', T1= 125, T2=30, P1=4, P2=8, pool_mode= 'mean',
                drop_out=0.25):
        super(backbone, self).__init__()
    
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(num_bands, F1, (1, T1),  padding=(0, T1//2), bias=False),
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (num_channels, 1), padding=0, groups=F1, bias=False, max_norm=1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1), stride=4),
            nn.Dropout(drop_out)
        )
        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2),  padding=(0, T2//2), groups=F1 * D),
            nn.Conv2d(F2, F2, 1,  stride=1, bias=False, padding=0),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, P2), stride=8),
            nn.Dropout(drop_out)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.spectral(x)
        x = self.spatial(x)
        x = self.temporal(x) # 此时x形状类似 [batch, F2, 1, some_length]
        output = self.flatten(x) # 展平: [batch, F2 * some_length]
        return x, output
    

class classifier(nn.Module):
    def __init__(self, input_features: int, num_classes: int): # 修改：需要知道输入特征数
        super(classifier, self).__init__()
        # 使用一个简单的线性层（全连接层）进行分类
        self.dense = nn.Sequential(
            nn.Linear(input_features, num_classes), # 从展平特征映射到类别数
            # 移除 LogSoftmax，因为常与 CrossEntropyLoss 搭配使用（其内部包含LogSoftmax）
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # x 现在是展平后的特征向量，形状为 [batch, input_features]
        x = self.dense(x)
        # 输出形状: [batch, num_classes]
        return x


class EEGNet(nn.Module):   
    def __init__(self,
                num_classes: int,
                num_channels: int,
                num_bands: int,
                input_length: int
                ):
        super(EEGNet, self).__init__()

        self.backbone = backbone(num_channels=num_channels, num_bands=num_bands)

        dummy_input = torch.randn(1, num_bands, num_channels, input_length)
        with torch.no_grad():
            _, dummy_output = self.backbone(dummy_input)
        flattened_features = dummy_output.shape[1] # 获取展平后的特征数量
        
        # 初始化classifier时传入展平特征数
        self.classifier = classifier(input_features=flattened_features, num_classes=num_classes)

    def forward(self, x):
        # 前向传播
        _, flattened_features = self.backbone(x) # 只取展平的特征
        x = self.classifier(flattened_features) # 将展平特征送入分类器
        # 此时 x 形状为 [batch_size, num_classes]
        return x