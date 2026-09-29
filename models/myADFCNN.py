import math
import torch
import torch.nn as nn
from models.layers import Conv2dWithConstraint, LazyLinearWithConstraint, PositionalEncodingFourier

class backbone(nn.Module):
    def __init__(self,
                 num_channels: int,  # 脑电通道数 C (如22, 3, 20)
                 input_timepoints: int, # 时间点 T (如750，对应250Hz*3s)
                 F1=8, D=1, F2='auto',
                 pool_mode='mean',
                 drop_out=0.25):
        super(backbone, self).__init__()

        if F2 == 'auto':
            F2 = F1 * D
        pooling_layer = {'max': nn.MaxPool2d, 'mean': nn.AvgPool2d}[pool_mode]

        # ---------- 分支 I：大尺度（提取低频、全局信息） ----------
        # 1. 时域卷积 (大核，提取低频信息)
        self.temporal_conv_b1 = nn.Sequential(
            # 输入形状: [B, 1, C, T]
            # 对“空间维度”（电极）做卷积 -> 跨电极混合信息？不，这里需要澄清。
            # 根据论文，时域卷积是在“时间维度”上滑动，输入通道应为1（代表一个虚拟的“高度”），输出F1个特征图。
            # 但常见EEG CNN实现（如EEGNet）将电极维度视为“空间高度”，时间维度是“宽度”。
            # 论文描述可能易混淆。结合其引用[27]和表I，更常见的实现是：
            # 卷积核形状: [F1, 1, 1, kernel_length]，在时间维(T)上做卷积，独立处理每个电极通道(C)。
            # 即深度可分离卷积的思想：先对每个电极做时域滤波，再用逐点卷积混合。
            # 然而，论文表I明确写 temporal conv 的 kernel 是 [F x 1 x 125]，输出是 [F x C x T]。
            # 这意味着输入被看作 [C, T] 的图像，用 F 个 1x125 的核去卷积，每个核产生一个通道的输出。
            # 所以，输入形状应为 [B, 1, C, T]，卷积层设置 groups=C 来实现“深度卷积”。
            # 但为简化，我们先按一个更直观、常见的结构实现，确保分支独立和注意力正确。

            # 修正方案：采用与EEGNet Block 1类似的逻辑
            # 层1: 深度卷积，对每个电极进行独立的时域滤波
            Conv2dWithConstraint(in_channels=1,  # 将电极C视为“空间”维度，用1个输入通道代表
                               out_channels=F1 * D,  # 深度乘子D，通常D=1
                               kernel_size=(1, 125),  # 在时间维卷积
                               padding='same',
                               groups=1,  # 这里不是深度卷积，因为我们把C放到了“高度”
                               bias=False,
                               max_norm=2.),
            nn.BatchNorm2d(F1 * D),
            # 注意：此时输出形状为 [B, F1*D, 1, T]
        )
        # 2. 可分离空间卷积 (提取全局空间信息)
        self.spatial_conv_b1 = nn.Sequential(
            # 深度卷积部分：对每个特征图进行空间滤波
            Conv2dWithConstraint(in_channels=F1 * D,
                               out_channels=F1 * D,
                               kernel_size=(num_channels, 1),  # 在电极（通道）维度卷积
                               padding=0,  # 无填充，会消减电极维度
                               groups=F1 * D,  # 深度卷积：每个输入通道独立卷积
                               bias=False,
                               max_norm=2.),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.Dropout(drop_out),
            # 逐点卷积部分 (Point-wise Conv): 混合特征图
            Conv2dWithConstraint(in_channels=F1 * D,
                               out_channels=F2,
                               kernel_size=(1, 1),
                               padding=0,
                               bias=True,
                               max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        )
        # 3. 池化层
        self.pool_b1 = nn.Sequential(
            pooling_layer(kernel_size=(1, 32), stride=(1, 32)),
            nn.Dropout(drop_out)
        )

        # ---------- 分支 II：小尺度（提取高频、细节信息） ----------
        # 1. 时域卷积 (小核，提取高频信息)
        self.temporal_conv_b2 = nn.Sequential(
            Conv2dWithConstraint(in_channels=1,
                               out_channels=F1,
                               kernel_size=(1, 30),
                               padding='same',
                               groups=1,
                               bias=False,
                               max_norm=2.),
            nn.BatchNorm2d(F1),
        )
        # 2. 标准空间卷积 (提取细节空间信息)
        self.spatial_conv_b2 = nn.Sequential(
            # 标准卷积：一次性混合所有输入特征图和空间信息
            Conv2dWithConstraint(in_channels=F1,
                               out_channels=F2,
                               kernel_size=(num_channels, 1),
                               padding=0,
                               groups=1,  # 标准卷积
                               bias=False,
                               max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer(kernel_size=(1, 75), stride=(1, 25)),  # 注意：论文中是平均池化
            ActLog(),
            nn.Dropout(drop_out)
        )

        # ---------- 注意力融合模块 ----------
        # 论文中注意力作用于通道维(F2)。我们需要知道拼接后时间维的长度来计算线性层的输入尺寸。
        # 由于池化步长，长度会变化。我们通过一个前向探测来计算。
        self._initialize_attention_params(num_channels, input_timepoints, F2)

    def _initialize_attention_params(self, num_channels, input_timepoints, F2):
        # 使用虚拟输入计算拼接后的特征图尺寸
        with torch.no_grad():
            dummy_input = torch.randn(2, 1, num_channels, input_timepoints)
            # 前向传播至拼接后
            out_b1 = self.temporal_conv_b1(dummy_input)
            out_b1 = self.spatial_conv_b1(out_b1)
            out_b1 = self.pool_b1(out_b1)  # [B, F2, 1, T1]

            out_b2 = self.temporal_conv_b2(dummy_input)
            out_b2 = self.spatial_conv_b2(out_b2)  # [B, F2, 1, T2]

            T1 = out_b1.shape[-1]
            T2 = out_b2.shape[-1]
            self.T1, self.T2 = T1, T2

            # 注意力层输入特征数: F2
            self.attention_feat_dim = F2
            # 定义Q, K, V的线性变换层
            self.w_q = nn.Linear(self.attention_feat_dim, self.attention_feat_dim)
            self.w_k = nn.Linear(self.attention_feat_dim, self.attention_feat_dim)
            self.w_v = nn.Linear(self.attention_feat_dim, self.attention_feat_dim)

        self.dropout = nn.Dropout(0.25)  # 用于注意力后的Dropout

    def forward(self, x):
        # 输入 x 形状: [B, 1, C, T]
        # ---------- 分支 I ----------
        x_b1 = self.temporal_conv_b1(x)  # -> [B, F1*D, 1, T]
        x_b1 = self.spatial_conv_b1(x_b1)  # -> [B, F2, 1, T] (电极维被消减)
        x_b1 = self.pool_b1(x_b1)         # -> [B, F2, 1, T1]

        # ---------- 分支 II ----------
        x_b2 = self.temporal_conv_b2(x)  # -> [B, F1, 1, T]
        x_b2 = self.spatial_conv_b2(x_b2)  # -> [B, F2, 1, T2]

        # ---------- 特征拼接 ----------
        # 在时间维度上拼接: [B, F2, 1, T1+T2]
        x_concat = torch.cat([x_b1, x_b2], dim=-1)

        # ---------- 通道自注意力 (关键修正) ----------
        # 目标: 计算F2个通道之间的注意力
        # 重塑: [B, F2, 1, T_total] -> [B, T_total, F2]
        B, C, H, W = x_concat.shape
        x_for_attn = x_concat.reshape(B, C, -1).permute(0, 2, 1)  # [B, T_total, F2]

        # 线性变换得到 Q, K, V
        q = self.w_q(x_for_attn)  # [B, T_total, F2]
        k = self.w_k(x_for_attn)  # [B, T_total, F2]
        v = self.w_v(x_for_attn)  # [B, T_total, F2]

        # 缩放点积注意力
        d_k = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, T_total, T_total]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T_total, T_total]

        # 应用注意力权重到V
        attended = torch.matmul(attn_weights, v)  # [B, T_total, F2]

        # 残差连接
        x_attn_out = x_for_attn + self.dropout(attended)  # [B, T_total, F2]

        # ---------- 输出 ----------
        # 为分类器展平: 恢复形状后展平
        # 先转回 [B, C, 1, T_total] 格式以便与后续结构兼容（如果需要），然后展平
        x_attn_out = x_attn_out.permute(0, 2, 1).unsqueeze(2)  # [B, F2, 1, T_total]
        x_flat = torch.flatten(x_attn_out, start_dim=1)  # [B, F2 * T_total]

        return x_flat

class classifier(nn.Module):
    def __init__(self, input_features: int, num_classes: int):
        super(classifier, self).__init__()
        # 使用一个简单的线性层（全连接层）进行分类
        self.dense = nn.Sequential(
            nn.Linear(input_features, num_classes),
            # 注意：通常与CrossEntropyLoss搭配时，不在这里添加LogSoftmax
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # x 的形状应为 [batch, input_features]
        x = self.dense(x)
        # 输出形状: [batch, num_classes]
        return x

class ADFCNN(nn.Module):   
    def __init__(self,
                num_classes: int,
                num_channels: int,
                num_bands: int,
                input_length: int):
        super(ADFCNN, self).__init__()
        self.backbone = backbone(num_channels=num_channels,input_timepoints=input_length)

        # 关键：动态获取backbone输出的特征维度
        # 创建一个虚拟输入来探测特征维度
        with torch.no_grad():
            # 假设输入形状为 [1, num_bands, num_channels, timepoints]，这里timepoints需与训练数据一致（例如2000）
            dummy_input = torch.randn(1,num_bands, num_channels, input_length)
            dummy_features = self.backbone(dummy_input)
            flattened_features = dummy_features.shape[1]

        self.classifier = classifier(input_features=flattened_features, num_classes=num_classes)

    def forward(self, x):
        # 前向传播
        features = self.backbone(x)  # features形状: [batch, flattened_features]
        out = self.classifier(features)  # out形状: [batch, num_classes]
        return out

class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass
    def forward(self, x):
        return torch.square(x)

class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps
    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))