import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    """可配置参数的CNN模型（Medium复杂度）"""
    def __init__(self, num_classes=10, activation='relu', use_dropout=True, 
                 dropout_rate=0.3, use_batchnorm=True, kernel_size=3):
        super(FlexibleCNN, self).__init__()
        
        # 确保kernel_size为奇数
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # 卷积层1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        
        # 卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        
        # 池化和Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(dropout_rate) if use_dropout else nn.Identity()
        self.dropout_fc = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)  # inplace加速
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # 全连接层
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


def get_model(model_type='flexible', **kwargs):
    """获取模型实例"""
    if model_type == 'flexible':
        return FlexibleCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")