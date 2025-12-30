import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEBasic(nn.Module):
    """基础VAE模型 - 仅处理MNIST数字0"""
    def __init__(self, latent_dim=20, input_dim=784):
        super(VAEBasic, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # 编码器：将输入映射到潜在空间的均值和对数方差
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)      # 潜在变量的均值向量
        self.fc_logvar = nn.Linear(128, latent_dim)  # 潜在变量的对数方差向量
        
        # 解码器：从潜在空间重建输入
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 输出像素值在[0,1]范围内
        )
    
    def encode(self, x):
        """编码器前向传播"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧：从N(mu, var)中采样，同时保持梯度可传递"""
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)    # 从标准正态分布采样
        return mu + eps * std
    
    def decode(self, z):
        """解码器前向传播"""
        return self.decoder(z)
    
    def forward(self, x):
        """完整的前向传播过程"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        VAE损失函数：重构损失 + KL散度
        
        参数:
            recon_x: 重建的图像
            x: 原始输入图像
            mu: 潜在变量的均值
            logvar: 潜在变量的对数方差
            beta: KL散度的权重系数
        """
        # 重构损失：使用BCE损失
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL散度：衡量潜在分布与标准正态分布的差异
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss