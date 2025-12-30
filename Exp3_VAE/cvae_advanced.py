import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAEAdvanced(nn.Module):
    """条件VAE模型 - 处理MNIST数字0-9"""
    def __init__(self, latent_dim=20, num_classes=10, input_dim=784, condition_dim=10):
        super(CVAEAdvanced, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # 条件嵌入：将类别标签转换为稠密向量
        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        
        # 编码器：输入图像 + 条件信息
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # 解码器：潜在变量 + 条件信息
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, labels):
        # 获取条件嵌入向量
        cond_emb = self.condition_embedding(labels)
        
        # 将输入图像和条件信息拼接
        x_cond = torch.cat([x, cond_emb], dim=1)
        
        # 通过编码器网络
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        # 获取条件嵌入向量
        cond_emb = self.condition_embedding(labels)
        
        # 将潜在变量和条件信息拼接
        z_cond = torch.cat([z, cond_emb], dim=1)
        
        # 通过解码器网络
        return self.decoder(z_cond)
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def generate_digit(self, digit_label, num_samples=1, device='cpu'):
        self.eval()
        with torch.no_grad():
            # 从标准正态分布采样潜在变量
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # 创建对应的标签
            labels = torch.full((num_samples,), digit_label, dtype=torch.long).to(device)
            
            # 生成图像
            generated_images = self.decode(z, labels)
            return generated_images.view(-1, 1, 28, 28)