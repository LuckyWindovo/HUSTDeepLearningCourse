import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import numpy as np
from vae_basic import VAEBasic
from cvae_advanced import CVAEAdvanced
import matplotlib.pyplot as plt
from datetime import datetime

class MNISTDataLoader:
    def __init__(self, data_dir, batch_size=128, target_digit=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_digit = target_digit
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  
        ])
        
    def load_data(self, train=True):
        # 加载完整数据集
        dataset = datasets.MNIST(
            root=self.data_dir, 
            train=train, 
            download=False, 
            transform=self.transform
        )
        
        # 如果指定了目标数字，则筛选数据
        if self.target_digit is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label == self.target_digit]
            dataset = Subset(dataset, indices)
            
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=train,
            num_workers=0,  
            pin_memory=False
        )
        
        return dataloader

def train_basic_vae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建结果保存目录
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    # 超参数设置
    hyperparams = {
        'latent_dim': 20,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'beta': 1.0  # KL散度权重
    }
    
    # 数据加载
    data_loader = MNISTDataLoader(
        data_dir='./data', 
        batch_size=hyperparams['batch_size'], 
        target_digit=0
    )
    train_loader = data_loader.load_data(train=True)
    test_loader = data_loader.load_data(train=False)
    
    print("数据加载完成")
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    
    # 模型初始化
    model = VAEBasic(latent_dim=hyperparams['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    # 训练记录
    train_losses = []
    train_recon_losses = []
    train_kl_losses = []
    test_recon_losses = []
    
    # 训练循环
    print("\n开始训练基础VAE模型...")
    for epoch in range(1, hyperparams['num_epochs'] + 1):
        model.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # 前向传播
            recon, mu, logvar = model(data)
            
            # 计算损失
            loss, recon_loss, kl_loss = model.loss_function(
                recon, data, mu, logvar, beta=hyperparams['beta']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # 每100个批次打印一次
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{hyperparams["num_epochs"]}] '
                      f'Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'(Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})')
        
        # 计算平均损失
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_recon_loss)
        train_kl_losses.append(avg_kl_loss)
        
        print(f'Epoch {epoch} 完成 - 平均损失: {avg_train_loss:.4f} '
              f'(重构: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})')
        
        # 在测试集上评估重构能力
        model.eval()
        test_recon_loss = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                _, recon_loss, _ = model.loss_function(recon, data, mu, logvar)
                test_recon_loss += recon_loss.item()
        
        avg_test_recon_loss = test_recon_loss / len(test_loader.dataset)
        test_recon_losses.append(avg_test_recon_loss)
        print(f'测试集重构损失: {avg_test_recon_loss:.4f}')
    
    np.save('results/logs/train_losses.npy', np.array(train_losses))
    np.save('results/logs/train_recon_losses.npy', np.array(train_recon_losses))
    np.save('results/logs/train_kl_losses.npy', np.array(train_kl_losses))
    np.save('results/logs/test_recon_losses.npy', np.array(test_recon_losses))
    
    torch.save(model.state_dict(), 'results/vae_basic_model.pth')
    print("\n基础VAE模型训练完成并已保存！")
    
    return model, hyperparams

def train_cvae_advanced():
    """训练进阶条件VAE模型（数字0-9）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 超参数
    hyperparams = {
        'latent_dim': 20,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'beta': 1.0
    }
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=0)
    
    # 模型初始化
    model = CVAEAdvanced(latent_dim=hyperparams['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    print("\n开始训练条件VAE模型...")
    for epoch in range(1, hyperparams['num_epochs'] + 1):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            # 前向传播
            recon, mu, logvar = model(data, labels)
            
            # 计算损失
            loss, recon_loss, kl_loss = model.loss_function(
                recon, data, mu, logvar, beta=hyperparams['beta']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{hyperparams["num_epochs"]}] '
                      f'Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch} 完成 - 平均损失: {avg_loss:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'results/cvae_advanced_model.pth')
    print("\n条件VAE模型训练完成并已保存！")
    
    return model, hyperparams

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 训练基础VAE
    basic_model, basic_params = train_basic_vae()
    
    # 训练进阶CVAE
    cvae_model, cvae_params = train_cvae_advanced()