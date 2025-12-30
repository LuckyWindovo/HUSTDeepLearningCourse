import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from vae_basic import VAEBasic
from cvae_advanced import CVAEAdvanced
from train_vae import MNISTDataLoader
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   

def plot_training_curves():
    """绘制训练损失曲线"""
    print("加载训练记录并绘制损失曲线...")
    
    # 加载训练记录
    train_losses = np.load('results/logs/train_losses.npy')
    train_recon_losses = np.load('results/logs/train_recon_losses.npy')
    train_kl_losses = np.load('results/logs/train_kl_losses.npy')
    test_recon_losses = np.load('results/logs/test_recon_losses.npy')
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1：训练损失
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='总损失', linewidth=2)
    axes[0].plot(epochs, train_recon_losses, label='重构损失', linewidth=2)
    axes[0].plot(epochs, train_kl_losses, label='KL散度', linewidth=2)
    axes[0].set_xlabel('训练轮数', fontsize=12)
    axes[0].set_ylabel('损失值', fontsize=12)
    axes[0].set_title('训练集损失变化', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 图2：测试集重构损失
    axes[1].plot(epochs, test_recon_losses, label='测试集重构损失', 
                 color='red', linewidth=2, marker='o', markersize=3)
    axes[1].set_xlabel('训练轮数', fontsize=12)
    axes[1].set_ylabel('损失值', fontsize=12)
    axes[1].set_title('测试集重构损失变化', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/images/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("损失曲线已保存至 results/images/training_curves.png")

def evaluate_reconstruction(model, device, target_digit=0, num_samples=5):
    """评估重构能力"""
    print(f"\n评估数字{target_digit}的重构能力...")
    
    # 加载测试数据
    data_loader = MNISTDataLoader(
        data_dir='./data', 
        batch_size=num_samples, 
        target_digit=target_digit
    )
    test_loader = data_loader.load_data(train=False)
    
    model.eval()
    with torch.no_grad():
        # 获取测试样本
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        
        # 重构
        recon, _, _ = model(test_data)
        
        # 转换为图像格式
        original = test_data.view(-1, 28, 28).cpu().numpy()
        reconstructed = recon.view(-1, 28, 28).cpu().numpy()
        
        # 可视化对比
        fig, axes = plt.subplots(2, num_samples, figsize=(12, 5))
        for i in range(num_samples):
            # 原始图像
            axes[0, i].imshow(original[i], cmap='gray')
            axes[0, i].set_title(f'原始图像', fontsize=11)
            axes[0, i].axis('off')
            
            # 重构图像
            axes[1, i].imshow(reconstructed[i], cmap='gray')
            axes[1, i].set_title(f'重构图像', fontsize=11)
            axes[1, i].axis('off')
        
        plt.suptitle(f'VAE重构效果对比 (数字{target_digit})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'results/images/reconstruction_digit_{target_digit}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"重构对比图已保存至 results/images/reconstruction_digit_{target_digit}.png")

def generate_samples(model, device, latent_dim=20, num_samples=10):
    """从潜在空间随机采样生成新图像"""
    print("\n从潜在空间随机采样生成新图像...")
    
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # 生成图像
        generated = model.decode(z)
        generated_images = generated.view(-1, 28, 28).cpu().numpy()
        
        # 可视化
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.flatten()
        for i in range(num_samples):
            axes[i].imshow(generated_images[i], cmap='gray')
            axes[i].set_title(f'生成样本 {i+1}', fontsize=11)
            axes[i].axis('off')
        
        plt.suptitle('从潜在空间随机采样生成的数字0图像', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/images/generated_samples_basic.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("生成图像已保存至 results/images/generated_samples_basic.png")

def generate_conditional_samples(cvae_model, device, num_samples_per_digit=5):
    """使用条件VAE生成指定数字的图像"""
    print("\n使用条件VAE生成0-9数字图像...")
    
    cvae_model.eval()
    fig, axes = plt.subplots(10, num_samples_per_digit, figsize=(12, 20))
    
    with torch.no_grad():
        for digit in range(10):
            # 生成指定数字的图像
            generated_images = cvae_model.generate_digit(
                digit_label=digit, 
                num_samples=num_samples_per_digit, 
                device=device
            )
            
            # 可视化
            for i in range(num_samples_per_digit):
                axes[digit, i].imshow(generated_images[i].cpu().squeeze(), cmap='gray')
                axes[digit, i].axis('off')
                if i == 0:
                    axes[digit, i].set_ylabel(f'数字 {digit}', fontsize=12, rotation=0, labelpad=30)
    
    plt.suptitle('条件VAE生成的0-9数字图像', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/images/generated_samples_cvae.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("条件生成图像已保存至 results/images/generated_samples_cvae.png")

def main():
    """主评估函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建结果目录
    os.makedirs('results/images', exist_ok=True)
    
    # 绘制损失曲线
    plot_training_curves()
    
    # 加载并评估基础VAE模型
    print("\n加载基础VAE模型...")
    vae_basic = VAEBasic(latent_dim=20).to(device)
    vae_basic.load_state_dict(torch.load('results/vae_basic_model.pth', map_location=device))
    vae_basic.eval()
    
    # 评估重构能力
    evaluate_reconstruction(vae_basic, device, target_digit=0, num_samples=5)
    
    # 随机采样生成
    generate_samples(vae_basic, device, latent_dim=20, num_samples=10)
    
    # 加载并评估条件VAE模型
    print("\n加载条件VAE模型...")
    cvae_model = CVAEAdvanced(latent_dim=20).to(device)
    cvae_model.load_state_dict(torch.load('results/cvae_advanced_model.pth', map_location=device))
    cvae_model.eval()
    
    # 条件生成
    generate_conditional_samples(cvae_model, device, num_samples_per_digit=5)
    
    print("\n所有评估完成！请检查 results/images/ 目录中的结果。")

if __name__ == '__main__':
    main()