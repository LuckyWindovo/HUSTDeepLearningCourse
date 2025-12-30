import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from vae_basic import VAEBasic
from cvae_advanced import CVAEAdvanced
from train_vae import MNISTDataLoader
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")   

def plot_training_curves():
    """绘制训练损失曲线"""
    print("加载训练记录并绘制损失曲线...")
    
    train_losses = np.load('results/logs/train_losses.npy')
    train_recon_losses = np.load('results/logs/train_recon_losses.npy')
    train_kl_losses = np.load('results/logs/train_kl_losses.npy')
    test_recon_losses = np.load('results/logs/test_recon_losses.npy')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VAE模型训练过程分析', fontsize=18, fontweight='bold', y=0.98)
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0, 0].plot(epochs, train_losses, linewidth=2.5, color='#2E86AB', label='总损失')
    axes[0, 0].fill_between(epochs, train_losses, alpha=0.2, color='#2E86AB')
    axes[0, 0].set_xlabel('训练轮数', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('损失值', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('总损失变化趋势', fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].legend(loc='upper right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    axes[0, 1].plot(epochs, train_recon_losses, linewidth=2.5, color='#A23B72', label='训练集重构损失')
    axes[0, 1].plot(epochs, test_recon_losses, linewidth=2.5, color='#F18F01', label='测试集重构损失', linestyle='--')
    axes[0, 1].set_xlabel('训练轮数', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('损失值', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('重构损失对比', fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].legend(loc='upper right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')


    axes[1, 0].plot(epochs, train_kl_losses, linewidth=2.5, color='#C73E1D', label='KL散度')
    axes[1, 0].fill_between(epochs, train_kl_losses, alpha=0.2, color='#C73E1D')
    axes[1, 0].set_xlabel('训练轮数', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('损失值', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('KL散度变化趋势', fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].legend(loc='upper right', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
  
    axes[1, 1].plot(epochs, train_recon_losses, linewidth=2.5, color='#A23B72', label='重构损失')
    axes[1, 1].plot(epochs, train_kl_losses, linewidth=2.5, color='#C73E1D', label='KL散度')
    axes[1, 1].plot(epochs, train_losses, linewidth=3, color='#2E86AB', label='总损失', alpha=0.7)
    axes[1, 1].set_xlabel('训练轮数', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('损失值', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('损失分量分解', fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].legend(loc='upper right', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/images/training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ 损失曲线已保存至 results/images/training_curves.png")

def evaluate_reconstruction(model, device, target_digit=0, num_samples=6):
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
        
        # 计算MSE（用于显示）
        mse = np.mean((original - reconstructed) ** 2, axis=(1, 2))
        
        # 可视化对比
        fig, axes = plt.subplots(3, num_samples, figsize=(16, 10))
        fig.suptitle(f'VAE重构效果深度分析 (数字{target_digit})', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        for i in range(num_samples):
            axes[0, i].imshow(original[i], cmap='gray')
            axes[0, i].set_title(f'原始图像 #{i+1}', fontsize=11, fontweight='bold', pad=5)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(reconstructed[i], cmap='gray')
            axes[1, i].set_title(f'重构图像 #{i+1}\nMSE: {mse[i]:.4f}', 
                                 fontsize=11, fontweight='bold', pad=5)
            axes[1, i].axis('off')
            
            diff = np.abs(original[i] - reconstructed[i])
            im = axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'差异热力图 #{i+1}', fontsize=11, fontweight='bold', pad=5)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/images/reconstruction_digit_{target_digit}.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"✓ 重构对比图已保存至 results/images/reconstruction_digit_{target_digit}.png")

def generate_samples(model, device, latent_dim=20, num_samples=16):
    """随机采样生成"""
    print("\n从潜在空间随机采样生成新图像...")
    
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样（4x4网格）
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # 生成图像
        generated = model.decode(z)
        generated_images = generated.view(-1, 28, 28).cpu().numpy()
        
        # 可视化（4x4网格）
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('从潜在空间随机采样生成的数字0图像', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        for i in range(num_samples):
            axes[i].imshow(generated_images[i], cmap='gray')
            axes[i].axis('off')
            # 添加样本编号
            axes[i].text(0.5, -0.15, f'样本 {i+1}', 
                         transform=axes[i].transAxes, 
                         fontsize=10, fontweight='bold', 
                         ha='center', va='top')
        
        plt.tight_layout()
        plt.savefig('results/images/generated_samples_basic.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("✓ 生成图像已保存至 results/images/generated_samples_basic.png")

def generate_conditional_samples(cvae_model, device, num_samples_per_digit=6):
    """条件生成展示"""
    print("\n使用条件VAE生成0-9数字图像...")
    
    cvae_model.eval()
    # 调整布局：10行数字，每行6个样本
    fig, axes = plt.subplots(10, num_samples_per_digit, figsize=(14, 18))
    fig.suptitle('条件VAE生成的0-9数字图像', 
                 fontsize=18, fontweight='bold', y=0.98)
    
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
                img = generated_images[i].cpu().squeeze()
                axes[digit, i].imshow(img, cmap='gray')
                axes[digit, i].axis('off')
                
                # 只在每行第一个图像添加标签
                if i == 0:
                    axes[digit, i].text(-0.2, 0.5, f'数字 {digit}', 
                                        transform=axes[digit, i].transAxes,
                                        fontsize=12, fontweight='bold',
                                        ha='right', va='center')
        
        # 添加整体说明
        fig.text(0.5, 0.02, '每个数字从独立采样潜在变量生成', 
                 ha='center', fontsize=12, fontweight='bold', style='italic')
    
    plt.tight_layout()
    plt.savefig('results/images/generated_samples_cvae.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ 条件生成图像已保存至 results/images/generated_samples_cvae.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs('results/images', exist_ok=True)
    
    plot_training_curves()
    
    print("\n加载基础VAE模型...")
    vae_basic = VAEBasic(latent_dim=20).to(device)
    vae_basic.load_state_dict(torch.load('results/vae_basic_model.pth', map_location=device, weights_only=True))
    vae_basic.eval()
    
    evaluate_reconstruction(vae_basic, device, target_digit=0, num_samples=6)
    
    generate_samples(vae_basic, device, latent_dim=20, num_samples=16)
    
    print("\n加载条件VAE模型...")
    cvae_model = CVAEAdvanced(latent_dim=20).to(device)
    cvae_model.load_state_dict(torch.load('results/cvae_advanced_model.pth', map_location=device, weights_only=True))
    cvae_model.eval()
    
    generate_conditional_samples(cvae_model, device, num_samples_per_digit=6)
    
    print("\n所有评估完成！请检查 results/images/ 目录中的结果。")

if __name__ == '__main__':
    main()