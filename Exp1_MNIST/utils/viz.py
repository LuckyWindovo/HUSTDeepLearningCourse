import matplotlib.pyplot as plt
import numpy as np

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def plot_training_history(history, save_path='training_curve.png', show_plot=False):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['test_loss'], label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['test_acc'], label='Test Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def plot_weights_distribution(model, save_path='weights_distribution.png'):
    """可视化权重分布"""
    weights = []
    biases = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.extend(param.detach().cpu().numpy().flatten())
        elif 'bias' in name:
            biases.extend(param.detach().cpu().numpy().flatten())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('权重分布', fontsize=14)
    ax1.set_xlabel('权重值')
    ax1.set_ylabel('频数')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(biases, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('偏置分布', fontsize=14)
    ax2.set_xlabel('偏置值')
    ax2.set_ylabel('频数')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()