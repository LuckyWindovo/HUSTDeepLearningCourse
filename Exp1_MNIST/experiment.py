import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.mlp import DebugMLP
from utils.viz import plot_training_history, plot_weights_distribution

def load_data(batch_size=64, data_dir='./data'):
    """加载并预处理 MNIST 数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    def flatten_tensor(x):
        return x.view(-1)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(flatten_tensor) 
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, 
        download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform
    )   

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader

def print_debug_info(model, loss_value):
    """打印详细的调试信息"""
    print("\n" + "="*80)
    print("前向传播详细信息")
    print("="*80)
    
    for info in model.debug_info['forward']:
        print(f"\n[层 {info['layer_idx']:2d}] {info['layer_type']:<20s}")
        print(f"  输入形状:  {info['input_shape']}")
        print(f"  输出形状:  {info['output_shape']}")
        print(f"  输出样本前5个值: {info['output_sample'][:5].cpu().numpy()}")
        
        if info['layer_type'] == 'Linear':
            print(f"  权重形状:  {info['weights_shape']}")
            print(f"  权重统计:  均值={info['weights_mean']:.6f}, 标准差={info['weights_std']:.6f}")
            print(f"  偏置形状:  {info['bias_shape']}")
            print(f"  偏置均值:  {info['bias_mean']:.6f}")
    
    print(f"\n[损失值] CrossEntropy Loss: {loss_value:.6f}")
    
    print("\n" + "="*80)
    print("反向传播梯度信息")
    print("="*80)
    
    if len(model.debug_info['backward']) == 0:
        print("警告: 未捕获到梯度信息！请确保调用了loss.backward()")
    
    for info in model.debug_info['backward']:
        print(f"\n[参数 {info['param_name']}]")
        print(f"  梯度形状:  {info['grad_shape']}")
        print(f"  梯度均值:  {info['grad_mean']:.8f}")
        print(f"  梯度最大值: {info['grad_max']:.8f}")
        print(f"  梯度范数:  {info['grad_norm']:.8f}")

def train_step_debug(model, data, target, loss_fn, optimizer, device):
    """单次训练步骤（带调试信息）"""
    model.train()
    
    data, target = data.to(device), target.to(device)
    
    # 前向传播
    output = model(data)
    loss = loss_fn(output, target)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 参数更新
    optimizer.step()
    
    return loss, output

def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01, 
                       debug_first_batch=True, save_dir='./results'):
    """完整训练与评估"""
    # 创建结果保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cpu')
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print("\n" + "="*80)
    print("开始训练")
    print("="*80)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 仅在第一个epoch的第一个batch展示详细过程
            if debug_first_batch and batch_idx == 0 and epoch == 0:
                loss, output = train_step_debug(model, data, target, loss_fn, optimizer, device)
                print_debug_info(model, loss.item())
            else:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        
        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / len(train_loader.dataset)
        
        # 测试集评估
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / len(test_loader.dataset)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 打印进度
        print(f"\nEpoch {epoch+1:2d}/{epochs}:")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:6.2f}%")
        print(f"  测试损失: {test_loss:.4f} | 测试准确率: {test_acc:6.2f}%")
        
        # 每5个epoch保存权重分布图
        if (epoch + 1) % 5 == 0:
            plot_weights_distribution(
                model, 
                save_path=os.path.join(save_dir, f'weights_epoch_{epoch+1}.png')
            )
    
    print("\n" + "="*80)
    print("训练完成")
    print("="*80)
    
    return history, device

def run_experiment():
    """运行完整实验"""
    # 配置参数
    CONFIG = {
        'batch_size': 64,
        'epochs': 10,
        'learning_rate': 0.01,
        'hidden_sizes': [128, 64],
        'activation': 'relu',
        'data_dir': './data',
        'results_dir': './results'
    }
    
    # 打印配置
    print("实验配置:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # 加载数据
    print("\n加载数据集...")
    train_loader, test_loader = load_data(
        batch_size=CONFIG['batch_size'], 
        data_dir=CONFIG['data_dir']
    )
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 创建模型
    print("\n初始化模型...")
    model = DebugMLP(
        input_size=784,
        hidden_sizes=CONFIG['hidden_sizes'],
        num_classes=10,
        activation=CONFIG['activation']
    )
    print(f"模型结构:\n{model}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 训练模型
    history, device = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        debug_first_batch=True,
        save_dir=CONFIG['results_dir']
    )
    
    # 可视化结果
    print("\n生成可视化图表...")
    plot_training_history(
        history,
        save_path=os.path.join(CONFIG['results_dir'], 'training_curves.png'),
        show_plot=True
    )
    
    # 保存最终权重分布
    plot_weights_distribution(
        model,
        save_path=os.path.join(CONFIG['results_dir'], 'weights_final.png')
    )
    
    # 保存训练历史
    import json
    with open(os.path.join(CONFIG['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n所有结果已保存到: {CONFIG['results_dir']}")
    
    model.eval()
    with torch.no_grad():
        # 在测试集上评估
        test_correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
        
        final_test_acc = 100. * test_correct / len(test_loader.dataset)
        print(f"最终测试集准确率: {final_test_acc:.2f}%")

if __name__ == '__main__':
    run_experiment()