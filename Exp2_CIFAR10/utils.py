import os
import math
import json
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from config import Config

def get_cifar10_dataloader(
    root='./data',
    batch_size=128,
    num_workers=0,
    download=True,
    prefetch_factor=2,
    pin_memory=None,
    persistent_workers=None,
):
    """加载CIFAR-10数据集（优化版，保持worker常驻以避免反复启动开销）"""

    # 检查数据是否已存在
    data_exists = os.path.exists(os.path.join(root, 'cifar-10-batches-py'))
    if data_exists and download:
        print(f"✓ CIFAR-10 dataset found at {root}")
        download = False

    # 数据增强和标准化
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # CUDA下开启pin_memory和持久化worker以减少每轮启动开销
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    keep_workers = (num_workers > 0) if persistent_workers is None else persistent_workers

    # 优化DataLoader参数
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=download, transform=transform_train)
    trainloader_kwargs = dict(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,  # 丢弃不完整的batch以保持稳定
    )
    if num_workers > 0:
        trainloader_kwargs.update(
            persistent_workers=keep_workers,
            prefetch_factor=prefetch_factor,
        )
    trainloader = torch.utils.data.DataLoader(**trainloader_kwargs)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=download, transform=transform_test)
    testloader_kwargs = dict(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    if num_workers > 0:
        testloader_kwargs.update(
            persistent_workers=keep_workers,
            prefetch_factor=prefetch_factor,
        )
    testloader = torch.utils.data.DataLoader(**testloader_kwargs)

    return trainloader, testloader

def train_epoch(model, trainloader, criterion, optimizer, device, epoch, writer, use_amp=True):
    """训练一个epoch（支持混合精度）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 混合精度上下文管理器
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and hasattr(torch.cuda, 'amp'))
    dtype = torch.bfloat16 if use_amp else torch.float32
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
        if use_amp and not hasattr(torch.cuda, 'amp'):  # CPU BF16
            inputs = inputs.to(dtype)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        if use_amp and hasattr(torch.cuda, 'amp'):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # 记录到TensorBoard
        if batch_idx % 20 == 0:
            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy', 100.*correct/total, global_step)
    
    return running_loss / len(trainloader), 100. * correct / total

def test_epoch(model, testloader, criterion, device, epoch, writer):
    """测试一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(testloader, desc=f'Epoch {epoch+1} [Test]', leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            if batch_idx % 20 == 0:
                global_step = epoch * len(testloader) + batch_idx
                writer.add_scalar('Test/Loss', loss.item(), global_step)
                writer.add_scalar('Test/Accuracy', 100.*correct/total, global_step)
    
    return running_loss / len(testloader), 100. * correct / total


def default_cifar10_classes():
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def calculate_per_class_accuracy(model, testloader, device, classes, save_dir=None):
    """按类别统计准确率并保存结果"""
    model.eval()
    num_classes = len(classes)
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    overall_correct = 0
    overall_total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            overall_total += targets.size(0)
            overall_correct += predicted.eq(targets).sum().item()

            for idx in range(num_classes):
                mask = targets == idx
                class_total[idx] += mask.sum().item()
                if class_total[idx] == 0:
                    continue
                class_correct[idx] += (predicted[mask] == idx).sum().item()

    per_class_acc = {}
    for name, corr, tot in zip(classes, class_correct, class_total):
        acc = 100.0 * corr / tot if tot > 0 else 0.0
        per_class_acc[name] = {
            "accuracy": acc,
            "correct": corr,
            "total": tot,
        }

    overall_acc = 100.0 * overall_correct / overall_total if overall_total > 0 else 0.0

    lines = [
        "Class Accuracy Report",
        "====================="
    ]
    for name in classes:
        stats = per_class_acc[name]
        lines.append(f"{name:<11}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    lines.append(f"Overall     : {overall_acc:.1f}% ({overall_correct}/{overall_total})")
    print('\n'.join(lines))

    target_dir = Path(save_dir) if save_dir else Path(Config.RESULTS_PATH)
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / 'per_class_accuracy.json', 'w') as f:
        json.dump({"per_class": per_class_acc, "overall": overall_acc}, f, indent=2)

    return per_class_acc, overall_acc


def visualize_predictions(model, testloader, device, classes, num_samples=16, save_dir=None):
    """随机可视化预测结果并保存网格图"""
    model.eval()
    dataset = testloader.dataset
    total = len(dataset)
    num_samples = min(num_samples, total)

    # 随机采样不重复索引
    indices = torch.randperm(total)[:num_samples]
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    batch = torch.stack(images).to(device)
    labels_tensor = torch.tensor(labels, device=device)

    with torch.no_grad():
        outputs = model(batch)
        preds = outputs.argmax(1)

    correct_mask = preds.eq(labels_tensor)
    sample_acc = 100.0 * correct_mask.sum().item() / num_samples if num_samples > 0 else 0.0

    # 反归一化便于可视化
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    grid = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(grid * grid):
        ax = axes[i]
        ax.axis('off')
        if i >= num_samples:
            continue
        img = images[i].cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        true_label = classes[labels[i]] if labels[i] < len(classes) else str(labels[i])
        pred_label = classes[preds[i]] if preds[i] < len(classes) else str(preds[i])
        color = 'green' if correct_mask[i].item() else 'red'
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"True: {true_label} | Pred: {pred_label}", color=color, fontsize=9)

    fig.suptitle(f"Model Predictions on CIFAR-10 Test Set (Acc: {sample_acc:.1f}%)", fontsize=14)
    plt.tight_layout()

    target_dir = Path(save_dir) if save_dir else Path(Config.RESULTS_PATH)
    target_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_dir / 'predictions_grid.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_results(train_losses, test_losses, train_accs, test_accs, save_path):
    """绘制训练和测试曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Training and Testing Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Training and Testing Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像释放内存

def save_model(model, path):
    """保存模型（仅保存权重以减小文件大小）"""
    torch.save(model.state_dict(), path)
    print(f"✓ Model saved to {path}")

def generate_comparison_report(all_results, results_path):
    """生成实验对比报告"""
    results_path = Path(results_path)
    
    # 创建对比表格
    report_lines = []
    report_lines.append("# CIFAR-10 CNN Experiments Comparison Report\n")
    report_lines.append("| Experiment | Activation | BatchNorm | Dropout | Final Train Acc | Final Test Acc | Best Test Acc | Time (min) |")
    report_lines.append("|------------|------------|-----------|---------|-----------------|----------------|---------------|------------|")
    
    # 收集结果
    summary_data = []
    for exp_name, result in all_results.items():
        cfg = result['config']
        report_lines.append(
            f"| {exp_name} | "
            f"{cfg['activation']} | "
            f"{'Yes' if cfg['use_batchnorm'] else 'No'} | "
            f"{'Yes' if cfg['use_dropout'] else 'No'} | "
            f"{result['final_train_acc']:.2f}% | "
            f"{result['final_test_acc']:.2f}% | "
            f"{result['best_test_acc']:.2f}% | "
            f"{result['total_time_minutes']:.1f} |"
        )
        summary_data.append({
            'exp_name': exp_name,
            'best_acc': result['best_test_acc'],
            'final_acc': result['final_test_acc']
        })
    
    # 找出最佳实验
    best_exp = max(summary_data, key=lambda x: x['best_acc'])
    report_lines.append(f"\n## Summary")
    report_lines.append(f"- **Best Experiment**: {best_exp['exp_name']} ({best_exp['best_acc']:.2f}%)")
    report_lines.append(f"- **Total Experiments**: {len(all_results)}")
    report_lines.append(f"- **Average Best Accuracy**: {np.mean([r['best_test_acc'] for r in all_results.values()]):.2f}%")
    
    # 保存报告
    report_path = results_path / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # 绘制对比图
    plot_comparison_chart(all_results, results_path)
    
    print(f"\n✓ Comparison report saved to: {report_path}")

def plot_comparison_chart(all_results, save_path):
    """绘制实验对比柱状图"""
    import matplotlib.pyplot as plt
    
    experiments = list(all_results.keys())
    best_accs = [all_results[exp]['best_test_acc'] for exp in experiments]
    final_accs = [all_results[exp]['final_test_acc'] for exp in experiments]
    
    x = range(len(experiments))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width/2 for i in x], best_accs, width, label='Best Accuracy', alpha=0.8)
    ax.bar([i + width/2 for i in x], final_accs, width, label='Final Accuracy', alpha=0.8)
    
    ax.set_xlabel('Experiments', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-10 CNN Experiments Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([exp.replace('_', '\n') for exp in experiments], rotation=0, ha='center')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / 'comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()