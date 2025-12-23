import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
import logging
import time
from pathlib import Path

# 禁用警告
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from config import Config
from model import get_model
from utils import *

# 删除这行：torch.backends.cudnn.benchmark = True
# 移到main函数内部

def run_single_experiment(exp_name, exp_cfg, cfg):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Starting {exp_name}")
    print(f"Config: {exp_cfg}")
    print(f"{'='*60}")
    
    # 创建实验目录
    exp_dir = Path(cfg.RESULTS_PATH) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子（仅在需要复现性时）
    if cfg.DETERMINISTIC:
        torch.manual_seed(cfg.SEED)
    
    # 加载数据
    print("\n[1/6] Loading data...")
    data_start = time.time()
    trainloader, testloader = get_cifar10_dataloader(
        root=cfg.DATA_ROOT, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS
    )
    classes = getattr(trainloader.dataset, 'classes', default_cifar10_classes())
    data_time = time.time() - data_start
    print(f"✓ Data loaded in {data_time:.1f}s: {len(trainloader)} train batches, {len(testloader)} test batches")
    
    # 创建模型
    print("\n[2/6] Building model...")
    model_start = time.time()
    model = get_model(
        activation=exp_cfg['activation'],
        use_dropout=exp_cfg['use_dropout'],
        dropout_rate=cfg.DROPOUT_RATE,
        use_batchnorm=exp_cfg['use_batchnorm'],
        kernel_size=exp_cfg['kernel_size']
    )
    model = model.to(cfg.DEVICE)
    model_time = time.time() - model_start
    print(f"✓ Model created in {model_time:.1f}s: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, 
                          momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=str(exp_dir / 'logs'))
    
    # 训练循环
    print("\n[3/6] Starting training...")
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = 0.0
    
    # 记录开始时间
    start_time = time.time()
    
    for epoch in range(cfg.EPOCHS):
        # 训练
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, cfg.DEVICE, epoch, writer, 
            use_amp=cfg.USE_MIXED_PRECISION
        )
        
        # 测试
        test_loss, test_acc = test_epoch(
            model, testloader, criterion, cfg.DEVICE, epoch, writer
        )
        
        # 记录结果
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, exp_dir / 'best_model.pth')
        
        # 打印epoch总结
        print(f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
              f"Best: {best_acc:.2f}%")
    
    # 计算总训练时间
    total_time = time.time() - start_time

    # 按类别统计准确率
    print("\n[4/6] Per-class analysis...")
    per_class_acc, overall_acc = calculate_per_class_accuracy(model, testloader, cfg.DEVICE, classes, save_dir=exp_dir)

    # 可视化预测结果
    print("\n[5/6] Visualizing predictions...")
    visualize_predictions(model, testloader, cfg.DEVICE, classes, num_samples=16, save_dir=exp_dir)

    # 可视化训练曲线
    print("\n[6/6] Generating plots...")
    plot_results(train_losses, test_losses, train_accs, test_accs, exp_dir)

    # 保存最终结果
    print("\nSaving results...")
    results = {
        "config": exp_cfg,
        "final_train_acc": train_accs[-1],
        "final_test_acc": test_accs[-1],
        "best_test_acc": best_acc,
        "total_time_minutes": total_time / 60,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "per_class_accuracy": per_class_acc,
        "overall_test_acc": overall_acc,
        "classes": classes
    }
    
    # 保存详细结果
    import json
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存简要结果
    with open(exp_dir / 'summary.txt', 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Config: {exp_cfg}\n\n")
        f.write(f"Final Train Accuracy: {train_accs[-1]:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_accs[-1]:.2f}%\n")
        f.write(f"Best Test Accuracy: {best_acc:.2f}%\n")
        f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
    
    writer.close()
    
    print(f"\n✓ Experiment completed in {total_time/60:.2f} minutes!")
    print(f"✓ Results saved to: {exp_dir}")
    
    return results

def main():
    cfg = Config()
    
    # 在这里启用cuDNN优化（只执行一次）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("🚀 CUDA enabled with cuDNN auto-tuning")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Workers: {cfg.NUM_WORKERS}")
    
    # 创建结果目录
    os.makedirs(cfg.RESULTS_PATH, exist_ok=True)
    os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)
    
    # 检查是否需要运行所有实验
    run_all = input("Run all 6 experiments? (y/n, default=y): ").strip().lower() != 'n'
    
    if run_all:
        # 运行所有实验
        all_results = {}
        for exp_name, exp_cfg in cfg.EXPERIMENT_CONFIGS.items():
            try:
                result = run_single_experiment(exp_name, exp_cfg, cfg)
                all_results[exp_name] = result
            except Exception as e:
                print(f"❌ Error in {exp_name}: {e}")
        
        # 生成汇总报告
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT...")
        print("="*60)
        generate_comparison_report(all_results, cfg.RESULTS_PATH)
    else:
        # 只运行基准实验
        print("Running baseline experiment only...")
        run_single_experiment("Exp1_Baseline_ReLU", cfg.EXPERIMENT_CONFIGS["Exp1_Baseline_ReLU"], cfg)

if __name__ == '__main__':
    main()