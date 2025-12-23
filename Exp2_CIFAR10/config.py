import torch
import sys  # 添加版本检测

class Config:
    # 性能配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加速优化
    USE_TORCH_COMPILE = False  # Python 3.12+不支持，直接禁用
    USE_MIXED_PRECISION = True  # 混合精度训练（GPU有效）
    DETERMINISTIC = False  # 牺牲复现性换取速度
    
    # 数据配置
    DATA_ROOT = './data'
    BATCH_SIZE = 256  # GPU可承受更大batch
    NUM_WORKERS = 4   # 多进程加载数据
    PIN_MEMORY = True  # 加速数据传输到GPU
    PREFETCH_FACTOR = 2 
    
    # 模型配置
    # CIFAR-10 with SGD通常需要较大学习率，0.1可显著提升收敛速度与最终精度
    LEARNING_RATE = 0.1
    EPOCHS = 30
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # 正则化
    USE_DROPOUT = True
    DROPOUT_RATE = 0.3
    
    USE_BATCHNORM = True
    
    # 路径配置
    MODEL_SAVE_PATH = './models'
    RESULTS_PATH = './results'
    
    # 随机种子
    SEED = 42

    # 实验配置矩阵
    EXPERIMENT_CONFIGS = {
        "Exp1_Baseline_ReLU": {
            "activation": "relu",
            "use_batchnorm": True,
            "use_dropout": True,
            "kernel_size": 3,
        },
        "Exp2_LeakyReLU_Baseline": {
            "activation": "leakyrelu",
            "use_batchnorm": True,
            "use_dropout": True,
            "kernel_size": 3,
        },
        "Exp3_ReLU_NoBatchNorm": {
            "activation": "relu",
            "use_batchnorm": False,
            "use_dropout": True,
            "kernel_size": 3,
        },
        "Exp4_ReLU_NoDropout": {
            "activation": "relu",
            "use_batchnorm": True,
            "use_dropout": False,
            "kernel_size": 3,
        },
        "Exp5_LeakyReLU_NoBatchNorm": {
            "activation": "leakyrelu",
            "use_batchnorm": False,
            "use_dropout": True,
            "kernel_size": 3,
        },
        "Exp6_LeakyReLU_NoDropout": {
            "activation": "leakyrelu",
            "use_batchnorm": True,
            "use_dropout": False,
            "kernel_size": 3,
        }
    }