from experiment import run_experiment, CONFIG
import copy

configs = [
    # 原有实验
    {'name': 'baseline', 'lr': 0.01, 'hidden': [128, 64], 'activation': 'relu'},
    {'name': 'small_lr', 'lr': 0.001, 'hidden': [128, 64], 'activation': 'relu'},
    {'name': 'wide_net', 'lr': 0.01, 'hidden': [256, 128], 'activation': 'relu'},
    
    # 新增激活函数实验（保持其他参数与baseline一致）
    {'name': 'sigmoid', 'lr': 0.01, 'hidden': [128, 64], 'activation': 'sigmoid'},
    {'name': 'tanh', 'lr': 0.01, 'hidden': [128, 64], 'activation': 'tanh'},
]

for cfg in configs:
    experiment_config = copy.deepcopy(CONFIG)
    experiment_config['learning_rate'] = cfg['lr']
    experiment_config['hidden_sizes'] = cfg['hidden']
    experiment_config['activation'] = cfg['activation']
    experiment_config['results_dir'] = f"Exp1_MNIST/results_{cfg['name']}"
    
    # 运行实验
    run_experiment(config=experiment_config)