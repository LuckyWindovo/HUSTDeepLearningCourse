from experiment import run_experiment, CONFIG

configs = [
    {'name': 'baseline', 'lr': 0.01, 'hidden': [128, 64]},
    {'name': 'small_lr', 'lr': 0.001, 'hidden': [128, 64]},
    {'name': 'wide_net', 'lr': 0.01, 'hidden': [256, 128]},
]

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"运行配置: {cfg['name']}")
    print(f"{'='*60}")
    
    CONFIG['learning_rate'] = cfg['lr']
    CONFIG['hidden_sizes'] = cfg['hidden']
    CONFIG['results_dir'] = f"Exp1_MNIST/results_{cfg['name']}"
    
    # 运行实验
    run_experiment()