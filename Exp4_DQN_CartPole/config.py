import torch

class DQNConfig:
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 环境参数
    env_name = "CartPole-v1"
    state_dim = 4  # CartPole状态空间维度
    action_dim = 2  # CartPole动作空间维度
    
    # 网络结构
    hidden_dims = [128, 128]  # 隐藏层维度
    
    # 训练参数
    lr = 1e-3  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon_start = 1.0  # 初始探索率
    epsilon_end = 0.05  # 最终探索率
    epsilon_decay = 0.999  # 按步衰减系数（越接近1衰减越慢）
    min_buffer_size = 1000  # 开始训练前至少收集的样本数
    gradient_clip = 5.0  # 梯度裁剪防止梯度爆炸
    target_update_freq = 10  # 硬更新频率（每10个episode）
    tau = 0.005  # 软更新系数
    
    # 经验回放
    buffer_size = 10000  # 缓冲区大小
    batch_size = 64  # 批次大小
    
    # 训练控制
    max_episodes = 500  # 最大训练回合数
    max_steps = 500  # 每回合最大步数
    success_threshold = 195  # 成功标准（100回合平均奖励）
    success_window = 100  # 滑动窗口大小
    eval_episodes = 30  # 训练结束评估回合数（提高统计稳定性）
    
    # 随机种子
    seed = 42
    
    # 日志和模型保存
    model_save_path = "artifacts/dqn_model.pth"
    best_model_path = "artifacts/best_dqn_model.pth"
    plot_save_path = "artifacts/training_curves.png"
    log_interval = 10  # 每10回合打印一次日志