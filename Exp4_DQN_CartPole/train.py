import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from dqn_agent import DQNAgent
import config

def plot_training_curve(episode_rewards, losses, save_path, success_threshold=195):
    """绘制训练曲线"""
    plt.style.use("seaborn-v0_8-darkgrid")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 奖励曲线
    ax1.plot(episode_rewards, label='Episode Reward', color="#1f77b4", alpha=0.9)
    
    # 计算滑动平均
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards, 
            np.ones(window)/window, 
            mode='valid'
        )
        ax1.plot(
            range(window-1, len(episode_rewards)),
            moving_avg,
            label=f'{window}-Episode Moving Avg',
            linestyle='--',
            color="#ff7f0e"
        )
    ax1.axhline(success_threshold, color="#2ca02c", linestyle=':', linewidth=1.2, label=f'Success {success_threshold}')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    if losses:
        # 过滤None值
        valid_losses = [l for l in losses if l is not None]
        ax2.plot(valid_losses, label='Training Loss', alpha=0.7, color="#9467bd")
        
        # 平滑损失曲线
        if len(valid_losses) >= window:
            smoothed_loss = np.convolve(
                valid_losses, 
                np.ones(window)/window, 
                mode='valid'
            )
            ax2.plot(
                range(window-1, len(valid_losses)),
                smoothed_loss,
                label=f'{window}-Step Smoothed',
                linestyle='--',
                color="#d62728"
            )
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存至: {save_path}")

def train_dqn():
    """主训练循环"""
    # 配置
    cfg = config.DQNConfig()
    
    # 创建环境
    env = gym.make(cfg.env_name)
    
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    
    # 创建智能体
    agent = DQNAgent(cfg)
    
    # 记录指标
    episode_rewards = []
    losses = []
    best_avg_reward = -float('inf')
    total_steps = 0
    
    print("="*60)
    print("开始DQN训练")
    print(f"设备: {cfg.device}")
    print(f"环境: {cfg.env_name}")
    print(f"状态维度: {cfg.state_dim}, 动作维度: {cfg.action_dim}")
    print("="*60)
    
    # 训练循环
    for episode in range(1, cfg.max_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []
        
        for step in range(cfg.max_steps):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.train()
            if loss is not None:
                losses.append(loss)
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            total_steps += 1
            
            if done:
                break
        
        # 软更新目标网络
        agent.soft_update_target_network()
        
        # 硬更新目标网络（可选）
        if episode % cfg.target_update_freq == 0:
            agent.hard_update_target_network()
        
        # 记录奖励
        episode_rewards.append(total_reward)
        
        # 计算最近100回合平均奖励
        recent_avg_reward = np.mean(episode_rewards[-cfg.success_window:])
        
        # 打印日志
        if episode % cfg.log_interval == 0:
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:5.1f} | "
                  f"Recent Avg: {recent_avg_reward:6.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.memory):5d}")

        # 保存最佳模型
        if recent_avg_reward > best_avg_reward:
            best_avg_reward = recent_avg_reward
            agent.save_model(cfg.best_model_path)
        
        # 检查是否达到成功标准
        if recent_avg_reward >= cfg.success_threshold and len(episode_rewards) >= cfg.success_window:
            print("\n" + "="*60)
            print(f"训练成功！在第 {episode} 回合达到成功标准")
            print(f"最近 {cfg.success_window} 回合平均奖励: {recent_avg_reward:.2f}")
            print("="*60)
            break
    
    # 最终评估
    print("\n开始最终评估...")
    best_path = Path(cfg.best_model_path)
    if best_path.exists():
        agent.load_model(best_path)
        print(f"已加载最佳模型进行评估: {best_path}")
    else:
        print(f"未找到最佳模型 {best_path}，使用当前权重评估")

    evaluate_agent(agent, env, episodes=cfg.eval_episodes)
    
    # 绘制训练曲线
    plot_training_curve(episode_rewards, losses, cfg.plot_save_path, success_threshold=cfg.success_threshold)
    
    # 保存最终模型
    agent.save_model(cfg.model_save_path)
    
    env.close()
    
    return episode_rewards, losses

def evaluate_agent(agent, env, episodes=10):
    """评估智能体性能"""
    total_rewards = []
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"评估回合 {episode}: 奖励 = {total_reward:.1f}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\n评估完成 - 平均奖励: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    # 运行训练
    train_dqn()