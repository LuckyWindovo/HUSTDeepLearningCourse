import gymnasium as gym
from dqn_agent import DQNAgent
import config

def demonstrate_trained_agent(model_path="best_dqn_model.pth", episodes=5):
    """演示训练好的智能体"""
    cfg = config.DQNConfig()
    env = gym.make(cfg.env_name, render_mode="human")
    
    # 创建并加载智能体
    agent = DQNAgent(cfg)
    try:
        agent.load_model(model_path)
    except FileNotFoundError:
        print(f"模型文件 {model_path} 未找到，请确保已训练并保存模型")
        return
    
    print("\n" + "="*60)
    print("开始演示训练好的智能体")
    print("="*60)
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"演示回合 {episode}: 总奖励 = {total_reward}")
    
    env.close()
    print("\n演示完成！")

if __name__ == "__main__":
    demonstrate_trained_agent()