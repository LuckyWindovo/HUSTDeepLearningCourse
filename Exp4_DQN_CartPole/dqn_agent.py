import torch
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from model import DQNNetwork
from replay_buffer import ReplayBuffer
import config

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 主网络和目标网络
        self.q_network = DQNNetwork(
            config.state_dim, 
            config.action_dim, 
            config.hidden_dims
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            config.state_dim, 
            config.action_dim, 
            config.hidden_dims
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(config.buffer_size)
        
        # 探索率
        self.epsilon = config.epsilon_start
        
        # 训练步数
        self.train_step = 0
    
    def select_action(self, state, training=True):
        """ε-greedy策略选择动作"""
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.randrange(self.config.action_dim)
        else:
            # 利用Q网络
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update_epsilon(self):
        """按步衰减探索率"""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
    
    def soft_update_target_network(self):
        """软更新目标网络"""
        for target_param, local_param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * local_param.data + 
                (1.0 - self.config.tau) * target_param.data
            )
    
    def hard_update_target_network(self):
        """硬更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self):
        """训练主网络"""
        if len(self.memory) < max(self.config.batch_size, self.config.min_buffer_size):
            return None
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值 (主网络)
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 目标Q值 (目标网络)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.config.gamma * max_next_q_values * (1 - dones))
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        
        self.train_step += 1
        self.update_epsilon()
        
        return loss.item()
    
    def save_model(self, path):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        print(f"模型已从 {path} 加载")

def test_agent():
    """测试智能体"""
    cfg = config.DQNConfig()
    agent = DQNAgent(cfg)
    
    print(f"网络结构:\n{agent.q_network}")
    print(f"\n设备: {agent.device}")
    print(f"初始探索率: {agent.epsilon}")
    
    # 测试动作选择
    test_state = np.random.randn(cfg.state_dim)
    action = agent.select_action(test_state, training=True)
    print(f"\n测试状态: {test_state}")
    print(f"选择动作: {action}")

if __name__ == "__main__":
    test_agent()