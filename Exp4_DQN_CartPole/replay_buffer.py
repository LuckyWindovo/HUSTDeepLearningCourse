import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience', 
                                     ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        experiences = random.sample(self.buffer, batch_size)
        
        # 转换为numpy数组
        states = np.vstack([e.state for e in experiences])
        actions = np.vstack([e.action for e in experiences])
        rewards = np.vstack([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([e.done for e in experiences]).astype(np.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def test_buffer():
    """测试缓冲区功能"""
    buffer = ReplayBuffer(100)
    
    # 填充一些假数据
    for i in range(10):
        buffer.push(
            state=[i, i+1, i+2, i+3],
            action=0,
            reward=1.0,
            next_state=[i+1, i+2, i+3, i+4],
            done=False
        )
    
    print(f"缓冲区当前大小: {len(buffer)}")
    
    # 采样测试
    if len(buffer) >= 5:
        states, actions, rewards, next_states, dones = buffer.sample(5)
        print(f"采样状态形状: {states.shape}")
        print(f"采样动作: {actions.flatten()}")

if __name__ == "__main__":
    test_buffer()