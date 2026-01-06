import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """深度Q网络（MLP）"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def test_model():
    """测试网络前向传播"""
    import config
    state = torch.randn(1, config.DQNConfig.state_dim)
    model = DQNNetwork(
        config.DQNConfig.state_dim,
        config.DQNConfig.action_dim,
        config.DQNConfig.hidden_dims
    )
    q_values = model(state)
    print(f"输入状态形状: {state.shape}")
    print(f"输出Q值形状: {q_values.shape}")
    print(f"Q值: {q_values}")

if __name__ == "__main__":
    test_model()