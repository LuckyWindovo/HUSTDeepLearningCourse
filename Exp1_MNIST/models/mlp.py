import torch
import torch.nn as nn
import torch.nn.functional as F

class DebugMLP(nn.Module):
    """可调试的多层感知机模型"""
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, activation='sigmoid'):
        super(DebugMLP, self).__init__()
        
        self.activation_name = activation
        self.hidden_sizes = hidden_sizes
        
        # 构建层
        layers = []
        layer_infos = []  # 存储每层信息用于调试
        prev_size = input_size
        
        # 输入层到隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            linear = nn.Linear(prev_size, hidden_size)
            layers.append(linear)
            layer_infos.append({'type': 'Linear', 'module': linear})
            
            if activation == 'relu':
                act = nn.ReLU()
            elif activation == 'sigmoid':
                act = nn.Sigmoid()
            elif activation == 'tanh':
                act = nn.Tanh()
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
                
            layers.append(act)
            layer_infos.append({'type': f'Activation({activation})', 'module': act})
            prev_size = hidden_size
        
        # 输出层
        output_linear = nn.Linear(prev_size, num_classes)
        layers.append(output_linear)
        layer_infos.append({'type': 'Linear', 'module': output_linear})
        
        self.network = nn.Sequential(*layers)
        self.layer_infos = layer_infos
        
        # 存储中间状态
        self._grad_hooks = []
        self._register_grad_hooks()
        self._clear_debug_info()
    
    def _clear_debug_info(self):
        """清空调试信息"""
        self.debug_info = {
            'forward': [],
            'backward': []
        }
    
    def _register_grad_hooks(self):
        """注册梯度hook以捕获梯度信息"""
        # 先移除旧的 hook，避免重复注册导致性能下降
        if hasattr(self, '_grad_hooks'):
            for handle in self._grad_hooks:
                handle.remove()
        self._grad_hooks = []

        def make_hook(name, param_type):
            def hook(grad):
                # 在反向传播时存储梯度信息
                self.debug_info['backward'].append({
                    'param_name': name,
                    'param_type': param_type,
                    'grad_shape': grad.shape,
                    'grad_mean': grad.abs().mean().item(),
                    'grad_max': grad.abs().max().item(),
                    'grad_norm': grad.norm().item()
                })
            return hook
        
        # 注册新的hooks
        for name, param in self.named_parameters():
            if 'weight' in name:
                handle = param.register_hook(make_hook(name, 'weight'))
                self._grad_hooks.append(handle)
            elif 'bias' in name:
                handle = param.register_hook(make_hook(name, 'bias'))
                self._grad_hooks.append(handle)
    
    def forward(self, x):
        self._clear_debug_info()
        batch_size = x.shape[0]
        
        for idx, (module, info) in enumerate(zip(self.network, self.layer_infos)):
            layer_input = x.clone()
            
            x = module(x)
            
            # 记录前向传播信息
            forward_info = {
                'layer_idx': idx,
                'layer_type': info['type'],
                'input_shape': layer_input.shape,
                'output_shape': x.shape,
                'input_sample': layer_input[0].cpu().detach().clone(),
                'output_sample': x[0].cpu().detach().clone(),
            }
            
            if info['type'] == 'Linear':
                forward_info.update({
                    'weights_shape': module.weight.shape,
                    'weights_mean': module.weight.mean().item(),
                    'weights_std': module.weight.std().item(),
                    'bias_shape': module.bias.shape,
                    'bias_mean': module.bias.mean().item(),
                })
            
            self.debug_info['forward'].append(forward_info)
        
        return x