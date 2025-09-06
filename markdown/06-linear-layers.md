# 六、手撕线性层
线性层(Linear Layer)是神经网络中最基础的层，对输入进行线性变换，常与非线性激活函数（如ReLU）配合使用形成MLP

公式: $$y = xW^T + b$$

其中$x$是输入张量, $W$是权重矩阵, $b$是偏置向量。如果输入$x$的形状是$[\text{batch\_size}, \text{in\_features}]$，那么:

- 权重$W$的形状是 $[\text{out\_features}, \text{in\_features}]$
- 偏置$b$的形状是 $[\text{out\_features}]$
- 输出$y$的形状是 $[\text{batch\_size}, \text{out\_features}]$

根据PyTorch源码，权重初始化采用 Kaiming 均匀分布初始化: $W \sim U(-\sqrt{\frac{6}{fan\_in}}, \sqrt{\frac{6}{fan\_in}})$

偏置初始化采用均匀分布: $b \sim U(-\frac{1}{\sqrt{fan\_in}}, \frac{1}{\sqrt{fan\_in}})$

其中$\text{fan\_in}$是输入特征数。

```py
import torch
from torch import nn
import math

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output


batch = 2
in_features = 4
out_features = 8

x = torch.randn(batch, in_features)
my_linear = MyLinear(in_features=in_features, out_features=out_features, bias=True)
torch_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

print(f"Input shape: {x.shape}")
print(f"My Linear output shape: {my_linear(x).shape}")
print(f"Pytorch Linear output shape: {torch_linear(x).shape}")

'''
Input shape: torch.Size([2, 4])
My Linear output shape: torch.Size([2, 8])
Pytorch Linear output shape: torch.Size([2, 8])
'''
```