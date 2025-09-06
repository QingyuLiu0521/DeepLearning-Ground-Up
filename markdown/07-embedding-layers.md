# 七、手撕Embedding层
Embedding用于将离散的输入(单词、字符)转换为连续的向量表示。

公式: $$y = W[idx]$$

其中$idx$是输入索引张量，$W$是嵌入矩阵。如果输入$idx$的形状是$[\text{batch\_size}, \text{seq\_length}]$，那么:

- 嵌入矩阵$W$的形状是$[\text{num\_embeddings}, \text{embedding\_dim}]$
- 输出$y$的形状是$[\text{batch\_size}, \text{seq\_length}, \text{embedding\_dim}]$

`padding_idx=k`: 初始化时将指定的embedding(第k行)置为0，并在训练时显式地阻止对应参数更新，使其保持零向量

```py
import torch
import torch.nn as nn

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)
            # 注册梯度钩子来防止padding_idx位置的权重更新
            self.weight.register_hook(self._backward_hook)

    def _backward_hook(self, grad):
        if self.padding_idx is not None:
            grad[self.padding_idx].fill_(0)
        return grad
                
    def forward(self, x):
        if (x < 0).any() or (x >= self.num_embeddings).any():
            raise ValueError(f"indices must be in the range [0, {self.num_embeddings-1}]")
        # x.shape: [B, L]
        # 直接 fancy-indexing
        # output.shape: [B, L, embedding_dim]
        output = self.weight[x]
        
        return output

# 测试数据
batch_size = 3
seq_length = 5
num_embeddings = 10
embedding_dim = 8
padding_idx = 0

x = torch.randint(0, num_embeddings, (batch_size, seq_length))
# 添加padding索引
x[0, 0] = padding_idx
x[1, 2] = padding_idx

# 使用相同的初始权重创建两个嵌入层
initial_weight = torch.randn(num_embeddings, embedding_dim)
initial_weight[padding_idx].fill_(0)

my_embedding = MyEmbedding(num_embeddings=num_embeddings, 
                          embedding_dim=embedding_dim, 
                          padding_idx=padding_idx)
with torch.no_grad():
    my_embedding.weight.copy_(initial_weight)

torch_embedding = nn.Embedding(num_embeddings=num_embeddings, 
                              embedding_dim=embedding_dim, 
                              padding_idx=padding_idx)
with torch.no_grad():
    torch_embedding.weight.copy_(initial_weight)

my_output = my_embedding(x)
torch_output = torch_embedding(x)

# 打印输出形状
print(f"My Embedding output shape: {my_output.shape}")
print(f"PyTorch Embedding output shape: {torch_output.shape}")
# 检查输出中padding位置的嵌入向量是否为零
print(f"My Embedding output at padding position [0,0]:\n{my_output[0,0]}")
print(f"PyTorch Embedding output at padding position [0,0]:\n{torch_output[0,0]}")
print(f"My Embedding output at padding position [1,2]:\n{my_output[1,2]}")
print(f"PyTorch Embedding output at padding position [1,2]:\n{torch_output[1,2]}")

# 创建相同的目标张量用于计算损失
target = torch.randn_like(my_output)

my_loss = (my_output - target).pow(2).sum()
my_loss.backward()
my_grad_at_padding = my_embedding.weight.grad[padding_idx].clone()  # 保存梯度以供后续比较

# 由于在同一个计算图中运行两个模型，当第一个模型的反向传播发生时，PyTorch会计算所有需要梯度的张量的梯度
# 因此需要清除torch_embedding的梯度
torch_embedding.zero_grad()
torch_output = torch_embedding(x)
torch_loss = (torch_output - target).pow(2).sum()
torch_loss.backward()

# 验证padding_idx处的梯度是否都为零
print(f"PyTorch gradient at padding_idx: {torch_embedding.weight.grad[padding_idx]}")
print(f"My gradient at padding_idx: {my_grad_at_padding}")
# 验证非padding_idx处的梯度是否相似
other_idx = 1  # 选择一个非padding的索引进行比较
print(f"PyTorch gradient at index {other_idx}: {torch_embedding.weight.grad[other_idx]}")
print(f"My gradient at index {other_idx}: {my_embedding.weight.grad[other_idx]}")
print(f"Non-padding gradients are similar: {torch.allclose(torch_embedding.weight.grad[1:], my_embedding.weight.grad[1:], rtol=1e-5)}")

'''
My Embedding output shape: torch.Size([3, 5, 8])
PyTorch Embedding output shape: torch.Size([3, 5, 8])
My Embedding output at padding position [0,0]:
tensor([0., 0., 0., 0., 0., 0., 0., 0.], grad_fn=<SelectBackward0>)
PyTorch Embedding output at padding position [0,0]:
tensor([0., 0., 0., 0., 0., 0., 0., 0.], grad_fn=<SelectBackward0>)
My Embedding output at padding position [1,2]:
tensor([0., 0., 0., 0., 0., 0., 0., 0.], grad_fn=<SelectBackward0>)
PyTorch Embedding output at padding position [1,2]:
tensor([0., 0., 0., 0., 0., 0., 0., 0.], grad_fn=<SelectBackward0>)
PyTorch gradient at padding_idx: tensor([0., 0., 0., 0., 0., 0., 0., 0.])
My gradient at padding_idx: tensor([0., 0., 0., 0., 0., 0., 0., 0.])
PyTorch gradient at index 1: tensor([ 3.3747,  3.7849,  1.3418, -3.5435,  2.1515, -3.6038, -0.8204, -2.5532])
My gradient at index 1: tensor([ 3.3747,  3.7849,  1.3418, -3.5435,  2.1515, -3.6038, -0.8204, -2.5532])
Non-padding gradients are similar: True
'''
```