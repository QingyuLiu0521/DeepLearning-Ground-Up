# DeepLearning-Ground-Up
This project demonstrates how to build core modern deep learning components without relying on high-level PyTorch abstractions.

### 📚 Series Contents

- **[Loss Functions from Scratch](./Markdown/01-loss-functions.md/)** - Cross-entropy, MSE, focal loss, and more
- **[Activation Functions from Scratch](./Markdown/02-activation-functions.md/)** - ReLU, GELU, Swish, and advanced variants
- **[Attention Mechanisms from Scratch](./Markdown/03-attention-mechanisms.md/)** - Scaled dot-product Attention, Multi-Head Attention
- **[Positional Encodings from Scratch](./Markdown/04-positional-encodings.md/)** - Sinusoidal, relative encodings, RoPE
- **[Normalization & Regularization from Scratch](./Markdown/05-normalization-regularization.md/)** - LayerNorm, BatchNorm, Dropout, and more
- **[Linear Layers from Scratch](./Markdown/06-linear-layers.md/)** - Dense layers and weight initialization
- **[Embedding Layers from Scratch](./Markdown/07-embedding-layers.md/)** - Token embeddings
- **[Transformer Architecture from Scratch](./Markdown/08-transformer.md/)** - Complete Transformer implementation with encoder-decoder architecture

### 🚫 Constraints (Following CS336 Guidelines)
**Not Allowed:**
- `torch.nn` modules (except containers)
- `torch.nn.functional` functions
- `torch.optim` optimizers (except base class)

**Allowed:**
- `torch.nn.Parameter` for trainable parameters
- Container classes: `torch.nn.Module`, `torch.nn.ModuleList`, `torch.nn.Sequential`
- `torch.optim.Optimizer` base class for custom optimizers
- Core PyTorch tensor operations

---

本项目展示如何在不依赖 PyTorch 高级抽象的情况下构建核心现代深度学习组件。

### 📚 系列内容

- **[从零实现损失函数](./Markdown/01-loss-functions.md/)** - 交叉熵、均方误差、焦点损失等
- **[从零实现激活函数](./Markdown/02-activation-functions.md/)** - ReLU、GELU、Swish 及高级变体
- **[从零实现注意力机制](./Markdown/03-attention-mechanisms.md/)** - 缩放点积注意力、多头注意力
- **[从零实现位置编码](./Markdown/04-positional-encodings.md/)** - 正弦编码、相对编码、旋转位置编码(RoPE)
- **[从零实现归一化和正则化](./Markdown/05-normalization-regularization.md/)** - 层归一化、批归一化、Dropout 等
- **[从零实现线性层](./Markdown/06-linear-layers.md/)** - 全连接层和权重初始化
- **[从零实现嵌入层](./Markdown/07-embedding-layers.md/)** - 词嵌入
- **[从零实现 Transformer 架构](./Markdown/08-transformer.md/)** - 完整的编码器-解码器 Transformer 实现

### 🚫 实现要求（遵循 CS336 准则）
**不允许使用:**
- `torch.nn` 模块（容器类除外）
- `torch.nn.functional` 函数
- `torch.optim` 优化器（基类除外）

**允许使用:**
- `torch.nn.Parameter` 用于可训练参数
- 容器类: `torch.nn.Module`、`torch.nn.ModuleList`、`torch.nn.Sequential`
- `torch.optim.Optimizer` 基类用于自定义优化器
- PyTorch 核心张量操作

> Zhihu Link: https://www.zhihu.com/column/c_1930370026797503503