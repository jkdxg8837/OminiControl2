# OminiControl2 (预览版)

感谢您尝试OminiControl2！这是一个预览版本，包含一些实验性功能。某些特性可能尚未完全稳定。如果您遇到任何问题，请向我们反馈。

## 联系方式

如有问题或反馈:
- **邮箱**: Yuanshi9815@outlook.com
- **微信**: +65 88200624

## TL;DR
OminiControl2通过两种主要方法提高了效率:
- **Feature Reusing**: 跨时间步缓存条件特征，避免重复计算
- **Compact Token Representation**: 通过压缩和剪枝技术减少token数量

使用`cache_per_n_steps`参数可立即提升现有模型的效率。如果是研究目的，可以通过设置`independent_condition: True`来探索Asymmetric Attention Masking。

## 核心功能

### 1. Feature Reusing
#### Conditional Feature Caching
为了优化推理性能，您可以通过在`generate`函数中使用`cache_per_n_steps`参数来复用特征。

例如：
- 设置`cache_per_n_steps=5`可以每5步重新计算一次条件图像特征
- 此功能与之前的OminiControl1模型兼容

为了获得最高效率，您可以将`cache_per_n_steps`设置为大于总推理步数的值，这样条件特征只会计算一次。注意，单次计算缓存在使用**Asymmetric Attention Masking**训练的模型上效果最佳。

#### Asymmetric Attention Masking
要训练带有Asymmetric Attention Masking的新模型：
- 在配置文件中设置`independent_condition: True`
- 示例配置文件可在`train/config/omini2`目录中找到

> **⚠️ 质量提示：** 请注意，Asymmetric Attention Masking当前可能在某些情况下导致质量下降，我们的后续研究正在尝试解决这个问题。此功能仅推荐用于研究目的。对于生产用例，我们建议仅使用Caching机制而不使用非对称掩码，以达到效率和质量的最佳平衡（即沿用OminiControl1的训练代码）。

### 2. Compact Token Representation

1. **Compression with Position Correcting** (已包含在此预览版中)
   - 通过在配置文件中设置较低的`condition_size`来启用
   - 调整`position_scale`参数以保持适当的空间对齐

2. **Token Pruning** (即将推出)
   - 目前正在优化中
   - 将允许选择性地移除非信息性token

3. **Token Integration for Mask Repainting** (即将推出)
   - 专为inpainting任务优化
   - 目前正在完善以达到最佳性能