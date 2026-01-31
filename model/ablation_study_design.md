# 消融实验设计方案

## 模型概述
当前模型 `Combine_Transformer` 包含以下核心组件：
1. 蛋白质嵌入投影层 (fc1 + fc1_5)
2. 交叉注意力机制 (单层/多层)
3. 残差连接 (fc2, fc1_5)
4. GO上下文嵌入 (go_context - 固定)
5. GO嵌入向量 (go_embedding_vector - 可学习)
6. 双路径融合 (alpha参数控制的context_scores和direct_scores融合)
7. GO偏置项 (go_bias)
8. 温度缩放 (temperature scaling - 可选)
9. 损失函数 (BCE/Focal/Asymmetric/Combined)

---

## 一、结构组件消融实验

### 1.1 交叉注意力机制的作用
**目的**: 验证交叉注意力机制对性能的贡献

| 实验ID | 配置 | 描述 |
|--------|------|------|
| Exp-1A | 完整模型 (num_layers=1) | 基线：单层交叉注意力 |
| Exp-1B | 移除交叉注意力 | 仅使用direct_scores路径 (设置alpha=0) |
| Exp-1C | 多层交叉注意力 (num_layers=2) | 验证深度的影响 |
| Exp-1D | 多层交叉注意力 (num_layers=3) | 验证更深的网络 |

**实现方式**:
- Exp-1B: 设置 `alpha=0.0`，相当于只使用直接打分路径
- Exp-1C/1D: 使用 `num_layers=2/3` 参数

---

### 1.2 双路径融合策略
**目的**: 验证context_scores和direct_scores融合的必要性

| 实验ID | 配置 | 描述 |
|--------|------|------|
| Exp-2A | alpha=0.5 (可学习) | 基线：自动学习融合权重 |
| Exp-2B | alpha=1.0 (固定) | 仅使用context_scores |
| Exp-2C | alpha=0.0 (固定) | 仅使用direct_scores |
| Exp-2D | alpha=0.3 (固定) | 偏向direct_scores |
| Exp-2E | alpha=0.7 (固定) | 偏向context_scores |

**实现方式**:
- 修改模型初始化，将 `self.alpha = nn.Parameter(torch.tensor(alpha_value))` 改为固定值
- 对于可学习的alpha，记录训练后的最终alpha值

---

### 1.3 残差连接的影响
**目的**: 验证残差连接对训练稳定性和性能的影响

| 实验ID | 配置 | 描述 |
|--------|------|------|
| Exp-3A | 完整残差 (fc1_5 + fc2) | 基线 |
| Exp-3B | 移除fc1_5残差 | 只保留fc2残差 |
| Exp-3C | 移除fc2残差 | 只保留fc1_5残差 |
| Exp-3D | 移除所有残差 | 纯前馈网络 |

**实现方式**:
- 将 `Residual(mlpblock(...))` 替换为 `mlpblock(...)`

---

### 1.4 GO嵌入向量的学习策略
**目的**: 验证GO嵌入向量是否需要可学习

| 实验ID | 配置 | 描述 |
|--------|------|------|
| Exp-4A | 可学习GO嵌入 | 基线：`nn.Parameter(embedding_vector)` |
| Exp-4B | 固定GO嵌入 | 不更新GO嵌入向量 |
| Exp-4C | 随机初始化GO嵌入 | 从头学习GO嵌入 |

**实现方式**:
- Exp-4B: 使用 `self.register_buffer('go_embedding_vector', embedding_vector)` 代替 `nn.Parameter`
- Exp-4C: 使用 `nn.Parameter(torch.randn_like(embedding_vector))` 随机初始化

---

### 1.5 GO偏置项的必要性
**目的**: 验证GO偏置项是否提升性能

| 实验ID | 配置 | 描述 |
|--------|------|------|
| Exp-5A | 有GO偏置 | 基线 |
| Exp-5B | 无GO偏置 | 移除 `self.go_bias` |

**实现方式**:
- 在forward函数中移除 `+ self.go_bias` 部分

---

### 1.6 温度缩放的影响
**目的**: 验证温度缩放对概率校准的作用

| 实验ID | 配置 | 描述 |
|--------|------|------|
| Exp-6A | 可学习温度 (初始=1.5) | 基线 |
| Exp-6B | 无温度缩放 (temp=1.0) | 标准sigmoid |
| Exp-6C | 固定温度 (temp=1.5) | 不学习温度 |
| Exp-6D | 固定温度 (temp=2.0) | 更平滑的概率分布 |

**实现方式**:
- 使用 `--temperature` 标志控制是否使用温度缩放
- 对于固定温度，将 `nn.Parameter` 改为常量

**评估指标**: 除了AUPR/AUROC，还需评估校准误差 (ECE - Expected Calibration Error)

---

## 二、损失函数消融实验

### 2.1 损失函数对比
**目的**: 找到最适合此任务的损失函数

| 实验ID | 损失函数 | 参数设置 |
|--------|----------|----------|
| Exp-7A | Binary Cross Entropy | 标准BCE |
| Exp-7B | Focal Loss | alpha=0.2, gamma=2 |
| Exp-7C | Asymmetric Loss | gamma_neg=2, gamma_pos=0, clip=0.05 |
| Exp-7D | Combined Loss | focal_weight=0.7, bce_weight=0.3 |
| Exp-7E | Focal Loss (不同gamma) | gamma=1, gamma=3 |
| Exp-7F | Combined Loss (不同权重) | focal_weight=0.5, 0.9 |

**实现方式**:
- 使用 `--loss` 参数控制

**评估重点**:
- AUPR (特别重要，因为数据不平衡)
- 各GO层级的性能
- 稀有GO term的召回率

---

## 三、网络深度和宽度实验

### 3.1 注意力层数
**目的**: 找到最优的交叉注意力层数

| 实验ID | 层数 | 描述 |
|--------|------|------|
| Exp-8A | num_layers=1 | 基线 |
| Exp-8B | num_layers=2 | 中等深度 |
| Exp-8C | num_layers=3 | 深层网络 |
| Exp-8D | num_layers=4 | 更深网络 |

**实现方式**:
- 修改配置文件中的 `num_layers` 参数

---

### 3.2 注意力头数
**目的**: 验证多头注意力的头数影响

| 实验ID | 头数 | 描述 |
|--------|------|------|
| Exp-9A | num_heads=8 | 基线 |
| Exp-9B | num_heads=4 | 较少的头 |
| Exp-9C | num_heads=16 | 较多的头 |
| Exp-9D | num_heads=1 | 单头注意力 |

**实现方式**:
- 修改配置文件中的 `num_heads` 参数

---

## 四、输入特征消融实验

### 4.1 GO上下文信息的作用
**目的**: 验证GO上下文嵌入的贡献

| 实验ID | 配置 | 描述 |
|--------|------|------|
| Exp-10A | 使用GO上下文 | 基线 |
| Exp-10B | 移除GO上下文 | 仅使用GO嵌入向量 |
| Exp-10C | 随机GO上下文 | 使用随机向量代替 |

**实现方式**:
- Exp-10B: 将 `go_context` 替换为 `go_embedding_vector`
- Exp-10C: 使用 `torch.randn_like(go_context)` 初始化

---

### 4.2 蛋白质嵌入维度压缩
**目的**: 验证中间层维度的影响

| 实验ID | hidden_dim | 描述 |
|--------|------------|------|
| Exp-11A | 与GO嵌入相同 | 基线 (通常是512或1024) |
| Exp-11B | hidden_dim=256 | 更小的瓶颈 |
| Exp-11C | hidden_dim=1024 | 更大的容量 |
| Exp-11D | hidden_dim=2048 | 不压缩 |

**实现方式**:
- 修改 `fc1` 的输出维度

---

## 五、训练策略消融实验

### 5.1 Dropout率影响
**目的**: 找到最优正则化强度

| 实验ID | Dropout | 描述 |
|--------|---------|------|
| Exp-12A | dropout=0.1 | 基线 |
| Exp-12B | dropout=0.0 | 无dropout |
| Exp-12C | dropout=0.2 | 中等正则化 |
| Exp-12D | dropout=0.3 | 强正则化 |

---

### 5.2 学习率调度策略
**目的**: 验证当前学习率调度的有效性

| 实验ID | 策略 | 描述 |
|--------|------|------|
| Exp-13A | StepLR (step=5, gamma=0.8) | 基线 |
| Exp-13B | 固定学习率 | 无调度器 |
| Exp-13C | CosineAnnealingLR | 余弦退火 |
| Exp-13D | ReduceLROnPlateau | 基于验证loss调整 |

---

## 六、实验执行计划

### 6.1 优先级分级

**高优先级** (核心组件，必须执行):
- Exp-1: 交叉注意力机制
- Exp-2: 双路径融合策略
- Exp-7: 损失函数对比

**中优先级** (重要组件):
- Exp-3: 残差连接
- Exp-4: GO嵌入学习策略
- Exp-6: 温度缩放
- Exp-8: 注意力层数

**低优先级** (调优实验):
- Exp-5, 9, 10, 11, 12, 13

### 6.2 实验执行顺序建议

1. **第一轮**: 核心组件验证
   - Exp-1A (基线), Exp-1B, Exp-2A, Exp-2B, Exp-2C
   - Exp-7A, Exp-7B, Exp-7C, Exp-7D

2. **第二轮**: 结构优化
   - Exp-1C, Exp-1D, Exp-8B, Exp-8C
   - Exp-3A, Exp-3B, Exp-3C, Exp-3D
   - Exp-4A, Exp-4B, Exp-4C

3. **第三轮**: 精细调优
   - Exp-6系列, Exp-9系列
   - Exp-11系列, Exp-12系列

---

## 七、评估指标

### 7.1 主要指标
- **AUPR** (Area Under Precision-Recall Curve) - 最重要
- **AUROC** (Area Under ROC Curve)
- **Fmax** (Maximum F1 score)
- **SMIN** (Semantic distance)

### 7.2 辅助指标
- **训练时间**
- **模型参数量**
- **推理速度**
- **验证集最优epoch**
- **最终学习到的alpha值** (对于可学习alpha的实验)
- **最终学习到的temperature值** (对于温度缩放实验)

### 7.3 分层分析
对于每个实验，还需要分析：
- 不同GO层级 (level 1-10) 的性能
- 不同频率GO term的性能 (高频 vs 低频)
- 三个GO aspect (MF, BP, CC) 的性能

---

## 八、实验记录模板

### 8.1 建议创建结果表格

```python
# 创建实验结果记录表
results = {
    'experiment_id': [],
    'description': [],
    'aupr': [],
    'auroc': [],
    'fmax': [],
    'smin': [],
    'train_time': [],
    'num_params': [],
    'best_epoch': [],
    'final_alpha': [],  # 如果适用
    'final_temperature': [],  # 如果适用
}
```

### 8.2 可视化建议
- 绘制各实验的AUPR/AUROC对比条形图
- 绘制不同alpha值的性能曲线
- 绘制不同层数/头数的性能曲线
- 绘制训练loss和验证loss曲线对比

---

## 九、实验脚本模板

以下是一个消融实验的执行脚本示例：

```bash
#!/bin/bash

# 实验1A: 基线模型
python train_best_model.py --aspect mf --loss focal --seed 42

# 实验1B: 移除交叉注意力 (需要修改代码设置alpha=0)
python train_best_model.py --aspect mf --loss focal --seed 42 --alpha 0.0

# 实验2系列: 不同损失函数
for loss in bce focal asymmetric combined; do
    python train_best_model.py --aspect mf --loss $loss --seed 42
done

# 实验6系列: 温度缩放
python train_best_model.py --aspect mf --loss focal --seed 42 --temperature

# 多随机种子验证 (对最佳配置)
for seed in 42 123 456 789 2023; do
    python train_best_model.py --aspect mf --loss focal --seed $seed
done
```

---

## 十、预期发现和假设

### 10.1 预期验证的假设
1. **交叉注意力假设**: 交叉注意力能显著提升性能 (+5-10% AUPR)
2. **双路径假设**: 融合context和direct路径优于单一路径
3. **残差假设**: 残差连接能提升训练稳定性和最终性能
4. **温度假设**: 温度缩放能改善概率校准，提升AUPR
5. **损失函数假设**: Focal/Asymmetric Loss在不平衡数据上优于BCE

### 10.2 需要回答的问题
1. 哪些组件是性能提升的关键？
2. 哪些组件可以移除以简化模型？
3. 最优的超参数组合是什么？
4. 模型的哪些部分对计算成本贡献最大？
5. 性能提升是否在统计上显著？(需要多次运行取平均)

---

## 十一、统计显著性验证

对于关键实验，建议：
1. 使用**多个随机种子** (至少3-5个): 42, 123, 456, 789, 2023
2. 计算**均值和标准差**
3. 使用**配对t检验**比较实验组和对照组
4. 设置显著性水平 α=0.05

---

## 十二、论文撰写建议

### 消融实验表格示例

| Model Variant | AUPR ↑ | AUROC ↑ | Fmax ↑ | ΔAUPRᵃ |
|---------------|--------|---------|--------|--------|
| Full Model | 0.XXX±0.XXX | 0.XXX±0.XXX | 0.XXX±0.XXX | - |
| w/o Cross-Attention | 0.XXX±0.XXX | 0.XXX±0.XXX | 0.XXX±0.XXX | -X.X% |
| w/o Dual-Path Fusion | 0.XXX±0.XXX | 0.XXX±0.XXX | 0.XXX±0.XXX | -X.X% |
| w/o Temperature | 0.XXX±0.XXX | 0.XXX±0.XXX | 0.XXX±0.XXX | -X.X% |
| w/o Residual | 0.XXX±0.XXX | 0.XXX±0.XXX | 0.XXX±0.XXX | -X.X% |
| Focal Loss → BCE | 0.XXX±0.XXX | 0.XXX±0.XXX | 0.XXX±0.XXX | -X.X% |

ᵃ Relative change compared to full model

---

## 总结

这套消融实验方案系统地验证了模型的每个关键组件。建议：
1. **先执行高优先级实验**，确定核心组件的有效性
2. **使用多个随机种子**确保结果可靠
3. **记录详细的实验日志**，包括超参数、训练时间、最终性能等
4. **可视化结果**，便于分析和论文撰写
5. **保存所有实验的模型检查点**，便于后续分析

通过这些实验，您将能够：
- 理解每个组件的贡献
- 找到最优的模型配置
- 为论文提供充分的实验支持
- 识别潜在的改进方向
