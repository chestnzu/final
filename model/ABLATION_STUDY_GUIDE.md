# 消融实验执行指南

本指南将帮助您系统地执行消融实验并分析结果。

## 📋 文件说明

1. **[ablation_study_design.md](ablation_study_design.md)** - 详细的消融实验设计方案
2. **[run_ablation_experiments.py](run_ablation_experiments.py)** - 自动化实验执行脚本
3. **[visualize_ablation_results.py](visualize_ablation_results.py)** - 结果可视化脚本
4. **[train_best_model.py](train_best_model.py)** - 原始训练脚本

---

## 🚀 快速开始

### 步骤 1: 准备环境

确保已安装必要的依赖:
```bash
pip install pandas numpy torch matplotlib seaborn tqdm click
```

### 步骤 2: 运行基线实验

首先运行基线实验，确保一切正常:
```bash
python train_best_model.py --aspect mf --loss focal --seed 42
```

### 步骤 3: 执行消融实验

#### 方法 A: 手动运行单个实验

```bash
# 实验1: 基线模型
python train_best_model.py --aspect mf --loss focal --seed 42

# 实验2: 使用温度缩放
python train_best_model.py --aspect mf --loss focal --seed 42 --temperature

# 实验3: 不同损失函数
python train_best_model.py --aspect mf --loss bce --seed 42
python train_best_model.py --aspect mf --loss combined --seed 42
```

#### 方法 B: 使用自动化脚本

```bash
# 运行所有预定义的实验
python run_ablation_experiments.py --aspect mf --seeds 42 123 456

# 运行特定实验
python run_ablation_experiments.py --aspect mf --exp-ids Exp-1A Exp-6A Exp-7-FOCAL

# 使用单个种子快速测试
python run_ablation_experiments.py --aspect mf --seeds 42 --exp-ids Exp-1A
```

### 步骤 4: 可视化结果

```bash
# 生成所有可视化图表
python visualize_ablation_results.py --results-file ablation_results/ablation_results.csv

# 同时生成LaTeX表格
python visualize_ablation_results.py --results-file ablation_results/ablation_results.csv --latex
```

---

## 🔧 代码修改指南

某些消融实验需要修改模型代码。以下是具体的修改指南:

### 1. 修改 Alpha 参数 (双路径融合)

在 [all_models.py](all_models.py) 的 `Combine_Transformer.__init__` 中修改:

```python
# 原始代码 (可学习alpha)
self.alpha = nn.Parameter(torch.tensor(0.5))

# 修改为固定alpha=0.0 (仅使用direct_scores)
# self.alpha = nn.Parameter(torch.tensor(0.5))
self.alpha = 0.0

# 修改为固定alpha=1.0 (仅使用context_scores)
# self.alpha = nn.Parameter(torch.tensor(0.5))
self.alpha = 1.0
```

或者，添加参数支持:

```python
class Combine_Transformer(nn.Module):
    def __init__(self, ..., alpha_value=None, learnable_alpha=True):
        ...
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_value or 0.5))
        else:
            self.alpha = alpha_value or 0.5
```

### 2. 移除残差连接

在 [all_models.py](all_models.py) 中修改:

```python
# 原始代码 (有残差)
self.fc1_5 = Residual(mlpblock(hidden_dim, hidden_dim, ...))
self.fc2 = Residual(mlpblock(hidden_dim, hidden_dim, ...))

# 修改为无残差
self.fc1_5 = mlpblock(hidden_dim, hidden_dim, ...)
self.fc2 = mlpblock(hidden_dim, hidden_dim, ...)
```

### 3. 固定或随机初始化 GO 嵌入

```python
# 原始代码 (可学习)
self.go_embedding_vector = nn.Parameter(embedding_vector.clone().detach())

# 修改为固定 (不学习)
self.register_buffer('go_embedding_vector', embedding_vector.clone().detach())

# 修改为随机初始化
self.go_embedding_vector = nn.Parameter(torch.randn_like(embedding_vector))
```

### 4. 移除 GO 偏置

在 forward 函数中:

```python
# 原始代码
final_scores = self.alpha * context_scores + (1 - self.alpha) * direct_scores + self.go_bias

# 修改为无偏置
final_scores = self.alpha * context_scores + (1 - self.alpha) * direct_scores
```

### 5. 修改注意力层数和头数

在配置文件 [config/model_config.json](config/model_config.json) 中:

```json
{
  "profiles": {
    "default": {
      "epoch_num": 100,
      "train_bs": 32,
      "eval_bs": 64,
      "num_heads": 8,      // 修改这里: 1, 4, 8, 16
      "num_layers": 1      // 修改这里: 1, 2, 3, 4
    }
  }
}
```

或者在命令行中添加参数支持 (需要修改 `train_best_model.py`):

```python
@ck.option('--num_heads', '-nh', default=8, type=int)
@ck.option('--num_layers', '-nl', default=1, type=int)
def main(..., num_heads, num_layers):
    # 使用这些参数而不是从配置文件读取
```

---

## 📊 实验组织建议

### 第一轮: 核心组件验证 (1-2天)

优先执行以下高优先级实验:

```bash
# 损失函数对比 (最关键)
python train_best_model.py --aspect mf --loss bce --seed 42
python train_best_model.py --aspect mf --loss focal --seed 42
python train_best_model.py --aspect mf --loss asymmetric --seed 42
python train_best_model.py --aspect mf --loss combined --seed 42

# 温度缩放
python train_best_model.py --aspect mf --loss focal --seed 42 --temperature
python train_best_model.py --aspect mf --loss focal --seed 42  # 无温度
```

### 第二轮: 结构组件 (2-3天)

需要修改代码:
- Alpha参数实验 (修改 `all_models.py`)
- 残差连接实验 (修改 `all_models.py`)
- GO嵌入学习策略 (修改 `all_models.py`)

### 第三轮: 超参数调优 (1-2天)

修改配置文件:
- 注意力层数 (1, 2, 3)
- 注意力头数 (4, 8, 16)

---

## 📈 结果分析

### 1. 查看实验结果

```bash
# 查看CSV结果
cat ablation_results/ablation_results.csv

# 或使用pandas
python -c "import pandas as pd; df = pd.read_csv('ablation_results/ablation_results.csv'); print(df)"
```

### 2. 生成可视化

```bash
python visualize_ablation_results.py \
    --results-file ablation_results/ablation_results.csv \
    --output-dir ablation_plots \
    --latex
```

将生成以下图表:
- `component_ablation_aupr.png` - 组件消融对比
- `loss_function_comparison_aupr.png` - 损失函数对比
- `temperature_scaling_effect.png` - 温度缩放效果
- `performance_heatmap.png` - 性能热力图
- `radar_chart_comparison.png` - 雷达图对比
- `performance_delta_aupr.png` - 性能变化图
- `ablation_table.tex` - LaTeX表格 (可直接用于论文)

### 3. 统计显著性检验

使用多个随机种子确保结果可靠:

```bash
# 对最佳配置运行5次
for seed in 42 123 456 789 2023; do
    python train_best_model.py --aspect mf --loss focal --seed $seed --temperature
done
```

然后计算均值和标准差:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('ablation_results/ablation_results.csv')

# 筛选特定实验
best_config = df[df['exp_id'] == 'Exp-6A']

# 计算统计信息
print(f"AUPR: {best_config['aupr'].mean():.4f} ± {best_config['aupr'].std():.4f}")
print(f"AUROC: {best_config['auroc'].mean():.4f} ± {best_config['auroc'].std():.4f}")

# 配对t检验
from scipy import stats
baseline = df[df['exp_id'] == 'Exp-1A']['aupr']
treatment = df[df['exp_id'] == 'Exp-6A']['aupr']
t_stat, p_value = stats.ttest_rel(baseline, treatment)
print(f"P-value: {p_value:.4f}")
```

---

## 🎯 关键指标解读

### AUPR (Area Under Precision-Recall Curve)
- **最重要的指标**，特别适合不平衡数据
- 关注正类的预测质量
- 越高越好 (0-1之间)

### AUROC (Area Under ROC Curve)
- 衡量分类器的整体性能
- 对类别不平衡不太敏感
- 越高越好 (0-1之间)

### Fmax (Maximum F1 Score)
- 精确率和召回率的调和平均
- 考虑最优阈值下的性能
- 越高越好 (0-1之间)

### 相对性能提升
```
ΔAUPR = (AUPR_treatment - AUPR_baseline) / AUPR_baseline × 100%
```

一般认为:
- **> 5%**: 显著提升
- **2-5%**: 中等提升
- **< 2%**: 轻微提升

---

## 📝 论文撰写建议

### 1. 消融实验表格

使用生成的 `ablation_table.tex`:

```latex
\begin{table}[htbp]
\centering
\caption{Ablation study results on MF aspect}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
Model Variant & AUPR $\uparrow$ & AUROC $\uparrow$ & $\Delta$AUPR (\%) \\
\midrule
Full Model & 0.XXX$\pm$0.XXX & 0.XXX$\pm$0.XXX & - \\
w/o Cross-Attention & 0.XXX$\pm$0.XXX & 0.XXX$\pm$0.XXX & -X.X \\
w/o Temperature & 0.XXX$\pm$0.XXX & 0.XXX$\pm$0.XXX & -X.X \\
...
\bottomrule
\end{tabular}
\end{table}
```

### 2. 文字描述模板

```latex
\subsection{Ablation Study}

We conducted comprehensive ablation experiments to validate the contribution
of each component in our model. As shown in Table~\ref{tab:ablation},
the full model achieves the best performance with an AUPR of XXX.

\textbf{Cross-attention mechanism:} Removing the cross-attention module
results in a X.X\% drop in AUPR, demonstrating its critical role in
capturing the relationship between protein features and GO terms.

\textbf{Temperature scaling:} The learnable temperature parameter improves
AUPR by X.X\%, indicating better probability calibration.

\textbf{Loss function:} We compared four loss functions (BCE, Focal,
Asymmetric, Combined). The Focal loss performs best, achieving X.X\%
improvement over standard BCE, which confirms its effectiveness for
imbalanced classification tasks.
```

### 3. 图表引用

```latex
Figure~\ref{fig:ablation} illustrates the performance comparison across
different model configurations. The results clearly show that [...]
```

---

## 🐛 常见问题

### Q1: 实验运行很慢怎么办?
A:
- 减少 epoch 数量进行快速测试
- 使用更小的 batch size
- 先在小数据集上验证

### Q2: 某些实验需要修改代码，如何管理?
A:
- 使用 git 分支管理不同的实验配置
- 或者创建配置参数，通过命令行控制
- 保存每个实验的代码快照

### Q3: 如何确保实验可复现?
A:
- 固定随机种子 (`--seed 42`)
- 记录所有超参数
- 保存模型检查点和配置文件
- 使用版本控制 (git)

### Q4: 多个GPU如何并行运行实验?
A:
```bash
# 在不同GPU上运行不同实验
CUDA_VISIBLE_DEVICES=0 python train_best_model.py --aspect mf --loss focal &
CUDA_VISIBLE_DEVICES=1 python train_best_model.py --aspect mf --loss bce &
CUDA_VISIBLE_DEVICES=2 python train_best_model.py --aspect mf --loss asymmetric &
wait
```

---

## 📚 参考资料

### 消融实验相关论文
1. "Attention Is All You Need" - 多头注意力消融
2. "Focal Loss for Dense Object Detection" - 损失函数对比
3. "On Calibration of Modern Neural Networks" - 温度缩放

### 评估指标
- AUPR vs AUROC: 何时使用哪个指标
- F1 score 最优阈值选择
- 统计显著性检验方法

---

## ✅ 检查清单

实验执行前:
- [ ] 数据集准备完成
- [ ] 环境依赖安装完成
- [ ] 基线实验成功运行
- [ ] 确定实验优先级

实验执行中:
- [ ] 记录每个实验的配置
- [ ] 保存模型检查点
- [ ] 监控训练过程
- [ ] 定期备份结果

实验完成后:
- [ ] 收集所有实验结果
- [ ] 生成可视化图表
- [ ] 计算统计显著性
- [ ] 撰写实验分析
- [ ] 准备论文表格和图

---

## 🎓 建议的实验执行顺序

1. **Day 1**: 运行基线 + 损失函数对比 (Exp-1A, Exp-7系列)
2. **Day 2**: 温度缩放 + 多随机种子验证 (Exp-6系列)
3. **Day 3**: 修改代码 - Alpha参数实验 (Exp-2系列)
4. **Day 4**: 修改代码 - 残差连接实验 (Exp-3系列)
5. **Day 5**: 注意力层数/头数实验 (Exp-8, Exp-9)
6. **Day 6-7**: 补充实验 + 结果分析 + 可视化

预计总时间: **1周**

---

## 📞 需要帮助?

如果遇到问题:
1. 检查错误日志
2. 验证数据路径
3. 确认GPU可用性
4. 查看配置文件格式

祝实验顺利! 🎉
