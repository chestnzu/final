# GO层级信息利用分析

## 1. 当前模型中GO层级信息的使用方式

### 方式1: GO Embedding中的隐式编码 ✅ (您已经做了)

```python
# GO embedding生成时已考虑层级结构
# 例如使用以下方法之一:
# - 图嵌入 (Node2Vec, TransE等)
# - 层级编码 (考虑父子关系)
# - 上下文窗口 (父节点、子节点作为上下文)

go_embeddings = go_data["go_embeddings"]
# 这些embedding已经包含了层级信息
# 相似的GO term (如父子关系) 会有相似的embedding
```

**优势**:
- ✅ 简洁：不需要在模型中显式处理层级
- ✅ 灵活：embedding可以捕获复杂的层级关系
- ✅ 高效：不增加模型计算复杂度

**局限**:
- ⚠️ 隐式：难以直接解释层级关系的作用
- ⚠️ 固定：embedding是预训练的，模型训练时不会调整层级关系

---

### 方式2: 预测后的层级传播 ✅ (您也做了)

```python
# 在train_best_model.py中
with Pool(32) as p:
    preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)
```

这个`propagate_annots`函数利用GO的层级关系进行预测传播：
- 如果预测一个子节点，自动传播到所有父节点
- 保证了层级一致性

**优势**:
- ✅ 确保预测符合GO的层级约束
- ✅ 后处理，不影响模型训练

---

### 方式3: 显式的层级感知注意力 ❓ (可选)

在注意力计算时显式使用层级关系：

```python
class HierarchyAwareAttention(nn.Module):
    def __init__(self, go_graph):
        self.go_graph = go_graph  # GO层级图
        self.level_embedding = nn.Embedding(max_level, dim)

    def forward(self, protein, go_context):
        # 1. 计算标准注意力
        attn = standard_attention(protein, go_context)

        # 2. 加入层级偏置
        # 例如：相同层级的GO terms之间的关系
        level_bias = self.compute_level_bias(go_context)

        # 3. 融合
        final_attn = attn + level_bias
        return final_attn
```

**可能的增益**:
- 🤔 显式建模层级关系
- 🤔 可以学习特定的层级交互模式
- 🤔 更强的可解释性

**成本**:
- ❌ 增加模型复杂度
- ❌ 需要额外的层级图数据结构
- ❌ 可能过拟合到特定的GO版本

---

## 2. 您是否需要显式的层级感知注意力？

### 判断依据

让我通过几个问题帮您判断：

#### Q1: 您的GO embedding是如何生成的？

```python
# 查看GO embedding的生成方式
go_data = torch.load(go_embedding_path)
```

**如果GO embedding包含以下信息，则已经足够**:
- ✅ 使用GO graph进行预训练（如Node2Vec, DeepWalk）
- ✅ 考虑了父子关系作为上下文
- ✅ 使用GO的定义文本 + 层级关系训练

**如果GO embedding仅是**:
- ⚠️ 随机初始化
- ⚠️ 仅基于GO名称的文本embedding
- ⚠️ 不考虑层级关系

→ 那么显式层级建模可能有帮助

#### Q2: 当前模型的性能如何？

**如果**:
- ✅ AUPR已经很高（如 > 0.6）
- ✅ 不同层级的GO term预测都不错
- ✅ 父子节点的预测一致性好

→ 不需要额外的层级建模

**如果**:
- ⚠️ 深层GO term（特异性强）预测不好
- ⚠️ 父子节点预测不一致
- ⚠️ 浅层GO term（通用）性能好，但深层差

→ 可以尝试显式层级建模

#### Q3: 您的计算资源和时间预算？

显式层级感知注意力会：
- ❌ 增加10-20%的计算成本
- ❌ 需要额外的实现和调试时间
- ❌ 增加超参数（层级权重等）

如果资源有限 → **不建议**

---

## 3. 建议方案

### 🎯 推荐: 先不使用显式层级感知注意力

**理由**:
1. **您的设计已经很完整**:
   - GO embedding已编码层级
   - 有预测后传播
   - 有双路径融合

2. **消融实验应该优先验证核心组件**:
   - 交叉注意力的作用
   - 双路径融合的必要性
   - 温度缩放的效果

3. **层级感知注意力属于"锦上添花"**，不是核心创新

### 🔬 如果确实想尝试，建议这样做：

#### 方案A: 轻量级层级偏置（最简单）

```python
class Combine_Transformer_WithHierarchy(Combine_Transformer):
    def __init__(self, ..., go_graph=None):
        super().__init__(...)

        if go_graph is not None:
            # 计算GO term的层级level
            self.go_levels = self.compute_levels(go_graph)
            # 为每个level学习一个bias
            self.level_bias = nn.Parameter(torch.zeros(max_level))

    def forward(self, protein_vectors):
        # 标准流程
        go_fusion = self.cross_attention_fusion(...)
        context_scores = go_fusion @ self.go_embedding_vector.T
        direct_scores = protein_vectors @ self.go_embedding_vector.T

        # 添加层级偏置
        if hasattr(self, 'level_bias'):
            # 根据GO term的层级添加不同的bias
            hierarchy_bias = self.level_bias[self.go_levels]
            final_scores = (self.alpha * context_scores +
                          (1 - self.alpha) * direct_scores +
                          self.go_bias +
                          hierarchy_bias)  # 新增
        else:
            final_scores = (self.alpha * context_scores +
                          (1 - self.alpha) * direct_scores +
                          self.go_bias)

        return torch.sigmoid(final_scores / self.temperature)
```

**优势**:
- 简单，只增加max_level个参数
- 可以作为一个消融实验项

**对应消融实验**:
```
Exp-14A: 无层级偏置
Exp-14B: 有层级偏置
```

#### 方案B: 层级正则化损失（推荐）

不改变模型架构，而是加入层级一致性的训练目标：

```python
class HierarchyConsistencyLoss(nn.Module):
    def __init__(self, go_graph):
        super().__init__()
        # 构建父子关系矩阵
        # parent_child_matrix[i,j] = 1 if GO_i is parent of GO_j
        self.parent_child = self.build_parent_child_matrix(go_graph)

    def forward(self, predictions):
        # 如果预测子节点的概率为p，父节点的概率应该 >= p
        # Loss = sum(max(0, child_prob - parent_prob))

        child_probs = predictions  # batch × go_num
        parent_probs = predictions @ self.parent_child  # 父节点概率

        # 违反层级约束的惩罚
        violation = F.relu(child_probs - parent_probs)
        return violation.mean()

# 在训练时
total_loss = criterion(predictions, labels) + lambda_h * hierarchy_loss(predictions)
```

**优势**:
- ✅ 不改变模型架构
- ✅ 软约束，不强制要求
- ✅ 可以调整lambda_h权重

**对应消融实验**:
```
Exp-15A: 无层级正则化
Exp-15B: 有层级正则化 (lambda_h=0.01)
Exp-15C: 有层级正则化 (lambda_h=0.05)
```

---

## 4. 如何验证GO embedding是否已包含层级信息

运行以下分析：

```python
import torch
import numpy as np
from scipy.stats import spearmanr

# 加载GO embedding
go_data = torch.load("../data/go_all_embeddings.pt")
go_embeddings = go_data["go_embeddings"]
go_list = go_data["terms"]

# 加载GO图
from defined_functions import Ontology
go = Ontology("../data/go.obo", with_rels=True)

# 分析1: 父子节点的embedding相似度
parent_child_sims = []
random_pair_sims = []

for i, go_term in enumerate(go_list):
    # 获取父节点
    parents = go.get_parents(go_term)

    for parent in parents:
        if parent in go_list:
            parent_idx = go_list.index(parent)
            # 计算余弦相似度
            sim = torch.cosine_similarity(
                go_embeddings[i:i+1],
                go_embeddings[parent_idx:parent_idx+1]
            ).item()
            parent_child_sims.append(sim)

    # 随机配对作为对照
    random_idx = np.random.randint(0, len(go_list))
    sim = torch.cosine_similarity(
        go_embeddings[i:i+1],
        go_embeddings[random_idx:random_idx+1]
    ).item()
    random_pair_sims.append(sim)

print(f"父子节点平均相似度: {np.mean(parent_child_sims):.4f}")
print(f"随机配对平均相似度: {np.mean(random_pair_sims):.4f}")

# 如果父子相似度显著高于随机配对，说明embedding已编码层级信息
from scipy import stats
t_stat, p_value = stats.ttest_ind(parent_child_sims, random_pair_sims)
print(f"T检验 p-value: {p_value:.6f}")

if p_value < 0.01 and np.mean(parent_child_sims) > np.mean(random_pair_sims):
    print("✅ GO embedding已经包含层级信息，不需要显式层级建模")
else:
    print("⚠️ GO embedding可能未充分编码层级信息，可以考虑显式层级建模")
```

---

## 5. 最终建议

### 对于您的消融实验：

**不建议**立即加入层级感知注意力，因为：
1. ✅ 您的GO embedding已经考虑了层级
2. ✅ 您有预测后传播确保层级一致性
3. ✅ 消融实验应该聚焦在核心创新点

### 如果未来想尝试（论文的Future Work部分）：

可以提到：
```latex
\section{Future Work}
While our model implicitly leverages GO hierarchy through pre-trained
GO embeddings, future work could explore explicit hierarchy-aware
attention mechanisms or hierarchy consistency regularization to
further improve performance on deep-level GO terms.
```

### 当前优先级排序：

1. 🔥 **高优先级** - 核心消融实验:
   - 交叉注意力
   - 双路径融合
   - 损失函数对比
   - 温度缩放

2. ⭐ **中优先级** - 结构优化:
   - 残差连接
   - 注意力层数
   - GO嵌入学习策略

3. 📊 **低优先级** - 可选增强:
   - 层级感知机制
   - 不同的融合策略
   - 更复杂的注意力变体

---

## 总结

**您当前的设计已经很合理了！**

- GO embedding中的层级编码 ✅
- 预测后的层级传播 ✅
- 双路径融合设计 ✅

**不需要**额外的显式层级感知注意力，除非：
- 验证GO embedding确实没包含层级信息
- 当前模型在深层GO term上表现不佳
- 有充足的时间和资源进行额外实验

**建议**: 先完成核心消融实验，如果性能已经很好，层级感知注意力就不是必需的。
