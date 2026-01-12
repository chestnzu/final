import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_adj
from evaluation import *


class mlpblock(nn.Module):
    def __init__(self,input_dim=2560, output_dim=1024,bias=True,layer_norm=True,dropout=0.1,activation=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim,bias)
        self.activation = activation
        self.use_layer_norm = layer_norm
        self.norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.use_dropout = dropout > 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        if self.use_layer_norm and self.norm is not None:
            x = self.norm(x)
        if self.use_dropout and self.dropout is not None:
            x = self.dropout(x)
        return x

class Residual(nn.Module):
    def __init__(self,function):
        super().__init__()
        self.fn=function
    
    def forward(self,x):
        return x+self.fn(x)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class SimpleAttentionPool(nn.Module):
    """正确的注意力池化：压缩sequence维度，保持feature维度"""
    def __init__(self, feature_dim):
        """
        feature_dim: go_dimension，即每个GO term的特征维度
        """
        super().__init__()
        # 计算每个GO term的重要性权重
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)  # 输出每个GO term的权重分数
        )
        
    def forward(self, x):
        """
        x: (batch, go_number, feature_dim)
        返回: (batch, feature_dim)
        """
        batch_size, go_number, feature_dim = x.shape
        
        # 1. 计算每个GO term的原始重要性分数
        # x: (batch, go_number, feature_dim) -> 每个GO term的特征
        raw_weights = self.weight_net(x)  # (batch, go_number, 1)
        
        # 2. 在go_number维度上做softmax，得到归一化的注意力权重
        attention_weights = torch.softmax(raw_weights, dim=1)  # (batch, go_number, 1)
        
        # 3. 加权求和：压缩go_number维度
        # attention_weights: (batch, go_number, 1)
        # x: (batch, go_number, feature_dim)
        # 结果: (batch, feature_dim)
        weighted_sum = torch.sum(attention_weights * x, dim=1)
        
        return weighted_sum, attention_weights

# class crossattentionfusion(nn.Module):
#         def __init__(self, go_dim, protein_dim, hidden_dim=512, num_heads=8):
#             super().__init__()
#             self.hidden_dim = hidden_dim      
#             self.go_proj = nn.Linear(go_dim, hidden_dim)
#             self.protein_proj = nn.Linear(protein_dim, hidden_dim)
#             self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,batch_first=True)
#             self.output_proj = nn.Linear(hidden_dim, hidden_dim)
#             self.norm = nn.LayerNorm(hidden_dim)
#             self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4),nn.ReLU(),nn.Linear(hidden_dim*4, hidden_dim))
        
#         def forward(self, go_matrix, protein_matrix):
#             batch_size=protein_matrix.shape[0]
#             go_matrix = go_matrix.unsqueeze(0).expand(batch_size,-1,-1)  ## batch * GO_term_num * go_dim
#             go_matrix = self.go_proj(go_matrix)  # batch * GO_term_num * hidden
#             protein_matrix = self.protein_proj(protein_matrix).unsqueeze(1)  # batch * 1 * hidden_dim
#             fused, _ = self.cross_attention(query=protein_matrix,     # (batch, 1, hidden)
#                         key=go_matrix,  # (batch, n, hidden) 
#             value=go_matrix) # (batch, 1, hidden)
#             fused = fused + protein_matrix
#             fused = self.norm(fused) # (batch, 1, hidden)
#             fused = fused.squeeze(1)  # batch * hidden_dim
#             output = self.ffn(fused)  # batch * hidden_dim
#             return output


class crossattentionfusion(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim),
                                 nn.Dropout(dropout)
                                )


    def forward(self,protein_matrix, go_matrix):
        attention_output, _ = self.attention(query=protein_matrix,key=go_matrix,value=go_matrix)
        x = self.norm1(protein_matrix + attention_output)
        ffn_out = self.ffn(x)
        output = self.norm2(x + ffn_out)
        output = output.squeeze(1)  # batch * dim
        return output


class Combine_Transformer(nn.Module):
    def __init__(self, num_heads, go_context, embedding_vector, device, num_layers=1, dropout=0.1, use_temperature=False):
        super(Combine_Transformer, self).__init__()
        self.heads = num_heads
        self.num_layers = num_layers
        self.use_temperature = use_temperature
        self.go_context = go_context.to(device)  # go_term_num * go_dim
        self.go_embedding_vector = nn.Parameter(embedding_vector.clone().detach())  # go_term_num * go_dim
        hidden_dim = self.go_context.shape[1]

        # fc1 with projection for residual connection (2560 -> hidden_dim)
        self.fc1 = mlpblock(2560, hidden_dim, layer_norm=False, dropout=dropout).to(device)
        self.fc1_5=Residual(mlpblock(hidden_dim, hidden_dim, layer_norm=False, dropout=dropout)).to(device)

        # fc2 with residual connection (hidden_dim -> hidden_dim)
        self.fc2 = Residual(mlpblock(hidden_dim, hidden_dim, layer_norm=False, dropout=dropout)).to(device)

        self.go_bias = nn.Parameter(torch.zeros(self.go_context.shape[0])).to(device)
        self.training = True
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Temperature scaling for better calibration
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # 初始化为1.5
        else:
            self.temperature = 1.0

        # Multi-layer cross-attention
        if num_layers == 1:
            # Single layer (original behavior)
            self.cross_attention_fusion = crossattentionfusion(dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        else:
            # Multiple layers
            self.attention_layers = nn.ModuleList([
                crossattentionfusion(dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])

    def forward(self, protein_vectors):
        # fc1 with projection residual: 2560 -> hidden_dim
        protein_vectors = self.fc1(protein_vectors)  # residual connection
        protein_vectors = self.fc1_5(protein_vectors)

        batch_size = protein_vectors.shape[0]
        go_matrix = self.go_context.unsqueeze(0).expand(batch_size, -1, -1)
        go_embeddings_transformed = self.go_embedding_vector

        # Multi-layer cross-attention
        if self.num_layers == 1:
            # Single layer (fast path)
            protein_matrix = protein_vectors.unsqueeze(1)  # batch * 1 * dim
            go_fusion = self.cross_attention_fusion(protein_matrix=protein_matrix, go_matrix=go_matrix)
        else:
            # Multiple layers with residual connections
            current_features = protein_vectors
            for layer in self.attention_layers:
                protein_matrix = current_features.unsqueeze(1)  # batch * 1 * dim
                layer_output = layer(protein_matrix=protein_matrix, go_matrix=go_matrix)
                # Residual connection
                current_features = current_features + layer_output
            go_fusion = current_features

        # fc2 with residual (already wrapped in Residual class)
        go_fusion = self.fc2(go_fusion)  # batch * hidden_dim

        context_scores = go_fusion @ go_embeddings_transformed.T
        direct_scores = protein_vectors @ go_embeddings_transformed.T
        final_scores = self.alpha * context_scores + (1 - self.alpha) * direct_scores + self.go_bias

        # Apply temperature scaling for better probability calibration
        if self.use_temperature:
            final_scores = final_scores / self.temperature

        output = torch.sigmoid(final_scores)
        return output

### MLP ###
class MLPModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        net = []
        net.append(mlpblock(input_dim, hidden_dim))
        net.append(Residual(mlpblock(hidden_dim, hidden_dim)))
        net.append(nn.Linear(hidden_dim,output_dim))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)
    
    def forward(self,x):
        return self.net(x)
    

