import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_adj
from evaluation import *


class mlpblock(nn.Module):
    def __init__(self,input_dim=2560, output_dim=1024,bias=True,layer_norm=False,dropout=0.1,activation=nn.ReLU()):
        super().__init__()       
        self.linear = nn.Linear(input_dim, output_dim,bias)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
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
        self.dropout = nn.Dropout(p=0.5)
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

class Attention_layer(nn.Module):
    def  __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels),
        )
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        ### 输入：GO_term_num * 200 (无batch维度，因为go_features是共享的)
        ### 输出：GO_term_num * GO_term_num
        if x.dtype == torch.float32:
            x = x.float()
        Z = self.model(x) ## GO_term_num * 200
        # 修复：正确转置最后两个维度
        score = torch.matmul(Z, Z.transpose(-2, -1))  # GO_term_num * GO_term_num
        # 添加温度缩放防止softmax饱和
        temperature = Z.size(-1) ** 0.5
        score = score / temperature
        W = F.softmax(score, dim=-1).to(torch.float32) # 归一化为0-1权重
        return W

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
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), 
                                 nn.GELU(),
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
    def __init__(self, num_heads,go_context,embedding_vector,device,go_embedding_dim=768):
        super(Combine_Transformer, self).__init__()
        self.heads = num_heads
        self.go_context=go_context.to(device)  # go_term_num * go_dim
        self.go_embedding_vector=nn.Parameter(embedding_vector.clone().detach()) # go_term_num * go_dim
        hidden_dim = self.go_context.shape[1]
        self.fc2 = mlpblock(hidden_dim,hidden_dim,layer_norm=False,dropout=0.1).to(device)
        self.fc1 = mlpblock(2560,hidden_dim,layer_norm=False,dropout=0.1).to(device)
        self.go_bias = nn.Parameter(torch.zeros(self.go_context.shape[0])).to(device)  # go_term_num
        self.training = True
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        self.cross_attention_fusion = crossattentionfusion(dim=hidden_dim, num_heads=num_heads)  

    def forward(self, protein_vectors):
        protein_vectors = self.fc1(protein_vectors) ## batch * hidden_dim
        batch_size=protein_vectors.shape[0]
        protein_matrix = protein_vectors.unsqueeze(1)  # batch * 1 * dim
        go_embeddings_transformed = self.go_embedding_vector
        # GO context merge
        go_matrix = self.go_context.unsqueeze(0).expand(batch_size, -1, -1)
        go_fusion = self.cross_attention_fusion(protein_matrix=protein_matrix, go_matrix=go_matrix) # batch * hidden_dim
        ## 2. GO term 特征变换
        go_fusion = self.fc2(go_fusion)  # go_term_num * hidden_dim
        ## 3. 蛋白质特征与GO term特征融合

        context_scores = go_fusion @ go_embeddings_transformed.T
        # project_protein = self.fc2(protein_vectors)
        direct_scores = protein_vectors @ go_embeddings_transformed.T+self.go_bias  # batch * go_term_num
        final_scores = self.alpha * context_scores + (1 - self.alpha) * direct_scores + self.go_bias
        go_fusion = torch.sigmoid(final_scores)
        return go_fusion

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
    

