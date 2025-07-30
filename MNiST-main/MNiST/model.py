import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

import torch
import torch.nn as nn

class SpatialAttentionFusion(nn.Module):
    """
    Spatial-aware attention fusion mechanism for combining Conv and ASSM branches
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention for spatial fusion
        self.q_conv = nn.Linear(dim, dim, bias=False)
        self.k_conv = nn.Linear(dim, dim, bias=False)
        self.v_conv = nn.Linear(dim, dim, bias=False)
        
        self.q_assm = nn.Linear(dim, dim, bias=False)
        self.k_assm = nn.Linear(dim, dim, bias=False)
        self.v_assm = nn.Linear(dim, dim, bias=False)
        
        # Cross-attention between branches
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
 
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature alignment
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_conv, x_assm, spatial_pos=None):
        """
        Args:
            x_conv: Conv branch output [B, N, C]
            x_assm: ASSM branch output [B, N, C]
            spatial_pos: Optional spatial position indices [B, N]
        """
        B, N, C = x_conv.shape
        
        # Self-attention within each branch
        x_conv_norm = self.norm1(x_conv)
        x_assm_norm = self.norm1(x_assm)
        
        # Multi-head self-attention for conv branch
        q_conv = self.q_conv(x_conv_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_conv = self.k_conv(x_conv_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_conv = self.v_conv(x_conv_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_conv = torch.matmul(q_conv, k_conv.transpose(-2, -1)) * self.scale
        attn_conv = F.softmax(attn_conv, dim=-1)
        out_conv = torch.matmul(attn_conv, v_conv).transpose(1, 2).reshape(B, N, C)
        
        # Multi-head self-attention for assm branch
        q_assm = self.q_assm(x_assm_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_assm = self.k_assm(x_assm_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_assm = self.v_assm(x_assm_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_assm = torch.matmul(q_assm, k_assm.transpose(-2, -1)) * self.scale
        attn_assm = F.softmax(attn_assm, dim=-1)
        out_assm = torch.matmul(attn_assm, v_assm).transpose(1, 2).reshape(B, N, C)
        
        # Cross-attention between branches
        cross_conv, _ = self.cross_attn(out_conv, out_assm, out_assm)
        cross_assm, _ = self.cross_attn(out_assm, out_conv, out_conv)
        
        # Enhanced features with residual connections
        enhanced_conv = x_conv + self.dropout(cross_conv)
        enhanced_assm = x_assm + self.dropout(cross_assm)
        
        # Dynamic spatial gating
        concat_features = torch.cat([enhanced_conv, enhanced_assm], dim=-1)  # [B, N, dim*2]
        spatial_gates = self.gate_net(concat_features)  # [B, N, 1]
        
        # Adaptive fusion with spatial-aware weights
        fused_output = spatial_gates * enhanced_conv + (1 - spatial_gates) * enhanced_assm
        
        return self.norm2(fused_output), spatial_gates


class AdaptiveSpatialGating(nn.Module):
    """
    Advanced adaptive gating mechanism considering spatial heterogeneity
    """
    def __init__(self, dim, num_clusters=7, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        # Spatial clustering for heterogeneity modeling
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, dim))
        self.cluster_weights = nn.Parameter(torch.ones(num_clusters))
        
        # Spatial context encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_clusters)
        )
        

        self.gate_predictor = nn.Sequential(
            nn.Linear(dim * 2 + num_clusters, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_conv, x_assm):
        """
        Args:
            x_conv: Conv branch features [B, N, C]
            x_assm: ASSM branch features [B, N, C]
        """
        B, N, C = x_conv.shape
        
        # Spatial clustering assignment
        combined_features = x_conv + x_assm  # [B, N, C]
        spatial_logits = self.spatial_encoder(combined_features)  # [B, N, num_clusters]
        spatial_probs = F.softmax(spatial_logits / self.temperature, dim=-1)
        
        # Cluster-aware features
        cluster_features = torch.matmul(spatial_probs, self.cluster_centers)  # [B, N, C]
        

        gate_input = torch.cat([x_conv, x_assm, spatial_probs], dim=-1)  # [B, N, dim*2+num_clusters]
        spatial_gates = self.gate_predictor(gate_input)  # [B, N, 1]
        
        # Adaptive fusion
        fused_features = spatial_gates * x_conv + (1 - spatial_gates) * x_assm
        
        return fused_features, spatial_gates, spatial_probs


class Encoder(nn.Module):
    """
    Enhanced Encoder with spatial-aware attention fusion (maintains original interface)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        graph_neigh,
        dropout: float = 0.0,
        act=F.relu,
        mlp_ratio: float = 2.0,
        num_tokens: int = 64,
        inner_rank: int = 128,
        fusion_type: str = "spatial_attention",  # "spatial_attention" or "adaptive_gating"
        num_heads: int = 8,
        num_clusters: int = 7,
    ):
        super().__init__()
        

        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.fusion_type = fusion_type
        
        # --- GCN weights ---
        self.weight1 = Parameter(torch.empty(in_features, out_features))
        self.weight2 = Parameter(torch.empty(out_features, in_features))
        
        # --- Conv branch ---
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_features, out_features, 3, 1, 1),
            nn.GroupNorm(1, out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, 3, 1, 1),
            nn.GroupNorm(1, out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, 1),
            nn.ReLU(inplace=True),
        )
        
        # --- ASSM branch ---
        self.assm = SpatialMamba(
            dim=out_features,
            d_state=min(out_features, 64), 
            input_resolution=graph_neigh.shape[-1],
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.token_embedding = nn.Embedding(num_tokens, inner_rank)
        nn.init.uniform_(self.token_embedding.weight,
                         -math.sqrt(3 / num_tokens),
                         math.sqrt(3 / num_tokens))
        
        # --- Enhanced fusion mechanisms ---
        if fusion_type == "spatial_attention":
            self.fusion_module = SpatialAttentionFusion(
                dim=out_features,
                num_heads=num_heads,
                dropout=dropout
            )
        elif fusion_type == "adaptive_gating":
            self.fusion_module = AdaptiveSpatialGating(
                dim=out_features,
                num_clusters=num_clusters
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # --- Feature alignment ---
        self.align_conv = nn.Linear(out_features, out_features)
        self.align_assm = nn.Linear(out_features, out_features)
        
        # --- Discriminator components ---
        self.disc = Discriminator(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.xavier_uniform_(self.align_conv.weight)
        nn.init.xavier_uniform_(self.align_assm.weight)
        nn.init.zeros_(self.align_conv.bias)
        nn.init.zeros_(self.align_assm.bias)
    
    def forward(self, feat, feat_a, adj):
        # --- GCN processing ---
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        # --- Reshape for 2D Conv ---
        z = z.unsqueeze(0)  # (1, N, C)
        B, N, C = z.shape
        H = int(math.sqrt(N))
        W = int(math.ceil(N / H))
        if H * W != N:
            H, W = N, 1
        
        # --- Conv branch ---
        z_2d = z.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_conv = self.conv3x3(z_2d)
        x_conv = x_conv.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        
        # --- ASSM branch ---
        token = self.token_embedding.to(feat.device)
        x_assm = self.assm(z, (H, W), token)  # (B, N, C)
        
        # --- Feature normalization and alignment ---
        x_conv = F.layer_norm(x_conv, (C,))
        x_assm = F.layer_norm(x_assm, (C,))
        x_conv = self.align_conv(x_conv)
        x_assm = self.align_assm(x_assm)
        
        # --- Enhanced fusion ---
        if self.fusion_type == "spatial_attention":
            hiden_emb, gate_weights = self.fusion_module(x_conv, x_assm)
        elif self.fusion_type == "adaptive_gating":
            hiden_emb, gate_weights, cluster_probs = self.fusion_module(x_conv, x_assm)
        
        # --- Original discriminator processing ---
        z_no_batch = z.squeeze(0)  # (N, C)
        h = torch.mm(z_no_batch, self.weight2)
        h = torch.mm(adj, h)
        emb = self.act(z_no_batch)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)
        
        g = self.sigm(self.read(emb, self.graph_neigh))
        g_a = self.sigm(self.read(emb_a, self.graph_neigh))
        
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a



class Selective_Scan(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.x_proj = nn.Linear(d_model, d_state * 2, bias=False)
        
        self.A_log = nn.Parameter(torch.randn(d_state))  
        self.Bs = nn.Parameter(torch.randn(d_state, d_model))
        self.Ds = nn.Parameter(torch.ones(d_state))  
        
        self.W_proj = nn.Linear(d_state, d_model, bias=False) 
        self.out_norm = nn.LayerNorm(d_state * 2)  
        
    def forward(self, x, prompt):
        B, N, C = x.shape  

        x_proj = self.x_proj(x)  
        dts, Cs = x_proj.split(self.d_state, dim=-1)  

        A = -torch.exp(self.A_log)
        dt = torch.exp(dts)  
        exp_A = torch.exp(A * dt)  

        prompt_proj = self.W_proj(prompt)  
        prompt_gate = torch.sigmoid(prompt_proj)  

        xs_transposed = x.permute(0, 2, 1)  
        xs_query = xs_transposed * (1 + prompt_gate.permute(0, 2, 1))  

        Bx = (self.Bs @ xs_query).permute(0, 2, 1)  
        updated_state = exp_A * Cs + (1 - exp_A) * Bx
        updated_state = self.Ds * updated_state

       
        if prompt.shape[-1] == self.d_state:
            updated_state = torch.cat([updated_state, prompt], dim=-1)
        else:
       
            prompt_aligned = nn.Linear(prompt.shape[-1], self.d_state, device=prompt.device)(prompt)
            updated_state = torch.cat([updated_state, prompt_aligned], dim=-1)

        updated_state = self.out_norm(updated_state)
        return updated_state


class SpatialMamba(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state

        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Conv2d(self.dim, hidden, 1, 1, 0)
        self.CPE = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H = W = int(math.sqrt(n))
        if H * W != n:
            H, W = n, 1 

        full_embedding = torch.matmul(self.embeddingB.weight, token.weight.T)
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        x = x.permute(0, 2, 1)  
        x = x.view(B, C, H, W).contiguous()  
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)

        y = self.selectiveScan(x, prompt)
        y = self.out_proj(self.out_norm(y))
        return y


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
        return F.normalize(global_emb, p=2, dim=1)

    
class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
         
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a =self.sigm(g_a)       
       
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a     

class Encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout
        
        #self.linear1 = torch.nn.Linear(self.dim_input, self.dim_output)
        #self.linear2 = torch.nn.Linear(self.dim_output, self.dim_input)
        
        #self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim_output))
        #self.weight1_de = Parameter(torch.FloatTensor(self.dim_output, self.dim_input))
        
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))
        
        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
      
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        
        #x = self.linear1(x)
        #x = self.linear2(x)
        
        #x = torch.mm(x, self.weight1_en)
        #x = torch.mm(x, self.weight1_de)
        
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        
        return x
    
class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x 
