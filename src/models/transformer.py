"""
Transformer模型
将Transformer架构适配到房价预测任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model=32, num_heads=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 生成Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # 注意力加权
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        return self.out_proj(context)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model=32, d_ff=64):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model=32, num_heads=2, d_ff=64):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class TransformerRegressor(nn.Module):
    """
    Transformer回归模型
    将Transformer架构适配到房价预测任务
    """

    def __init__(self, input_size=13, d_model=32, num_heads=2,
                 d_ff=64, num_layers=1, dropout=0.1):
        """
        初始化Transformer回归模型

        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            num_layers: Transformer层数
            dropout: Dropout概率
        """
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_size]

        Returns:
            预测值 [batch_size, 1]
        """
        # 投影到d_model维度
        x_proj = self.input_projection(x).unsqueeze(1)  # [batch, 1, d_model]

        # 添加位置编码
        x_proj = x_proj + self.pos_encoding
        x_proj = self.dropout(x_proj)

        # 通过Transformer层
        for layer in self.layers:
            x_proj = layer(x_proj)

        # 输出预测
        output = self.output_layer(x_proj.squeeze(1))
        return output

    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'Transformer',
            'params': sum(p.numel() for p in self.parameters()),
            'description': f'Transformer (d_model={self.d_model}, heads={len(self.layers)})'
        }
