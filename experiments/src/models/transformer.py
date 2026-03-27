"""
Transformer模型
将Transformer架构适配到房价预测任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """
    Transformer模型配置
    
    核心参数说明:
    - d_model: 模型嵌入维度，越大模型容量越大
    - num_heads: 多头注意力的头数，必须能整除 d_model
    - d_ff: 前馈网络隐藏层维度，通常为 d_model 的 2-4 倍
    - num_layers: Transformer 编码器层数，越深模型表达能力越强
    - dropout: Dropout 概率，用于防止过拟合
    - input_size: 输入特征维度（数据集决定）
    """
    input_size: int = 13
    d_model: int = 32
    num_heads: int = 2
    d_ff: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    
    def __post_init__(self):
        """验证配置参数"""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) 必须能被 num_heads ({self.num_heads}) 整除")
        if self.d_model <= 0:
            raise ValueError(f"d_model 必须大于 0，当前: {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads 必须大于 0，当前: {self.num_heads}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers 必须大于 0，当前: {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，当前: {self.dropout}")
    
    @classmethod
    def small(cls, input_size: int = 13) -> 'TransformerConfig':
        """小型配置 - 快速训练，参数量约 7K"""
        return cls(
            input_size=input_size,
            d_model=32,
            num_heads=2,
            d_ff=64,
            num_layers=1,
            dropout=0.1
        )
    
    @classmethod
    def medium(cls, input_size: int = 13) -> 'TransformerConfig':
        """中型配置 - 平衡性能，参数量约 25K"""
        return cls(
            input_size=input_size,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            dropout=0.15
        )
    
    @classmethod
    def large(cls, input_size: int = 13) -> 'TransformerConfig':
        """大型配置 - 更强表达力，参数量约 80K"""
        return cls(
            input_size=input_size,
            d_model=128,
            num_heads=4,
            d_ff=256,
            num_layers=3,
            dropout=0.2
        )
    
    @classmethod
    def xlarge(cls, input_size: int = 13) -> 'TransformerConfig':
        """超大配置 - 最大容量，参数量约 200K"""
        return cls(
            input_size=input_size,
            d_model=256,
            num_heads=8,
            d_ff=512,
            num_layers=4,
            dropout=0.25
        )
    
    def estimate_params(self) -> int:
        """估算参数量"""
        # 输入投影层
        input_proj = self.input_size * self.d_model + self.d_model
        
        # 每层 Transformer
        # 注意力: QKV投影 + 输出投影 + LayerNorm * 2
        attn_params = (self.d_model * 3 * self.d_model + 3 * self.d_model +  # QKV
                       self.d_model * self.d_model + self.d_model +          # out_proj
                       self.d_model * 2) * 2                                  # LayerNorm * 2
        # FFN
        ffn_params = self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model
        
        layer_params = (attn_params + ffn_params) * self.num_layers
        
        # 输出层
        output_proj = self.d_model + 1
        
        # 位置编码
        pos_encoding = self.d_model
        
        return input_proj + layer_params + output_proj + pos_encoding
    
    def __str__(self) -> str:
        return (f"TransformerConfig(d_model={self.d_model}, num_heads={self.num_heads}, "
                f"d_ff={self.d_ff}, num_layers={self.num_layers}, dropout={self.dropout}, "
                f"~{self.estimate_params():,} params)")


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class TransformerRegressor(nn.Module):
    """
    Transformer回归模型
    将Transformer架构适配到房价预测任务
    
    使用方式:
    1. 使用预设配置:
        model = TransformerRegressor.from_preset('medium')
    
    2. 使用自定义配置:
        config = TransformerConfig(d_model=64, num_heads=4, num_layers=2)
        model = TransformerRegressor(config)
    
    3. 直接传参（向后兼容）:
        model = TransformerRegressor(input_size=13, d_model=32, num_heads=2)
    """

    def __init__(self, 
                 input_size: int = 13, 
                 d_model: int = 32, 
                 num_heads: int = 2,
                 d_ff: int = 64, 
                 num_layers: int = 1, 
                 dropout: float = 0.1,
                 config: Optional[TransformerConfig] = None):
        """
        初始化Transformer回归模型
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            num_layers: Transformer层数
            dropout: Dropout概率
            config: 配置对象（优先使用）
        """
        super().__init__()
        
        # 支持通过 config 对象初始化
        if config is not None:
            self.config = config
        else:
            self.config = TransformerConfig(
                input_size=input_size,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_layers=num_layers,
                dropout=dropout
            )
        
        self.input_size = self.config.input_size
        self.d_model = self.config.d_model

        # 输入投影层
        self.input_projection = nn.Linear(self.input_size, self.d_model)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Transformer编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.config.d_model, 
                self.config.num_heads, 
                self.config.d_ff,
                self.config.dropout
            )
            for _ in range(self.config.num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(self.d_model, 1)

        self.dropout = nn.Dropout(self.config.dropout)

    @classmethod
    def from_preset(cls, preset: str, input_size: int = 13) -> 'TransformerRegressor':
        """
        从预设配置创建模型
        
        Args:
            preset: 预设名称，可选值: 'small', 'medium', 'large', 'xlarge'
            input_size: 输入特征维度
            
        Returns:
            TransformerRegressor 实例
        """
        presets = {
            'small': TransformerConfig.small,
            'medium': TransformerConfig.medium,
            'large': TransformerConfig.large,
            'xlarge': TransformerConfig.xlarge,
        }
        
        if preset not in presets:
            available = list(presets.keys())
            raise ValueError(f"未知预设: {preset}。可用预设: {available}")
        
        config = presets[preset](input_size)
        return cls(config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_model_info(self) -> dict:
        """获取模型信息"""
        actual_params = sum(p.numel() for p in self.parameters())
        return {
            'name': 'Transformer',
            'params': actual_params,
            'config': self.config,
            'description': f'Transformer (d_model={self.config.d_model}, '
                          f'heads={self.config.num_heads}, '
                          f'layers={self.config.num_layers}, '
                          f'params={actual_params:,})'
        }
    
    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 便捷函数
def create_transformer(preset: str = 'small', input_size: int = 13) -> TransformerRegressor:
    """
    快速创建 Transformer 模型
    
    Args:
        preset: 预设名称 ('small', 'medium', 'large', 'xlarge')
        input_size: 输入特征维度
        
    Returns:
        TransformerRegressor 实例
    """
    return TransformerRegressor.from_preset(preset, input_size)


def print_model_comparison():
    """打印不同预设配置的对比"""
    print("\n" + "="*70)
    print("Transformer 预设配置对比")
    print("="*70)
    
    presets = ['small', 'medium', 'large', 'xlarge']
    
    print(f"\n{'预设':<10} {'d_model':<10} {'heads':<8} {'d_ff':<10} {'layers':<8} {'参数量':<12}")
    print("-" * 70)
    
    for preset in presets:
        config = getattr(TransformerConfig, preset)(input_size=13)
        print(f"{preset:<10} {config.d_model:<10} {config.num_heads:<8} "
              f"{config.d_ff:<10} {config.num_layers:<8} {config.estimate_params():>10,}")
    
    print("\n使用示例:")
    print("  model = TransformerRegressor.from_preset('medium')")
    print("  model = TransformerRegressor.from_preset('large', input_size=20)")
    print("  config = TransformerConfig(d_model=64, num_heads=4, num_layers=2)")
    print("  model = TransformerRegressor(config=config)")
    print("="*70 + "\n")


if __name__ == "__main__":
    # 测试不同配置
    print_model_comparison()
    
    print("测试创建模型:")
    for preset in ['small', 'medium', 'large', 'xlarge']:
        model = TransformerRegressor.from_preset(preset)
        info = model.get_model_info()
        print(f"  {preset}: {info['params']:,} params")
