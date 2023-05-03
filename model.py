import copy
import torch
from labml_helpers.module import TypedModuleList, M
from torch import nn
from mha import MultiHeadAttention
from feed_forward import FeedForward


def clone_module_list(module: M, n: int) -> TypedModuleList[M]:
    """
    ## Clone Module

    Make a `nn.ModuleList` with clones of a given module
    """
    return TypedModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerLayer(nn.Module):
    """Transformer Block : multi-head self attention + feedforward"""

    def __init__(self, *, d_model: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention = None,
                 feed_forward: FeedForward, dropout_prob: float):
        super(TransformerLayer, self).__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        """

        :param x: encoder中为source向量， decoder中为target向量
        :param mask: encoder中为source掩码向量， decoder中为target掩码向量
        :param src: decoder中source的编码向量，encoder中为None
        :param src_mask:decoder中source的掩码向量，encoder中为None
        :return:
        """
        # multi-head self attention
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # 加上多头自注意力机制的结果
        x = x + self.dropout(self_attn)

        # decoder模块部分
        if src is not None:
            z = self.norm_src_attn(x)
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(attn_src)

        # feedforward
        z = self.norm_ff(x)
        if self.is_save_ff_input:  # Save the input to the feed forward layer if specified
            self.ff_input = z.clone()
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x


class Encoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super(Encoder, self).__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super(Decoder, self).__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """

        :param x: target，encoder的输入
        :param memory: source经过encoder输出的向量
        :param src_mask: source掩码向量
        :param tgt_mask: target掩码向量
        :return:
        """
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module,
                 generator: nn.Module):
        """

        :param encoder: transformer编码器
        :param decoder: transformer解码器
        :param src_embed: source embedding + position encoder， encoder部分的embedding
        :param tgt_embed: 解码器部分输入的embedding
        :param generator:
        """
        super(Transformer, self).__init__()
        # encoder decoder
        self.encoder = encoder
        self.decoder = decoder
        # embedding layer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

        # self.generator = generator

        # 初始化参数: Glorot / fan_avg
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """

        :param src: encoder input (source)
        :param tgt: decoder input (target)
        :param src_mask: source mask
        :param tgt_mask: target mask
        :return:
        """
        source_embed = self.src_embed(src)
        encode = self.encoder(source_embed, src_mask)
        target_embed = self.tgt_embed(tgt)
        out = self.decoder(target_embed, encode, src_mask, tgt_mask)
        return out