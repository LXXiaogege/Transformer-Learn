"""
多头注意力机制
"""
import math
from typing import Optional, List
import torch
from torch import nn


class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        """
        划分多头
        :param d_model: 原始嵌入维度
        :param heads: 头的个数
        :param d_k: 没个头的嵌入维度
        :param bias: 线性变换偏置
        """
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        """
        先做一个线性变换，再划分成多个头
        :param x: [batch, seq_len, d_model] 或 [batch,d_model]
        :return: [seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        """
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool, dropout_prob: float = 0.1):
        """

        :param d_model: 原先的嵌入维度
        :param heads: 头的个数
        :param d_k: 每个头的维度
        :param bias: 划分多头时，线性变换偏置
        :param dropout_prob:
        """
        super().__init__()
        self.heads = heads
        self.d_k = d_k

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.scale = 1 / math.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)

        # 将注意力得分另存为，可以用于logging输出或者其他计算
        self.attn = None
        # 输出层
        self.output = nn.Linear(heads * d_k, d_model)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        :param mask: [batch_size,seq_len_q, seq_len_k]
        :param query_shape:
        :param key_shape:
        :return:  [batch_size, seq_len_q, seq_len_k, heads]， 非0即1
        """

        # 不满足以下条件则报错
        assert mask.shape[1] == 1 or mask.shape[1] == query_shape[1]
        assert mask.shape[2] == key_shape[1]
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        return mask

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        计算queries和keys之间的分数（可以理解为匹配度），注意力分数 attention scores
        不同的注意力机制有不同的实现方法，可以根据实际情况重写这个方法
        """
        return torch.einsum('bihd,bjhd->bijh', query, key)  # 爱因斯坦求和更节约内存

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """

        :param query: [batch,seq_len,d_model]
        :param key: [batch,seq_len,d_model]
        :param value:[batch,seq_len,d_model]
        :param mask:
        :return:
        """
        batch_size, seq_len, d_model = query.shape  # batch, seq_len, d_model

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # 计算注意力分数attention scores,shape: [batch, seq_len_q, seq_len_k, heads]
        scores = self.get_scores(query, key)
        # 根号缩放
        scores *= self.scale

        if mask is not None:  # mask
            scores = scores.masked_fill(mask == 0, float('-inf'))  # mask==0来转为bool型
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        # 乘value, [batch, seq_len_q, heads, d_k]
        x = torch.einsum("bijh,bjhd->bihd", attn, value)

        # 另存为attn
        self.attn = attn.detach()

        # 重新拼接多个头, [batch, seq_len_q, heads*d_k]
        x = x.reshape(batch_size, seq_len, -1)
        # 线性变换，[batch, seq_len_q, d_model]
        output = self.output(x)
        return output


# if __name__ == '__main__':
#     q = k = v = torch.zeros((32, 20, 100), dtype=torch.float)
#     mask = torch.ones((32, 20, 20), dtype=torch.long)
#     model = MultiHeadAttention(d_model=100, heads=3, d_k=50, bias=False, dropout_prob=0.2)
#     model.forward(query=q, key=k, value=v, mask=mask)
