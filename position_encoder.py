"""
位置编码
"""
import math
import torch
from torch import nn


def get_positional_encoding(d_model: int, max_len: int = 5000):
    """

    :param d_model: embedding dim
    :param max_len: seq_len
    :return: max_len,1 , d_model
    """
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)  # max_len, d_model
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # max_len, 1
    # $2 * i$, 偶数
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)  # max_len/2
    # 奇数
    one_i = torch.arange(1, d_model, 2, dtype=torch.float32)

    # $10000^{\frac{2i}{d_{model}}}$
    two_div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    one_div_term = torch.exp(one_i * -(math.log(10000.0) / d_model))

    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * two_div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * one_div_term)

    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)

    return encodings


class EmbeddingsWithPositionalEncoding(nn.Module):
    """固定不变的位置编码器"""

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        """

        :param d_model: 词嵌入维度
        :param n_vocab: 词表大小
        :param max_len: 序列长度
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

        # register_buffer定义的参数不被更新
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        """

        :param x: shape: [batch,max_len]
        :return: output: [batch,max_len,d_model]
        """
        x = torch.permute(x, (1, 0))
        embed = self.embedding(x)  # batch,max_len,d_model
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)  # max_len,1,d_model
        output = embed * math.sqrt(self.d_model) + pe
        output = torch.permute(output, (1, 0, 2))
        return output


class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
    """可学习的位置编码器"""

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        """

        :param d_model: 词嵌入维度
        :param n_vocab: 词表大小
        :param max_len: 序列长度
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """

        :param x: shape: batch,max_len
        :return: output: batch,max_len,d_model
        """
        x = torch.permute(x, (1, 0))
        embed = self.embedding(x)
        pe = self.positional_encodings[:x.shape[0]]
        output = embed * math.sqrt(self.d_model) + pe
        output = torch.permute(output, (1, 0, 2))
        return output


# if __name__ == '__main__':
#     encoder = EmbeddingsWithPositionalEncoding(d_model=9, n_vocab=1000, max_len=20)
#     encoder2 = EmbeddingsWithLearnedPositionalEncoding(d_model=100, n_vocab=1000, max_len=20)
#     inputs = torch.zeros((32, 20), dtype=torch.long)
#     encoder(inputs)
#     encoder2(inputs)
