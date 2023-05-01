import torch
from torch import nn


class feedForward(nn.Module):
    """前馈神经网络，通过非线性变换增强模型表示能力"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation=nn.ReLU(), is_gated: bool = False,
                 bias1: bool = True, bias2: bool = True, bias_gate: bool = True):
        super(feedForward, self).__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        """

        :param x: [batch,seq_len,d_model]
        :return: output: [batch,seq_len,d_model]
        """
        # 非线性变换
        g = self.activation(self.layer1(x))

        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g

        x = self.dropout(x)
        output = self.layer2(x)
        return output


# if __name__ == '__main__':
#     x = torch.zeros((32, 20, 100), dtype=torch.float)
#     model = feedForward(d_model=100, d_ff=200)
#     model.forward(x)
