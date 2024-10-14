import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type  # 时间步嵌入类型
        self.time_emb_dim = emb_size  # 时间步嵌入维度
        self.norm = norm  # 是否对输入进行归一化

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        # 将时间步嵌入与输入拼接
        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        # 定义输入层的线性层列表（用于逐层传播）
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        # 定义输出层的线性层列表
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        # 初始化输入层的权重和偏置
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]  # 输出维度
            fan_in = size[1]  # 输入维度
            std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier 初始化标准差
            layer.weight.data.normal_(0.0, std)  # 权重正态分布初始化
            layer.bias.data.normal_(0.0, 0.001)  # 偏置正态分布初始化

        # 初始化输出层的权重和偏置
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        # 初始化嵌入层的权重和偏置
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        # 对时间步进行嵌入
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建用于时间步的正弦嵌入。

    :param timesteps: 一个 1D 的 Tensor，包含每个批次元素的时间步。
    :param dim: 嵌入的维度。
    :param max_period: 控制嵌入的最小频率。
    :return: 一个 [N x dim] 形状的时间步嵌入张量。
    """

    half = dim // 2
    # 生成频率向量
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    # 将时间步乘以频率，生成正弦和余弦嵌入
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # 如果嵌入维度为奇数，则补充一个零张量
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
