import math
import torch
from torch import nn
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """
    X: [batch_size, seq_len, num_hiddens]
    return [batch_size * num_heads, seq_len, num_hiddens / num_heads]
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_outputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.multi_head_attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=2000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class TransformerEncoder(nn.Module):
    def __init__(self, num_hiddens, norm_shape, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), EncoderBlock(
                num_hiddens, norm_shape, ffn_num_hiddens, num_heads, dropout, use_bias))
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, X):
        X = self.pos_encoding(X)
        # 可视化
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.multi_head_attention.attention.attention_weights
        return X


class Transformer(nn.Module):
    def __init__(self, input_dim, seq_length, num_hiddens, ffn_num_hiddens, num_heads, num_blks,
                dropout, use_bias=False, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(input_dim, num_hiddens, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm1d(num_hiddens)
        self.relu = nn.ReLU()
        self.transformer_encoder = TransformerEncoder(num_hiddens, [seq_length, num_hiddens], 
                ffn_num_hiddens, num_heads, num_blks, dropout, use_bias)
        self.linear1 = nn.Linear(num_hiddens * seq_length, 2)
        self.linear2 = nn.Linear(num_hiddens, num_hiddens)

    def forward(self, input):
        output = self.conv1(input)
        # output = self.bn(output)
        # output = self.relu(output)
        output = output.transpose(1, 2)
        output = self.transformer_encoder(output)
        # output = output.mean(dim=1)
        output = torch.flatten(output, start_dim=1)
        # output = self.linear1(self.relu(self.linear2(output)))
        output = self.linear1(output)
        return output


if __name__ == '__main__':
    input = torch.randn(128, 32, 290)
    model = Transformer(32, 290, 512, 2048, 8, 6, 0.1, True)
    output = model(input)