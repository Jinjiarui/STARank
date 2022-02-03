import math
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class PredictMLP(nn.Module):
    def __init__(self, input_size, predict_hidden_sizes, output_size, dropout):
        super(PredictMLP, self).__init__()
        self.dropout = dropout
        self.activate = nn.LeakyReLU()

        self.linear_layers = nn.Sequential(
            self.dropout,
            nn.Linear(input_size, predict_hidden_sizes[0]),
            self.activate)
        for i in range(1, len(predict_hidden_sizes)):
            self.linear_layers.add_module('{}_dropout'.format(i), self.dropout)
            self.linear_layers.add_module('{}_linear'.format(i),
                                          nn.Linear(predict_hidden_sizes[i - 1], predict_hidden_sizes[i]))
            self.linear_layers.add_module('{}_activate'.format(i), self.activate)
        self.fc = nn.Linear(predict_hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.linear_layers(x)
        x = self.fc(x)
        return x


def get_output_mask(real_len, label_len):
    batch_size = len(real_len)
    max_len = torch.max(real_len)
    seq_range_expand = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = torch.unsqueeze(real_len, 1).expand_as(seq_range_expand)
    label_length_expand = seq_length_expand - label_len
    out_mask = torch.logical_and(
        torch.less(seq_range_expand, seq_length_expand),
        torch.greater_equal(seq_range_expand, label_length_expand)
    )
    return out_mask


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, input_size, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert input_size % head == 0
        self.d_k = input_size // head
        self.head = head
        self.linears = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(dropout_rate)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask, None)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    def __init__(self, head, input_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.mh = MultiHeadedAttention(head, input_size, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.activate = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.ln2 = nn.LayerNorm(input_size)

    def forward(self, s, mask):
        s = s + self.dropout(self.mh(s, s, s, mask))
        s = self.ln1(s)
        s_ = self.activate(self.fc1(s))
        s_ = self.dropout(self.fc2(s_))
        s = self.ln2(s + s_)
        return s


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=1, b=1, batch_first=True):
        super(Transformer, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.SAs = nn.ModuleList([MultiHeadedAttention(head, hidden_size, dropout_rate) for _ in range(b)])
        self.FFNs = nn.ModuleList([FeedForward(head, hidden_size, dropout_rate) for _ in range(b)])
        self.b = b
        self.batch_first = batch_first

    def forward(self, inputs, label_len):
        inputs = rnn_utils.PackedSequence(self.fc(inputs.data), inputs.batch_sizes)
        inputs, real_len = rnn_utils.pad_packed_sequence(inputs, batch_first=self.batch_first)
        batch_size = inputs.size(0)
        max_len = torch.max(real_len)
        transformer_mask = torch.tril(torch.ones((1, max_len, max_len), device=inputs.device))
        out_mask = get_output_mask(real_len, label_len)
        for i in range(self.b):
            inputs = self.SAs[i](inputs, inputs, inputs, transformer_mask)
            inputs = self.FFNs[i](inputs, transformer_mask)
        return inputs[out_mask].view((batch_size, label_len, -1))


class ContextAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        input_layers=1,
        context_layers=1,
        context_type='conv',    # conv, mlp
        norm='none',
        input_drop=0.0,
        dropout=0.0
    ):
        super(ContextAttention, self).__init__()
        self.input_layers = input_layers
        self.context_layers = context_layers
        
        self.inputs = nn.ModuleList()
        self.contexts = nn.ModuleList()
        if norm != 'none':
            self.inputs_norms = nn.ModuleList()
            self.context_norms = nn.ModuelList()
        
        for i in range(input_layers):
            in_hidden = hidden_size if i > 0 else input_size
            out_hidden = hidden_size
            self.inputs.append(nn.Linear(in_hidden, out_hidden)) 
            if i < input_layers - 1:
                if norm == 'batch':
                    self.inputs_norms.append(nn.BatchNorm1d(out_hidden))
                elif norm == 'layer':
                    self.inputs_norms.append(nn.LayerNorm(out_hidden))
                else:
                    self.inputs_norms = None

        for i in range(context_layers):
            in_hidden = hidden_size if i > 0 else input_size
            out_hidden = hidden_size
            if context_type == 'conv':
                self.contexts.append(nn.Conv1d(in_hidden, out_hidden, 1, 1))
            else:
                self.contexts.append(nn.Linear(in_hidden, out_hidden))
            if i < context_layers - 1:
                if norm == 'batch':
                    self.context_norms.append(nn.BatchNorm1d(out_hidden))
                elif norm == 'layer':
                    self.context_norms.append(nn.LayerNorm(out_hidden))
                else:
                    self.context_norms = None
        
        self.V = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(self.V)

        self._mask = nn.Parameter(torch.FloatTensor([float['-inf']]), requires_grad=False)
        self._tanh = nn.Tanh()
        self._softmax = nn.Softmax()

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)

    def set_mask(self, mask_size):
        self.mask = self._mask.unsqueeze(1).expand(*mask_size)

    def forward(self, inputs, contexts, mask):
        """
        mask: Byte tensor: selection mask
        """
        inputs = self.input_drop(inputs)

        for i in range(self.input_layers):
            inputs = self.inputs[i](inputs)
            if i < self.input_layers - 1:
                if self.inputs_norms:
                    inputs = self.inputs_norms[i](inputs)
            inputs = self.dropout(inputs)
        
        inputs = inputs.unsqueeze(2).expand(-1, -1, contexts.size(1))    # (batch, hidden_dim, seq_len)
        contexts = contexts.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        
        for i in range(self.context_layers):
            contexts = self.contexts[i](contexts)
            if i < self.context_layers - 1:
                if self.context_norms:
                    contexts = self.context_norms[i](contexts)
            contexts = self.dropout(contexts)

        V = self.V.unsqueeze(0).expand(contexts.size(0), -1).unsqueeze(1)    # (batch, 1, hidden_dim)
        att = torch.bmm(V, self._tanh(inputs + contexts)).squeeze(1)  # (batch, seq_len)
        if att[mask].size(1) > 0:
            att[mask] = self.mask(mask)
        score = self._softmax(att)
        hidden = torch.bmm(contexts, score.unsqueeze(2)).squeeze(2)

        return hidden, score


