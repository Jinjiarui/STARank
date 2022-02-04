import math

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


class PointerNetWork(nn.Module):
    def __init__(self, input_size, weight_size, hidden_size, allow_repeat=False):
        super(PointerNetWork, self).__init__()
        self.allow_repeat = allow_repeat

        self.enc = nn.Linear(input_size, hidden_size)
        self.dec = nn.LSTMCell(input_size, hidden_size)

        self.W1 = nn.Linear(hidden_size, weight_size, bias=False)  # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)  # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)  # scaling sum of enc and dec by v.T

    def forward(self, targets, history_state):
        # Encoding
        encoder_states = self.enc(targets)  # encoder_state: (B, L, H)
        hidden = history_state  # (B, H)
        blend1 = self.W1(encoder_states)  # (B, L, W)
        # Decoding states initialization
        decoder_input = torch.zeros_like(targets[:, 0])  # (B, I)
        probs = []
        a1 = torch.arange(targets.size(0))
        selected = torch.zeros_like(targets[:, :, 0], dtype=torch.bool)
        minimum_fill = torch.zeros_like(selected, dtype=torch.float32) - 1e9
        # Decoding
        for _ in range(targets.size(1)):
            hidden = self.dec(decoder_input, hidden)
            # Compute blended representation at each decoder time step
            blend2 = self.W2(hidden[0])  # (B, W)
            blend_sum = torch.tanh(blend1 + blend2.unsqueeze(1))  # (B, L, W)
            out = self.vt(blend_sum).squeeze()  # (B, L)
            if not self.allow_repeat:
                out = torch.where(selected, minimum_fill, out)
                selecting = torch.argmax(out, dim=-1)
                selected2 = torch.zeros_like(targets[:, :, 0], dtype=torch.bool)
                selected2[a1, selecting] = True
                selected = selected + selected2
            out = torch.softmax(out, dim=-1).clamp_min(1e-9)
            decoder_input = targets[a1, torch.argmax(out, dim=-1)]
            probs.append(out)
        probs = torch.stack(probs, dim=1)  # (B, L, L)
        return probs


class DIN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(DIN, self).__init__()
        self.ln1 = nn.Linear(input_size, hidden_size)
        self.ln2 = nn.Linear(2 * hidden_size, hidden_size)
        self.activation_layer = nn.Sequential(nn.Linear(3 * hidden_size, 36),
                                              nn.LeakyReLU(),
                                              nn.Dropout(dropout),
                                              nn.Linear(36, 1))

    def forward(self, target, histories):  # (B, 1, I), (B, L, H)
        target = self.ln1(target)
        target = torch.tile(target, [1, histories.size(-2), 1])  # (B, L, I)
        activation_weight = self.activation_layer(torch.concat([histories, target, histories * target], -1))
        histories = torch.sum(histories * activation_weight, dim=1)  # (B, H)

        return self.ln2(torch.cat([histories, target[:, 0]], dim=-1))


class AUGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AUGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dense_layer = nn.ModuleDict()
        self.hidden_num = 1
        self.build()

    @staticmethod
    def get_dense_name():
        return ['xu', 'hu', 'xr', 'hr', 'xg', 'hg']

    def build(self):
        dense_layer_name = self.get_dense_name()
        for i in dense_layer_name:
            input_size = self.input_size if i[0] == 'x' else self.hidden_size
            self.dense_layer[i] = nn.Linear(input_size, self.hidden_size)

    def forward(self, inputs, h):
        h, = h
        inputs, attention = inputs[:, :-1], inputs[:, -1:]
        u = torch.sigmoid(self.dense_layer['xu'](inputs) + self.dense_layer['hu'](h))
        u = u * attention
        r = torch.sigmoid(self.dense_layer['xr'](inputs) + self.dense_layer['hr'](h))
        g = torch.tanh(self.dense_layer['xg'](inputs) + r * self.dense_layer['hg'](h))
        h = (1 - u) * h + u * g
        return h,


class RNN(nn.Module):
    def __init__(self, cell, hidden_size, batch_first=True):
        super(RNN, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.batch_first = batch_first

    def forward(self, inputs, initial_states=None):
        if initial_states is None:
            zeros = torch.zeros(inputs.size(0), self.hidden_size, device=inputs.device)
            initial_states = (zeros,) * self.cell.hidden_num
        if self.batch_first:
            inputs = torch.transpose(inputs, 1, 0)
        states = initial_states
        outputs = []
        for x in inputs:
            h = self.cell(x, states)
            states = h
            outputs.append(h[0])
        return outputs, states


class DIEN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(DIEN, self).__init__()
        self.W = nn.Linear(input_size, hidden_size)
        cell = AUGRUCell(hidden_size, hidden_size)
        self.au_gru = RNN(cell, hidden_size, batch_first)
        self.ln2 = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, targets, history_states):  # (B, 1, I), (B, L, H)
        activation_weight = self.W(targets)  # (B, 1, H)
        activation_weight = torch.tile(activation_weight, [1, history_states.size(-2), 1])  # (B, L, H)
        activation_weight = torch.softmax(torch.sum(activation_weight * history_states, -1, keepdim=True),
                                          1)  # (B, L, 1)
        history_states = torch.concat([history_states, activation_weight], -1)
        _, (h,) = self.au_gru(history_states)
        return self.ln2(torch.cat([h, targets[:, 0]], dim=-1))
