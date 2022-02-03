from unicodedata import bidirectional
import torch
import torch.nn.functional as F

from torch import nn
from Script.BackModels import PredictMLP, MultiHeadedAttention, ContextAttention


class BaseModel(nn.Module):
    def __init__(self,
                 input_size,
                 embed_size,
                 hidden_size,
                 model):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.model = model
        self.embedding_layer = nn.Embedding(input_size, embed_size)


class SeqBasedModel(BaseModel):
    def __init__(self,
                 input_size,
                 embed_size,
                 hidden_size,
                 predict_hidden_sizes,
                 output_size=1,
                 dropout_rate=0.5,
                 model_name='LSTM',
                 ):
        super(SeqBasedModel, self).__init__(input_size, embed_size, hidden_size, model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = PredictMLP(hidden_size, predict_hidden_sizes, output_size, self.dropout)
        if model_name == 'LSTM':
            self.back_model = nn.LSTM(embed_size, hidden_size, batch_first=True)
        elif model_name == 'GRU':
            self.back_model = nn.GRU(embed_size, hidden_size, batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        # X : (B, 2*L, fields)
        x = torch.mean(self.embedding_layer(x), dim=-2)  # (B, 2*L, E)
        history, targets = torch.split(x, dim=1, split_size_or_sections=x.size(1) // 2)  # Both (B, L, E)
        _, history_state = self.back_model(history)  # (B, H)
        targets = torch.transpose(targets, 1, 0)  # (L, B, E)
        target_states = []
        for target in targets:
            target_state, _ = self.back_model(target[:, None], history_state)  # Input is (B, 1, E)
            target_states.append(target_state)
        target_states = torch.stack(target_states, dim=1)  # (B, L, H)
        target_states = self.mlp(target_states).squeeze()  # (B, L)
        target_states = torch.sigmoid(target_states)
        return target_states


class Encoder(BaseModel):
    def __init__(
        self, 
        input_size,  
        embed_size, 
        hidden_size,
        predict_hidden_sizes, 
        activation,
        model_name='att',    # options: att, lstm, gru
        n_layers=1,  # number of layers for encoding feat
        n_heads=3,
        input_drop=0.0,
        dropout=0.0,
        output_size=1,
        norm='none',
        use_bidir=False
        ):
        super().__init__(input_size, embed_size, hidden_size)
        if n_layers == 0:
            predict_hidden_sizes = input_size
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            in_hidden = predict_hidden_sizes if i > 0 else input_size
            out_hidden = predict_hidden_sizes
            self.linears.append(nn.Linear(in_hidden, out_hidden))
            if i < n_layers - 1:
                if norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(out_hidden))
                elif norm == 'layer':
                    self.norms.append(nn.LayerNorm(out_hidden))
                else:
                    self.norms = None

        if model_name == 'att':
            self.back_model = MultiHeadedAttention(n_heads, predict_hidden_sizes, dropout_rate=dropout)
        elif model_name == 'lstm':
            self.back_model = nn.LSTM(embed_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=use_bidir)
        elif model_name == 'gru':
            self.back_model = nn.GRU(embed_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=use_bidir)
        else:
            raise NotImplementedError

        self.activation = activation
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, hidden):
        inputs = inputs.permute(1, 0, 2)
        outputs, hidden = self.back_model(inputs, hidden)

        return outputs.permute(1, 0, 2), hidden
        

class Decoder(BaseModel):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        ):
        super().__init__(input_size, hidden_size)

        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

        self.attention = ContextAttention(hidden_size, hidden_size)
        self.mask = nn.Parameter(torch.ones(1), requires_grad=False)
        self.runner = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embeded_inputs, decoder_input, hidden, context):
        """
        embeded_inputs: inputs of Pointer-Network: (batch, seq_len)
        decoder_input: first decoder's input
        hidden: first decoder's hidden state
        context: encoder's outputs
        """
        batch_size, seq_len = embeded_inputs.size(0), embeded_inputs.size(1)
        mask = self.mask.repeat(seq_len).unsqueeze(0).repeat(batch_size) # (batch, seq_len)
        self.attention.set_mask(mask.size())

        # Generate arang(seq_len), broadcasted across batch
        runner = self.runner.repeat(seq_len)
        for i in range(seq_len):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs, pointers = [], []
        
        def step(x, hidden):
            """
            x: input at position i
            hidden: hidden state at i-1
            return: hidden states at time c, attention score
            """
            # Following is LSTM + ATT
            h, c = hidden
            _gates = self.input2hidden(x) + self.hidden2hidden(h)
            _input, _forget, _cell, _out = _gates.chunk(4, 1) 
            _input = F.sigmoid(_input)
            _forget = F.sigmoid(_forget)
            _cell = F.sigmoid(_cell)
            _out = F.sigmoid(_out)
            c_i = (_forget * c) + (_input * _cell)
            h_i = _out * F.tanh(c_i)

            hidden_i, output = self.attention(h_i, context, torch.eq(mask, 0))
            hidden_i = F.tanh(self.hidden2out(torch.cat((hidden_i, h_i), 1)))

            return (hidden_i, c_i), output
        
        for _ in range(seq_len):
            hidden, output = step(decoder_input, hidden)

            # mask selected inputs
            masked_output = output * mask
            _, indices = masked_output.max(1)
            pointer = (runner == indices.unsqueeze(1).expand(-1, output.size()[1])).float()
            mask = mask * (1 - pointer)

            embed_mask = pointer.unsqueeze(2).expand(-1, -1, self.hidden_size).byte()
            decoder_input = embeded_inputs[embed_mask.data].view(batch_size, self.hidden_size)
            outputs.append(output.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
        
        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointer, 1)

        return (outputs, pointer), hidden


class PLNet(nn.Module):
    """
    Plackett-Luce Module: Pointer Network
    """
    def __init__(self):
        super(PLNet, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()



if __name__ == '__main__':
    model = SeqBasedModel(16, 32, 48, [32, 16], model_name='GRU')
    x = torch.randint(0, 16, size=(7, 10, 3))
    y_ = model(x)
    print(y_)