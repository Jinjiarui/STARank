import torch
from torch import nn

from Script.BackModels import PredictMLP, PointerNetWork, DIN, DIEN


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
        self.sub_forward = self.forward_1
        self.use_last = True
        if model_name == 'LSTM':
            self.enc = nn.LSTM(embed_size, hidden_size, batch_first=True)
            self.dec = self.enc
        elif model_name == 'GRU':
            self.enc = nn.GRU(embed_size, hidden_size, batch_first=True)
            self.dec = self.enc
        elif model_name == 'DIN':
            self.enc = nn.Linear(embed_size, hidden_size)
            self.dec = DIN(embed_size, hidden_size, dropout_rate)
        elif model_name == 'DIEN':
            self.enc = nn.GRU(embed_size, hidden_size, batch_first=True)
            self.dec = DIEN(embed_size, hidden_size, batch_first=True)
            self.use_last = False
        elif model_name == 'Pointer':
            self.enc = nn.LSTM(embed_size, hidden_size, batch_first=True)
            self.dec = PointerNetWork(embed_size, hidden_size, hidden_size)
            self.sub_forward = self.forward_2

    def forward_1(self, targets, history):
        history_state = self.enc(history)  # (B, H)
        if isinstance(history_state, tuple):
            history_state = history_state[-1 if self.use_last else 0]
        targets = torch.transpose(targets, 1, 0)  # (L, B, E)
        target_states = []
        for target in targets:
            target_state = self.dec(target[:, None], history_state)  # Input is (B, 1, E)
            if isinstance(target_state, tuple):
                target_state, _ = target_state
            target_states.append(target_state)
        target_states = torch.stack(target_states, dim=1)  # (B, L, H)
        target_states = self.mlp(target_states).squeeze()  # (B, L)
        target_states = torch.sigmoid(target_states)
        return target_states

    def forward_2(self, targets, history):
        _, history_state = self.enc(history)  # (B, H)
        history_state = [_.squeeze(0) for _ in history_state]
        target_states = self.dec(targets, history_state)  # (B, L, L)
        return target_states

    def forward(self, x):
        # X : (B, 2*L, fields)
        x = torch.mean(self.embedding_layer(x), dim=-2)  # (B, 2*L, E)
        history, targets = torch.split(x, dim=1, split_size_or_sections=x.size(1) // 2)  # Both (B, L, E)
        return self.sub_forward(history, targets)


if __name__ == '__main__':
    model = SeqBasedModel(16, 32, 48, [32, 16], model_name='GRU')
    x = torch.randint(0, 16, size=(7, 10, 3))
    y_ = model(x)
    print(y_)
