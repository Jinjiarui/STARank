import torch
from torch import nn

from Script.BackModels import PredictMLP


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


if __name__ == '__main__':
    model = SeqBasedModel(16, 32, 48, [32, 16], model_name='GRU')
    x = torch.randint(0, 16, size=(7, 10, 3))
    y_ = model(x)
    print(y_)
