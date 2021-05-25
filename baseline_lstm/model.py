import torch.nn as nn

class SentimentClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(SentimentClassificationModel, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear_layer = nn.Linear(in_features=hidden_size, out_features=5)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        out, _ = self.lstm_layer(x)
        hidden = self.dropout(out[:, -1, :])
        print(out.shape, hidden.shape)
        out = self.linear_layer(hidden)
        return out, hidden

    def linear_forward(self, h):
        out = self.linear_layer(h)
        return out