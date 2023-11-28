from torch import nn

from .base_model import BaseModel


class RNN(BaseModel):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int = 1,
            num_layers: int = 1, 
            hidden_dim: int = 128, 
            dropout: float = 0.,
            act_fn: nn.Module = nn.SELU()
        ):

        super(RNN, self).__init__()
        self.act_fn = act_fn
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.act_fn(x)
        
        return self.fc(x)  # dimension [batch_size, seq_length, output_dim]
    

class GRUModel(BaseModel):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int = 1,
            num_layers: int = 1, 
            hidden_dim: int = 128, 
            dropout: float = 0.,
            act_fn: nn.Module = nn.SELU(),
        ):

        super(GRUModel, self).__init__()
        self.act_fn = act_fn
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.act_fn(x)

        return self.fc(x)  # dimension [batch_size, seq_length, output_dim]


class LSTMModel(BaseModel):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int = 1,
            num_layers: int = 1, 
            hidden_dim: int = 128, 
            dropout: float = 0.,
            act_fn: nn.Module = nn.SELU()
        ):

        super(LSTMModel, self).__init__()
        self.act_fn = act_fn
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.act_fn(x)

        return self.fc(x)  # dimension [batch_size, seq_length, output_dim]
