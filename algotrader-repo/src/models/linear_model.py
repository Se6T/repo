from torch import nn

from .base_model import BaseModel


class LinearRegression(BaseModel):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int = 1, 
        ):
    
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLP(BaseModel):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int = 1,
            num_layers: int = 1, 
            hidden_dim: int = 128, 
            dropout: float = 0.3,
            act_fn: nn.Module = nn.SELU()
        ):

        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), act_fn, nn.Dropout(dropout)]  # input layer
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn, nn.Dropout(dropout)])  # hidden layers
        layers.append(nn.Linear(hidden_dim, output_dim))  # output layer
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
