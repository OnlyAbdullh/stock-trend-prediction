

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        self.actual_hidden_size = hidden_size * self.num_directions

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.fc1 = nn.Linear(self.actual_hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
            elif name.startswith("fc") and "weight" in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        gru_out, h_n = self.gru(x)
        # h_n: (num_layers*num_directions, batch, hidden_size)

        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            context = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden)
        else:
            context = h_n[-1, :, :]

        out = self.fc1(context)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out.squeeze(-1)                 # (batch,)


