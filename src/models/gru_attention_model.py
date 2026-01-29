# src/models/gru_model_with_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module): 
    
    def __init__(self, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention weights
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        
        # Dropout for attention scores
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, gru_output: torch.Tensor) -> tuple: 
        scores = self.attention_weights(gru_output)
        
        # Apply softmax to get attention weights
        # (batch, seq_len, 1) -> (batch, seq_len, 1)
        attention_weights = F.softmax(scores, dim=1)
        
        # Apply dropout to attention weights (regularization)
        attention_weights = self.dropout(attention_weights)
        
        # Compute context as weighted sum
        # (batch, seq_len, hidden) * (batch, seq_len, 1) -> (batch, seq_len, hidden)
        context = torch.sum(attention_weights * gru_output, dim=1)
        
        return context, attention_weights.squeeze(-1)


class GRUModelWithAttention(nn.Module): 
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True,
        attention_dropout: float = 0.3,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        self.actual_hidden_size = hidden_size * self.num_directions
        
        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.attention = AttentionLayer(
            hidden_size=self.actual_hidden_size,
            dropout=attention_dropout
        )

        self.batch_norm = nn.BatchNorm1d(self.actual_hidden_size)

        self.fc1 = nn.Linear(self.actual_hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
            elif "fc" in name and "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "attention" in name and "weight" in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):

        # GRU forward pass
        # gru_out: (batch, seq_len, hidden_size * num_directions)
        gru_out, h_n = self.gru(x)

        # context: (batch, hidden_size * num_directions)
        # attn_weights: (batch, seq_len)
        context, attention_weights = self.attention(gru_out)

        context = self.batch_norm(context)

        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        logits = out.squeeze(-1)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_weights(self, x: torch.Tensor):

        self.eval()
        with torch.no_grad():
            gru_out, _ = self.gru(x)
            _, attention_weights = self.attention(gru_out)
        return attention_weights


 