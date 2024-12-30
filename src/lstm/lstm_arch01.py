import torch
import torch.nn as nn

class LSTMBasedModel(nn.Module):
    def __init__(self,   hidden_dim, num_layers, num_classes, bidirectional=False, dropout=0.3, num_antennas=3, num_subcarriers=30, num_time_slices=10):
        super(LSTMBasedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.num_time_slices = num_time_slices
        input_dim = num_antennas * num_subcarriers * 2

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )

        # Fully connected layer for classification
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x):
        """
        x: [batch_size, num_time_slices, num_antennas*num_subcarriers*2]
        """
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, num_time_slices, hidden_dim]
        
        # Use the output from the final time step
        final_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        # Pass through fully connected layer
        logits = self.fc(final_out)  # [batch_size, num_classes]
        return logits