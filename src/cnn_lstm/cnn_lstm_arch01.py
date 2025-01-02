import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, amp_output_features, phase_output_features, lstm_hidden_dim, 
                 lstm_num_layers, bidirectional, num_classes=8, num_antennas=3, 
                 num_subcarriers=30, num_time_slices=10):
        super(CNNLSTMModel, self).__init__()
        self.amp_output_features = amp_output_features
        self.phase_output_features = phase_output_features
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.num_time_slices = num_time_slices
        self.num_classes = num_classes

        # Conv1D for amplitude
        self.amp_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, amp_output_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Conv1D for phase
        self.phase_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, phase_output_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=amp_output_features + phase_output_features,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attn_weight_layer = nn.Linear(
            lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim, 1, bias=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(
            lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim, num_classes
        )
    
    def forward(self, x):
        """
        x: [batch_size, num_time_slices, num_antennas*num_subcarriers*2]
        """
        num_antennas = self.num_antennas
        num_subcarriers = self.num_subcarriers
        num_features = num_antennas * num_subcarriers
        num_time_slices = self.num_time_slices
        batch_size = x.shape[0]
        
        # Separate amplitude and phase
        amplitude = x[:, :, :num_features].unsqueeze(3)
        phase = x[:, :, num_features:].unsqueeze(3)

        # Process amplitude through amp_conv
        amplitude = amplitude.permute(0, 1, 3, 2)  # Reshape to [batch_size, num_time_slices, 1, num_features]
        amplitude = amplitude.reshape(-1, 1, num_features)  # Merge batch and time for parallel processing
        amp_out = self.amp_conv(amplitude)  # Shape: [batch_size * num_time_slices, amp_output_features, reduced_features]
        amp_out = amp_out.mean(dim=-1)      # Global pooling along feature dimension
        amp_out = amp_out.view(batch_size, num_time_slices, -1)  # Reshape back to [batch_size, num_time_slices, amp_output_features]

        # Process phase through phase_conv
        phase = phase.permute(0, 1, 3, 2)  # Reshape to [batch_size, num_time_slices, 1, num_features]
        phase = phase.reshape(-1, 1, num_features)  # Merge batch and time for parallel processing
        phase_out = self.phase_conv(phase)  # Shape: [batch_size * num_time_slices, phase_output_features, reduced_features]
        phase_out = phase_out.mean(dim=-1)  # Global pooling along feature dimension
        phase_out = phase_out.view(batch_size, num_time_slices, -1)  # Reshape back to [batch_size, num_time_slices, phase_output_features]

        # Concatenate amplitude and phase outputs
        combined_out = torch.cat([amp_out, phase_out], dim=-1)  # Shape: [batch_size, num_time_slices, amp_output_features + phase_output_features]

        # Pass through LSTM
        lstm_out, _ = self.lstm(combined_out)  # Shape: [batch_size, num_time_slices, lstm_hidden_dim * (2 if bidirectional else 1)]

        # Attention mechanism
        attn_scores = self.attn_weight_layer(lstm_out).squeeze(-1)  # Shape: [batch_size, num_time_slices]
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize scores with softmax
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)  # Weighted sum of LSTM outputs

        # Fully connected layer for classification
        output = self.fc(context_vector)  # Shape: [batch_size, num_classes]
        return output