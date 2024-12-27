import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self,src):
        output = self.transformer_encoder(src)
        return output

class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalEncoder,self).__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=3,padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.mean(dim=2)
        return x
    

class CSI2HARModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, num_decoder_layers, 
                 num_antennas=3, num_subcarriers=30, num_time_slices=10, num_classes=8):
        super(CSI2HARModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.num_time_slices = num_time_slices
        self.num_heads = num_heads

        self.temporal_encoder = TemporalEncoder(in_channels=2, out_channels=embedding_dim)

        #Positional encodings
        self.antenna_embeddings = nn.Embedding(num_antennas, embedding_dim)
        self.subcarrier_embeddings = nn.Embedding(num_subcarriers, embedding_dim)

        self.encoder = TransformerEncoder(embedding_dim, num_heads, num_encoder_layers)

        # Improved classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),            # Flatten for dense layers
            nn.Linear(embedding_dim, 128),  # First dense layer with reduced dimension
            nn.ReLU(),               # Non-linear activation
            nn.Dropout(p=0.3),       # Dropout for regularization
            nn.Linear(128, num_classes)  # Final layer for class logits
        )

    def forward(self, wifi_csi_frame):
        # wifi_csi_frame shape: [batch_size, num_time_slices, num_antennas*num_subcarriers*2]
        batch_size = wifi_csi_frame.size(0)
        num_antennas = self.num_antennas
        num_subcarriers = self.num_subcarriers
        num_time_slices = self.num_time_slices
        num_features = num_antennas * num_subcarriers

        #Split magnitudes and phases
        #Amplitudes: [:,:,:90], phases: [:,:,90:]
        amplitudes = wifi_csi_frame[:,:,:num_antennas*num_subcarriers]
        phases = wifi_csi_frame[:,:,num_antennas*num_subcarriers:]

        # Stack amplitudes and phases into a new dimension
        # Resulting shape: [batch_size, time_slices, num_antennas * num_subcarriers, 2]
        csi_data = torch.stack((amplitudes,phases), dim=-1)
        
        #Permute to get shape [batch_size, num_antennas, num_subcarriers, 2, num_time_slices]
        csi_data = csi_data.permute(0,2,3,1)

        # Combine batch_size, num_antennas, and num_subcarriers into a single dimension
        # Shape becomes [batch_size*num_antennas*num_subcarriers, 2, 10]
        csi_data = csi_data.reshape(-1, 2, 10)

        temporal_features = self.temporal_encoder(csi_data)
        
        # Reshape back to [batch_size, num_features, embedding_dim]
        embeddings = temporal_features.view(batch_size, num_features, self.embedding_dim)  # [batch_size, 342, embedding_dim]

        # Generate Positional Encodings
        device = wifi_csi_frame.device

        # Antenna and subcarrier positional encodings
        antenna_indices = torch.arange(num_antennas, device=device).unsqueeze(1).expand(-1, num_subcarriers).reshape(-1)
        subcarrier_indices = torch.arange(num_subcarriers, device=device).unsqueeze(0).expand(num_antennas, -1).reshape(-1)

        antenna_encodings = self.antenna_embeddings(antenna_indices)  # Shape: [342, embedding_dim]
        subcarrier_encodings = self.subcarrier_embeddings(subcarrier_indices)  # Shape: [342, embedding_dim]

        # Sum antenna and subcarrier encodings
        positional_encodings = antenna_encodings + subcarrier_encodings  # Shape: [342, embedding_dim]

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, 342, embedding_dim]

        # Add positional encodings to embeddings
        embeddings = embeddings + positional_encodings  # Shape: [batch_size, 342, embedding_dim]

        # Transformer Encoder
        encoder_output = self.encoder(embeddings)  # Shape: [batch_size, 342, embedding_dim]

        # Classification head
        # Permute to [batch_size, embedding_dim, num_features] for pooling
        encoder_output = encoder_output.permute(0, 2, 1)
        class_logits = self.classification_head(encoder_output)  # Shape: [batch_size, num_classes]

        return class_logits

        