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

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, num_points):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Initialize learnable query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(num_points, embedding_dim))

    def forward(self, memory):
        # memory: Encoder outputs, shape [batch_size, sequence_length, embedding_dim]
        batch_size = memory.size(0)
        # Expand query embeddings to match batch size
        tgt = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, num_points, embedding_dim]
        output = self.transformer_decoder(tgt, memory)  # Shape: [batch_size, num_points, embedding_dim]
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