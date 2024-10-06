import torch
import torch.nn as nn
import torch.nn.functional as F

# Define CNN + MLP + Transformer Model
class CipherTransformer(nn.Module):
    def __init__(self, num_ciphers=11, d_model=256, num_heads=8, num_layers=8):
        super(CipherTransformer, self).__init__()

        # CNN to process 1x1000 matrix (sequence treated as 1D image)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),  # (1, 64, 1000)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (1, 64, 500)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),  # (1, 128, 500)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # (1, 128, 250)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),  # (1, 256, 250)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # (1, 256, 125)
        )

        # MLP to flatten CNN output and prepare for transformer input
        self.mlp = nn.Sequential(
            nn.Linear(256 * 125, d_model),  # d_model is the expected input size for transformer
            nn.ReLU(),
            nn.Linear(d_model, d_model),  # Additional hidden layer in MLP
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Positional encoding for Transformer
        self.positional_encoding = nn.Parameter(torch.zeros(1, 125, d_model))

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final MLP for output
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_ciphers),  # num_ciphers = 10
        )

    def forward(self, x):
        # Input x: (1, 1000)
        x = x.unsqueeze(0)  # Add a channel dimension to make it (1, 1, 1000)

        # Pass through CNN
        cnn_out = self.cnn(x)  # Output shape: (1, 256, 125)

        # Flatten CNN output and pass through MLP
        cnn_out_flattened = cnn_out.view(-1)  # Flatten to (256*125)
        mlp_out = self.mlp(cnn_out_flattened)  # Output shape: (d_model)

        # Add positional encoding: (1, 125, d_model)
        mlp_out = mlp_out.unsqueeze(0) + self.positional_encoding

        # Transformer expects (sequence_length, 1, d_model), so we need to transpose
        transformer_input = mlp_out.transpose(0, 1)  # Shape: (125, 1, d_model)

        # Pass through Transformer
        transformer_out = self.transformer(transformer_input)  # Shape: (125, 1, d_model)

        # Take the output of the last token for classification
        final_out = transformer_out[-1]  # (1, d_model)

        # Pass through final MLP to get probabilities for cipher types
        output = self.fc_out(final_out)  # (1, num_ciphers)

        return output  # Categorical output with probabilities