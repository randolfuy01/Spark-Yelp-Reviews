"""
-------------------------------------------------------------------------
Classification Model based on the attention transformer
-------------------------------------------------------------------------
"""
import torch.nn as nn
import torch
from attention_transformer import SelfAttention, Transformer, Encoder

class TransformerForClassification(nn.Module):
    """
    Transformer model for text classification, combining an Encoder and a classification head.

    Attributes:
        encoder (Encoder): Encoder module to process input sequences.
        fc_out (nn.Linear): Fully connected layer for classification output.
        src_pad_idx (int): Padding index for source tokens.
        device (str): Device on which the model runs.
    """

    def __init__(
        self,
        src_vocab_size,
        src_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        max_length=100,
        num_classes=5,  # Number of output classes
        device="cuda",
    ):
        super(TransformerForClassification, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, num_classes),
        )
        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Creates a mask for the source sequence to ignore padding tokens.

        Args:
            src (torch.Tensor): Source input tensor of shape (N, src_seq_length).

        Returns:
            torch.Tensor: Source mask of shape (N, 1, 1, src_seq_length).
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def forward(self, src):
        """
        Forward pass for the Transformer model for classification.

        Args:
            src (torch.Tensor): Input tensor of shape (N, src_seq_length).

        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        # Global average pooling over sequence length
        enc_src = enc_src.mean(dim=1)
        out = self.fc_out(enc_src)
        return out
