"""
File: attention_classifier.py

Description:
    This file implements a text classification model based on the Transformer architecture. 
    The model combines an Encoder module with a classification head to process tokenized 
    text sequences and output class probabilities. It leverages attention mechanisms 
    to capture contextual relationships in the input data.

Structure:
    - TransformerForClassification: A PyTorch module that integrates:
        1. An Encoder (from the `attention_transformer` module) for sequence representation.
        2. A fully connected classification head for generating class predictions.
    - Methods:
        1. make_src_mask: Creates a mask to ignore padding tokens in the source input.
        2. forward: Defines the forward pass through the Transformer model.

Usage:
    - Initialize the `TransformerForClassification` class with desired parameters, 
      such as vocabulary size, embedding size, number of layers, etc.
    - Use the `forward` method to process input tensors and generate output logits.

Attributes:
    - Encoder: Handles the attention-based encoding of input sequences.
    - Fully Connected Layer: Maps the encoded sequence representation to class probabilities.

Requirements:
    - PyTorch (torch, torch.nn)
    - Custom `Encoder` module from `attention_transformer`.

Key Features:
    - Attention-based sequence encoding with a Transformer architecture.
    - Global average pooling for sequence aggregation.
    - Flexible configuration options, including embedding size, attention heads, and dropout.

How to Use:
    1. Import the `TransformerForClassification` class.
    2. Initialize the model with the required parameters.
    3. Pass tokenized input sequences through the `forward` method to obtain predictions.

Example:
    from transformer_classification import TransformerForClassification

    model = TransformerForClassification(
        src_vocab_size=10000,
        src_pad_idx=0,
        num_classes=5,
        device="cuda"
    )
    src = torch.randint(0, 10000, (32, 100))  # Example input tensor
    logits = model(src)  # Output logits for classification
"""

import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from attention_transformer import Encoder  # Ensure this is your correct import
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerForClassification(nn.Module):
    """
    Transformer model for text classification, combining an Encoder and a classification head.

    Attributes:
        encoder (Encoder): Encoder module to process input sequences.
        fc_out (nn.Sequential): Fully connected layer for classification output.
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
        max_length=1200,  # UPDATED to match data preprocessing
        num_classes=5,
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

class EncodedTextDataset(Dataset):
    """
    Dataset for encoded text data and corresponding labels.
    """

    def __init__(self, encoded_texts, labels):
        self.encoded_texts = encoded_texts
        self.labels = labels

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        encoded_text = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded_text, label


def loader(load_path: str, sample_size: int, max_length: int, batch_size: int, src_pad_idx: int):
    logger.info(f"Loading data from {load_path}")
    spark = SparkSession.builder.getOrCreate()
    loaded_data = spark.read.parquet(load_path).dropna(how="any")
    loaded_data = loaded_data.select(
        col("tokenized_text").alias("encoded_text"), 
        col("stars").alias("label")
    )

    # Sample data for each star rating
    sampled_dataframe = []
    for star in range(1, 6):
        filtered_data = loaded_data.filter(col("label") == star)
        sampled_data = filtered_data.limit(sample_size).toPandas()
        sampled_dataframe.append(sampled_data)

    pandas_data = pd.concat(sampled_dataframe, ignore_index=True)

    # Convert label column to integer
    pandas_data["label"] = (
        pd.to_numeric(pandas_data["label"], errors="coerce")  
        .astype(int)                                          
    )

    pandas_data["label"] = pandas_data["label"] - 1
    
    # Split data
    train, test = train_test_split(pandas_data, test_size=0.2, random_state=42)

    # Extract encoded texts and labels
    train_texts = train["encoded_text"].tolist()
    train_labels = train["label"].tolist()
    test_texts = test["encoded_text"].tolist()
    test_labels = test["label"].tolist()

    # Pad/truncate
    train_texts = [
        seq[:max_length] + [src_pad_idx] * (max_length - len(seq))
        if len(seq) < max_length else seq[:max_length]
        for seq in train_texts
    ]
    test_texts = [
        seq[:max_length] + [src_pad_idx] * (max_length - len(seq))
        if len(seq) < max_length else seq[:max_length]
        for seq in test_texts
    ]

    # Create datasets
    train_dataset = EncodedTextDataset(train_texts, train_labels)
    test_dataset = EncodedTextDataset(test_texts, test_labels)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    spark.stop()
    return train_loader, test_loader


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=10,
    lr=1e-4,
    src_pad_idx=None,
):
    """
    Training loop for the TransformerForClassification model.

    Args:
        model (nn.Module): The Transformer model to train.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to use ("cuda" or "cpu").
        epochs (int): Number of epochs for training.
        lr (float): Learning rate.
        src_pad_idx (int, optional): Padding index for masking.

    Returns:
        None
    """
    # Move model to the specified device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Update progress bar
            loop.set_postfix(loss=loss.item(), 
                             acc=100.0 * train_correct / train_total)

        # Evaluation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Compute accuracy
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        # Log metrics at the end of each epoch
        train_accuracy = 100.0 * train_correct / train_total
        test_accuracy = 100.0 * test_correct / test_total
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss/len(test_loader):.4f} | "
            f"Test Acc: {test_accuracy:.2f}%"
        )


def main():

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    src_pad_idx = 0 

    # Hyperparameters
    data_path = "../data/encoded_format/encoded_data.parquet"
    sample_size = 20000
    max_length = 1200
    batch_size = 32
    epochs = 5
    lr = 1e-4

    train_loader, test_loader = loader(
        load_path=data_path,
        sample_size=sample_size,
        max_length=max_length,
        batch_size=batch_size,
        src_pad_idx=src_pad_idx,
    )

    # Because the loader returns DataLoaders, we can access the underlying Dataset.
    train_dataset = train_loader.dataset 
    test_dataset = test_loader.dataset

    # Find the maximum token ID in both training and testing sets
    max_token_id_train = max(max(seq) for seq in train_dataset.encoded_texts)
    max_token_id_test = max(max(seq) for seq in test_dataset.encoded_texts)
    overall_max_id = max(max_token_id_train, max_token_id_test)

    print(f"Maximum token ID in dataset: {overall_max_id}")
    dynamic_vocab_size = overall_max_id + 1

    model = TransformerForClassification(
        src_vocab_size=dynamic_vocab_size, 
        src_pad_idx=src_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.2,
        max_length=max_length,  
        num_classes=5,        
        device=device,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        src_pad_idx=src_pad_idx,
    )
    
    torch.save(model.state_dict(), "./attention_classifier")

if __name__ == "__main__":
    main()
