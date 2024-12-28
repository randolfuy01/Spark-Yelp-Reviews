import torch.nn as nn
from transformers import BertTokenizer, BertModel


class ReviewTransformerClassifier(nn.Module):
    """
    A BERT-based classifier for star ratings.
        - pretrained_model_name: The name of the pretrained BERT model to use
        - num_classes: The number of classes to classify
    """

    def __init__(self, pretrained_model_name="bert-base-uncased", num_classes=5):
        super(ReviewTransformerClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(
            self.bert.config.hidden_size, num_classes
        )

    def forward(self, input_texts):
        """
        Forward pass of the model
        - input_texts: The input text to classify
        - Returns: The logits of the model
        """
        tokens = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        outputs = self.bert(**tokens)
        cls_output = outputs.pooler_output
        logits = self.fc(self.dropout(cls_output))

        return logits
