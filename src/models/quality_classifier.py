import torch
import torch.nn as nn
from transformers import AutoModel

class CodeQualityClassifier(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", dropout_rate=0.1):
        super(CodeQualityClassifier, self).__init__()
        
        # Load the pre-trained model
        self.codebert = AutoModel.from_pretrained(model_name)
        
        # Add classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.codebert.config.hidden_size, 2)  # Binary classification
        
    def forward(self, input_ids, attention_mask):
        # Get CodeBERT outputs
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits