import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DiffQualityDataset(Dataset):
    def __init__(self, file_path, tokenizer_name="microsoft/codebert-base", max_length=512):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Load and process the data
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process the diff text
        diff_text = item.get('diff', '')  # Get the diff text
        label = 1 if item.get('label', '').lower() == 'good' else 0  # Convert label to binary
        
        # Tokenize the diff
        encoding = self.tokenizer(
            diff_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_diff_data(file_path, tokenizer_name="microsoft/codebert-base", max_length=512):
    """Helper function to load the dataset"""
    return DiffQualityDataset(file_path, tokenizer_name, max_length)