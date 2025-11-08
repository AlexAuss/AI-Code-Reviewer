import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path

from data.diff_processor import load_diff_data
from models.quality_classifier import CodeQualityClassifier

def train_model(
    train_file,
    val_file,
    model_save_path,
    batch_size=16,
    epochs=3,
    learning_rate=2e-5,
    max_length=512,
    model_name="microsoft/codebert-base"
):
    # Create save directory if it doesn't exist
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = load_diff_data(train_file, model_name, max_length)
    val_dataset = load_diff_data(val_file, model_name, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeQualityClassifier(model_name).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_accuracy = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                
                accuracy = (predictions == labels).float().mean()
                val_accuracy += accuracy.item()
                val_steps += 1
        
        avg_val_accuracy = val_accuracy / val_steps
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}")
        
        # Save best model
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with validation accuracy: {avg_val_accuracy:.4f}")

if __name__ == "__main__":
    # Set paths
    base_path = Path(__file__).parent.parent
    train_file = str(base_path / "Datasets/Diff_Quality_Estimation/cls-train-chunk-0.jsonl")
    val_file = str(base_path / "Datasets/Diff_Quality_Estimation/cls-valid.jsonl")
    model_save_path = str(base_path / "models/saved/quality_classifier.pth")
    
    # Train the model
    train_model(train_file, val_file, model_save_path)