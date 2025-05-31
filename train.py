import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from config import Config
from evaluate import evaluate_model
import json
import os
import pickle

random.seed(42)

# Apply random masking to training items
def mask_items(seqs, num_items, mask_prob):
    MASK_ID = num_items + 1
    masked_seqs, labels = [], []

    for seq in seqs:
        masked, label = [], []
        for item in seq:
            if item != 0 and random.random() < mask_prob:
                masked.append(MASK_ID)
                label.append(item)
            else:
                masked.append(item)
                label.append(0)
        masked_seqs.append(masked)
        labels.append(label)

    return torch.LongTensor(masked_seqs), torch.LongTensor(labels)

# Training function
def train_model(model, train_data, val_data, num_items, device, model_path, result_dir):
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)

    history = []
    best_ndcg = 0.0
    best_recall = 0.0
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss = 0
        random.shuffle(train_data)

        loop = tqdm(range(0, len(train_data), Config.BATCH_SIZE), desc=f"Epoch {epoch}")
        for i in loop:
            batch = train_data[i:i + Config.BATCH_SIZE]
            masked_inputs, labels = mask_items(batch, num_items, Config.MASK_PROB)
            masked_inputs, labels = masked_inputs.to(device), labels.to(device)
            mask = (masked_inputs == 0)

            logits = model(masked_inputs, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            train_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.detach().item())

        val_loss = evaluate_val_loss(model, val_data, criterion, num_items, device)
        val_metrics = evaluate_model(model, val_data, num_items, device, k_values=[10])
        val_ndcg = val_metrics['ndcg'][10]
        val_recall = val_metrics['recall'][10]

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_data),
            "val_loss": val_loss,
            "val_ndcg": val_ndcg,
            "val_recall": val_recall
        }
        history.append(log_entry)

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("\nEarly stopping triggered.\n")
                break
        
        print(f"Epoch {epoch} | Train Loss: {log_entry['train_loss']:.4f} | Val Loss: {val_loss:.4f} | Recall@10: {val_recall:.4f} | NDCG@10: {val_ndcg:.4f} | Best NDCG@10: {best_ndcg:.4f} | Patience counter: {patience_counter}\n")
        
        scheduler.step(val_ndcg)

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'model_performance.json'), 'w') as f:
        json.dump(history, f, indent=2)

# Validation loss calculator
def evaluate_val_loss(model, val_data, criterion, num_items, device, mask_prob=Config.MASK_PROB):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        loop = tqdm(range(0, len(val_data), Config.BATCH_SIZE), desc="Validating", leave=False)
        for i in loop:
            batch = val_data[i:i + Config.BATCH_SIZE]
            masked_inputs, labels = mask_items(batch, num_items, mask_prob)
            masked_inputs, labels = masked_inputs.to(device), labels.to(device)
            mask = (masked_inputs == 0)
            logits = model(masked_inputs, mask)

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss / len(val_data)
