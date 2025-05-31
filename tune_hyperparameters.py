import optuna
import torch
import pickle
import os
from train import train_model
from evaluate import evaluate_model
from BERT4Rec_hybrid_title import BERT4RecHybridTitle
from data_preprocessing import process_and_save
from embed_titles import generate_title_and_desc_embeddings
from config import Config

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def objective(trial):
    opt_model_path='results/optuna_tmp/tmp.pt'
    opt_result_dir='results/optuna_tmp'
    
    os.makedirs(opt_result_dir, exist_ok=True)
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)


    print("Processing data...")
    process_and_save()

    print("Preparing title embeddings...")
    # subprocess.run(["./.venv/Scripts/python", "embed_titles.py"])
    generate_title_and_desc_embeddings()
    print("Title embeddings generated.\n")

    # Suggest hyperparameters
    embed_dim = trial.suggest_categorical('embed_dim', [256, 512, 768, 1024])
    num_layers = trial.suggest_int('num_layers', 2, 8)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    seq_len = trial.suggest_int('seq_len', 20, 30)

    # Load data
    train_data = load_pickle(os.path.join(Config.PROCESSED_DIR, 'train_seqs.pkl'))
    val_data = load_pickle(os.path.join(Config.PROCESSED_DIR, 'val_seqs.pkl'))
    test_data = load_pickle(os.path.join(Config.PROCESSED_DIR, 'test_seqs.pkl'))

    # Load metadata
    meta = load_pickle(os.path.join(Config.PROCESSED_DIR, "item_meta.pkl"))
    item_to_category = meta["item_to_category"]
    num_categories = meta["num_categories"]
    with open(os.path.join(Config.PROCESSED_DIR, "title_embeddings.pkl"), "rb") as f:
        title_embedding_dict = pickle.load(f)
    with open(os.path.join(Config.PROCESSED_DIR, "desc_embeddings.pkl"), "rb") as f:
        desc_embedding_dict = pickle.load(f)
    with open(os.path.join(Config.PROCESSED_DIR, "numeric_meta.pkl"), "rb") as f:
        numeric_meta = pickle.load(f)

    # Model params
    num_items = max(item_to_category.keys()) + 1  # or get from your config

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = BERT4RecHybridTitle(
        num_items=num_items,
        num_categories=num_categories,
        item_to_category=item_to_category,
        title_embedding_dict=title_embedding_dict,
        desc_embedding_dict=desc_embedding_dict,
        numeric_meta=numeric_meta,
        embed_dim=embed_dim,
    )

    # Use your training and evaluation functions
    # For speed, you may want to set EPOCHS to a lower number during tuning
    Config.BATCH_SIZE = batch_size
    Config.LR = lr
    Config.EMBED_DIM = embed_dim
    Config.NUM_HEADS = num_heads
    Config.NUM_LAYERS = num_layers
    Config.DROPOUT = dropout
    Config.SEQ_LEN = seq_len
    Config.EPOCHS = 15  # or a lower number for faster tuning

    print("\nTraining...\n")

    # Train the model (save to a temp path, e.g. 'results/tmp.pt')
    train_model(
        model,
        train_data,
        val_data,
        num_items,
        device,
        model_path=opt_model_path,
        result_dir=opt_result_dir,
    )

    # Evaluate
    val_metrics = evaluate_model(model, test_data, num_items, device, k_values=[10])
    ndcg10 = val_metrics['ndcg'][10]
    # For Optuna, we want to maximize ndcg
    return ndcg10

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save the best hyperparameters
    import json
    with open("results/best_hyperparams.json", "w") as f:
        json.dump(trial.params, f, indent=2)
