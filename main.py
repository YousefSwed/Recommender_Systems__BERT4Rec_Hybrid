import pickle
import torch
import os
import json
from data_preprocessing import process_and_save
from BERT4Rec_hybrid_title import BERT4RecHybridTitle
from train import train_model
from evaluate import evaluate_model
from embed_titles import generate_title_and_desc_embeddings
from plot import plot_learning_curves, plot_metrics
from config import Config


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def main():
    # result_dir = os.path.join(Config.RESULTS_DIR, model_type)
    result_dir = Config.RESULTS_DIR
    model_path = os.path.join(result_dir, "best_model.pt")
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print("Processing data...")
    process_and_save()
    print("Data processing complete.\n")

    print("Preparing title embeddings...")
    # subprocess.run(["./.venv/Scripts/python", "embed_titles.py"])
    generate_title_and_desc_embeddings()
    print("Title embeddings generated.\n")

    print("Loading preprocessed data...")
    train_data = load_pickle(os.path.join(
        Config.PROCESSED_DIR, 'train_seqs.pkl'))
    val_data = load_pickle(os.path.join(Config.PROCESSED_DIR, 'val_seqs.pkl'))
    test_data = load_pickle(os.path.join(
        Config.PROCESSED_DIR, 'test_seqs.pkl'))

    meta = load_pickle(os.path.join(Config.PROCESSED_DIR, "item_meta.pkl"))
    item_to_category = meta["item_to_category"]
    num_categories = meta["num_categories"]
    num_items = max(max(seq) for seq in train_data + val_data + test_data)

    print(f"\nBuilding model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    title_embedding_dict = load_pickle(os.path.join(
        Config.PROCESSED_DIR, "title_embeddings.pkl"))
    desc_embedding_dict = load_pickle(os.path.join(
        Config.PROCESSED_DIR, "desc_embeddings.pkl"))
    numeric_meta = load_pickle(os.path.join(
        Config.PROCESSED_DIR, "numeric_meta.pkl"))

    model = BERT4RecHybridTitle(num_items, num_categories, item_to_category,
                                title_embedding_dict, desc_embedding_dict, numeric_meta)

    print("Training...")
    train_model(model, train_data, val_data, num_items,
                device, model_path, result_dir)

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    print("Evaluating...")
    metrics = evaluate_model(model, test_data, num_items, device)
    print("Evaluation Results:\n", metrics)

    with open(os.path.join(result_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save num_items for inference consistency
    with open(os.path.join(result_dir, "meta.json"), "w") as f:
        json.dump({"num_items": int(num_items)}, f)

    print("=" * 30)

    print("Generating plots...")
    plot_learning_curves(result_dir)
    plot_metrics(result_dir)

    print("=" * 30)


if __name__ == "__main__":
    main()
