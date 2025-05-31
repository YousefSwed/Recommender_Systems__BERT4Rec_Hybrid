import json
import matplotlib.pyplot as plt
import os
from config import Config

def plot_metrics(result_dir):
    metrics_file = os.path.join(result_dir, "model_metrics.json")
    if not os.path.exists(metrics_file):
        print("Metrics file not found.")
        return

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    recall = metrics["recall"]
    ndcg = metrics["ndcg"]

    ks = sorted([int(k) for k in recall.keys()])
    recall_values = [recall[str(k)] for k in ks]
    ndcg_values = [ndcg[str(k)] for k in ks]

    plt.figure()
    plt.plot(ks, recall_values, marker='o', label='Recall@k')
    plt.plot(ks, ndcg_values, marker='x', label='NDCG@k')
    plt.xlabel("k")
    plt.ylabel("Metric")
    plt.title("Model Performance at Different k")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "metrics_at_k.png"))
    print("Saved metrics_at_k.png")

def plot_learning_curves(result_dir):
    performance_file = os.path.join(result_dir, "model_performance.json")
    if not os.path.exists(performance_file):
        print("Performance file not found.")
        return

    with open(performance_file, "r") as f:
        history = json.load(f)

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "learning_curves.png"))
    print("Saved learning_curves.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hybrid_title"], default="hybrid_title")
    args = parser.parse_args()
    result_dir = os.path.join(Config.RESULTS_DIR, args.model)
    plot_learning_curves(result_dir)
    plot_metrics(result_dir)
