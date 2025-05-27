# Directory: Final_project/generate_submission.py

import torch
import pickle
import os
import pandas as pd
import json
from config import Config
from BERT4Rec_hybrid_title import BERT4RecHybridTitle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def generate_submission(model_type="category"):
    test_data = load_pickle(os.path.join(Config.PROCESSED_DIR, "test_seqs.pkl"))
    meta = load_pickle(os.path.join(Config.PROCESSED_DIR, "item_meta.pkl"))
    item_to_category = meta["item_to_category"]
    num_categories = meta["num_categories"]

    result_dir = os.path.join(Config.RESULTS_DIR, model_type)
    model_path = os.path.join(result_dir, "best_model.pt")
    os.makedirs(result_dir, exist_ok=True)

    # Load the exact num_items used during training
    with open(os.path.join(result_dir, "meta.json"), "r") as f:
        num_items = json.load(f)["num_items"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "hybrid_title":
        title_embedding_dict = load_pickle(os.path.join(Config.PROCESSED_DIR, "title_embeddings.pkl"))
        model = BERT4RecHybridTitle(num_items, num_categories, item_to_category, title_embedding_dict)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    submission = []
    for idx, seq in enumerate(test_data):
        user_id = idx + 1
        padded = [0] * (Config.SEQ_LEN - len(seq)) + seq[-Config.SEQ_LEN:]
        input_tensor = torch.LongTensor(padded).unsqueeze(0).to(device)
        mask = (input_tensor == 0)

        with torch.no_grad():
            output = model(input_tensor, mask)
            logits = output[:, -1, :]
            top_items = torch.topk(logits, k=10, dim=-1).indices.squeeze().tolist()

        top_items_str = ",".join(map(str, top_items))
        submission.append([idx, user_id, top_items_str])

    df = pd.DataFrame(submission, columns=["ID", "user_id", "item_id"])
    df.to_csv(os.path.join(result_dir, "submission.csv"), index=False)
    print(f"Submission file saved to {os.path.join(result_dir, 'submission.csv')}")
    print("=" *  30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hybrid_title"], default="hybrid_title")
    args = parser.parse_args()
    generate_submission(model_type=args.model)
