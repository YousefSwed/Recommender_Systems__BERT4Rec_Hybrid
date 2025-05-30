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

def generate_submission():
    # Load sample_submission.csv for correct user list and order
    sample_df = pd.read_csv('data_set/sample_submission.csv')
    target_user_ids = sample_df['user_id'].tolist()  # 892 user_ids

    # Build dict of {user_id: sequence} from test.csv
    test_df = pd.read_csv(Config.TEST_FILE)
    test_df = test_df.sort_values(["user_id", "timestamp"])
    user2seq = test_df.groupby('user_id')['item_id'].apply(list).to_dict()

    # --- Compute top-10 most popular items from train.csv ---
    train_df = pd.read_csv(Config.DATA_DIR)
    pop_items = train_df["item_id"].value_counts().sort_values(ascending=False).index.tolist()
    top10_popular = list(map(int, pop_items[:10]))  # Ensure ints

    # Load all model and metadata as before
    meta = load_pickle(os.path.join(Config.PROCESSED_DIR, "item_meta.pkl"))
    item_to_category = meta["item_to_category"]
    num_categories = meta["num_categories"]
    result_dir = Config.RESULTS_DIR
    model_path = os.path.join(result_dir, "best_model.pt")
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "meta.json"), "r") as f:
        num_items = json.load(f)["num_items"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    title_embedding_dict = load_pickle(os.path.join(Config.PROCESSED_DIR, "title_embeddings.pkl"))
    desc_embedding_dict = load_pickle(os.path.join(Config.PROCESSED_DIR, "desc_embeddings.pkl"))
    numeric_meta = load_pickle(os.path.join(Config.PROCESSED_DIR, "numeric_meta.pkl"))

    model = BERT4RecHybridTitle(num_items, num_categories, item_to_category, title_embedding_dict, desc_embedding_dict, numeric_meta)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Generate predictions for each user_id in sample_submission
    submission = []
    for user_id in target_user_ids:
        seq = user2seq.get(user_id, [])
        if len(seq) == 0:
            # Use most popular items for cold-start
            top_items = top10_popular
        else:
            padded = [0] * (Config.SEQ_LEN - len(seq)) + seq[-Config.SEQ_LEN:]
            input_tensor = torch.LongTensor(padded).unsqueeze(0).to(device)
            mask = (input_tensor == 0)
            with torch.no_grad():
                output = model(input_tensor, mask)
                logits = output[:, -1, :]
                top_items = torch.topk(logits, k=10, dim=-1).indices.squeeze().tolist()
                if isinstance(top_items, int):
                    top_items = [top_items]
        top_items_str = ",".join(map(str, top_items))
        # ID == user_id in this competition
        submission.append([user_id, user_id, top_items_str])

    # Save submission with required header
    df = pd.DataFrame(submission, columns=["ID", "user_id", "item_id"])
    df.to_csv(os.path.join(result_dir, "submission.csv"), index=False)
    print(f"Submission file saved to {os.path.join(result_dir, 'submission.csv')}")
    print("=" * 30)

if __name__ == "__main__":
    generate_submission()
