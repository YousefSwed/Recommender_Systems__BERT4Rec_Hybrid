import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from config import Config

def load_data():
    # Load training data from CSV (sorted by timestamp)
    df = pd.read_csv(Config.DATA_DIR)
    df = df.sort_values(by=["user_id", "timestamp"])
    return df

def build_user_sequences(df):
    # Create user -> list of item_ids (in order)
    user_seqs = defaultdict(list)
    for _, row in df.iterrows():
        user_seqs[row['user_id']].append(row['item_id'])
    return user_seqs

def pad_sequence(seq, max_len):
    # Pad or truncate to max_len
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

def split_sequences(user_seqs):
    train_data, val_data = [], []
    for user, seq in user_seqs.items():
        if len(seq) < 3:
            continue

        train_seq = seq[:-1]  # All but last
        val_seq = seq         # Full sequence (target is last item)

        train_data.append(pad_sequence(train_seq, Config.SEQ_LEN))
        val_data.append(pad_sequence(val_seq, Config.SEQ_LEN))

    return train_data, val_data

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def process_and_save():
    process_test_data()
    preprocess_item_metadata()
    df = load_data()
    user_seqs = build_user_sequences(df)
    train_data, val_data = split_sequences(user_seqs)

    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    save_pickle(train_data, os.path.join(Config.PROCESSED_DIR, 'train_seqs.pkl'))
    save_pickle(val_data, os.path.join(Config.PROCESSED_DIR, 'val_seqs.pkl'))
    print("Saved preprocessed train and val sequences.")

def process_test_data():
    test_path = Config.TEST_FILE
    if not os.path.exists(test_path):
        print("test.csv not found. Skipping test preprocessing.")
        return []

    df = pd.read_csv(test_path)
    df = df.sort_values(by=["user_id", "timestamp"])
    user_seqs = defaultdict(list)
    for _, row in df.iterrows():
        user_seqs[row["user_id"]].append(row["item_id"])

    test_seqs = [pad_sequence(seq, Config.SEQ_LEN) for seq in user_seqs.values()]
    save_pickle(test_seqs, os.path.join(Config.PROCESSED_DIR, "test_seqs.pkl"))
    print("Saved preprocessed test sequences.")


def preprocess_item_metadata(meta_file=Config.META_FILE, save_path="preprocessed_data/item_meta.pkl"):
    df = pd.read_csv(meta_file)
    df = df.dropna(subset=["item_id", "main_category"])
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df["category_idx"] = le.fit_transform(df["main_category"])
    item_to_category = dict(zip(df["item_id"], df["category_idx"]))

    with open(save_path, "wb") as f:
        pickle.dump({
            "item_to_category": item_to_category,
            "num_categories": len(le.classes_)
        }, f)
    print("Saved item_meta.pkl")


if __name__ == "__main__":
    process_and_save()
