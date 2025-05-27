import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from config import Config


def generate_title_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv(Config.META_FILE)

    titles = df["title"].fillna("unknown").astype(str).tolist()
    item_ids = df["item_id"].tolist()

    print("Encoding titles with BERT...")
    title_vectors = model.encode(titles, show_progress_bar=True)

    title_dict = {
        int(item_id): vector.tolist()
        for item_id, vector in zip(item_ids, title_vectors)
    }

    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(Config.PROCESSED_DIR, "title_embeddings.pkl"), "wb") as f:
        pickle.dump(title_dict, f)

    print("Saved title_embeddings.pkl")


if __name__ == "__main__":
    generate_title_embeddings()