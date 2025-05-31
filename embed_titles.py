import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from config import Config


def generate_title_and_desc_embeddings():
    # Check if the '.pkl' files exist
    if os.path.exists(os.path.join(Config.PROCESSED_DIR, 'title_embeddings.pkl')) and \
       os.path.exists(os.path.join(Config.PROCESSED_DIR, 'desc_embeddings.pkl')):
        print("Title and description embeddings already exist. Skipping generation.")
        return

    df = pd.read_csv(Config.META_FILE)

    # Load transformer model once
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Title embeddings
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

    # Description embeddings
    print("Encoding item descriptions...")
    desc_embeddings = model.encode(df['description'].astype(str).tolist(), show_progress_bar=True)
    desc_emb_dict = {item_id: emb for item_id, emb in zip(df['item_id'], desc_embeddings)}
    with open(os.path.join(Config.PROCESSED_DIR, "desc_embeddings.pkl"), "wb") as f:
        pickle.dump(desc_emb_dict, f)
    print("Saved desc_embeddings.pkl")


if __name__ == "__main__":
    generate_title_and_desc_embeddings()