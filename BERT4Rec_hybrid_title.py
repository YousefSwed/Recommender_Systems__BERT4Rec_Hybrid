import torch
import torch.nn as nn
import numpy as np
from config import Config

class BERT4RecHybridTitle(nn.Module):
    def __init__(
        self,
        num_items,
        num_categories,
        item_to_category,
        title_embedding_dict,
        desc_embedding_dict,
        numeric_meta,
        embed_dim=Config.EMBED_DIM,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        self.item_to_category = item_to_category

        # Text embedding projections
        title_dim = len(next(iter(title_embedding_dict.values())))
        desc_dim = len(next(iter(desc_embedding_dict.values())))
        self.title_proj = nn.Linear(title_dim, embed_dim)
        self.desc_proj = nn.Linear(desc_dim, embed_dim)

        # Numeric embedding projection: 2 features (rating, price) -> embed_dim
        self.numeric_proj = nn.Linear(2, embed_dim)
        self.title_embedding_dict = title_embedding_dict
        self.desc_embedding_dict = desc_embedding_dict
        self.numeric_meta = numeric_meta

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=Config.NUM_HEADS, dropout=Config.DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.output = nn.Linear(embed_dim, num_items + 2)  # +2 for mask & padding

    def forward(self, input_ids, mask):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        input_ids_np = input_ids.cpu().numpy()

        # --- Item Embedding ---
        item_emb = self.item_embedding(input_ids)  # [B, S, D]

        # --- Category Embedding ---
        cat_indices = np.array([[self.item_to_category.get(i, 0) for i in seq] for seq in input_ids_np])
        cat_emb = self.category_embedding(torch.LongTensor(cat_indices).to(device))

        # --- Title Embedding ---
        title_vectors = [
            [self.title_embedding_dict.get(i, np.zeros(self.title_proj.in_features)) for i in seq]
            for seq in input_ids_np
        ]
        title_tensor = torch.tensor(np.array(title_vectors), dtype=torch.float32).to(device)
        title_emb = self.title_proj(title_tensor)

        # --- Description Embedding ---
        desc_vectors = [
            [self.desc_embedding_dict.get(i, np.zeros(self.desc_proj.in_features)) for i in seq]
            for seq in input_ids_np
        ]
        desc_tensor = torch.tensor(np.array(desc_vectors), dtype=torch.float32).to(device)
        desc_emb = self.desc_proj(desc_tensor)

        # --- Numeric Metadata ---
        num_vectors = [
            [
                *self.numeric_meta.get(i, {'average_rating_norm': 0.0, 'price_norm': 0.0}).values()
            ]
            for seq in input_ids_np for i in seq
        ]
        num_vectors = np.array(num_vectors).reshape(batch_size, seq_len, 2)
        num_tensor = torch.tensor(num_vectors, dtype=torch.float32).to(device)
        num_emb = self.numeric_proj(num_tensor)

        # combine all embeddings:
        x = item_emb + cat_emb + title_emb + desc_emb + num_emb  # [B, S, D]
        x = self.dropout(x)
        # batch_first=True: no need to permute
        x = self.transformer(x, src_key_padding_mask=mask)  # mask shape: [B, S]
        logits = self.output(x)
        return logits
