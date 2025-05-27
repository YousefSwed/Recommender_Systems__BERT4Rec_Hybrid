import torch
import torch.nn as nn
from config import Config

class BERT4RecHybridTitle(nn.Module):
    def __init__(self, num_items, num_categories, item_to_category, title_embedding_dict):
        super(BERT4RecHybridTitle, self).__init__()

        self.item_embedding = nn.Embedding(num_items + 2, Config.EMBED_DIM, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, Config.EMBED_DIM, padding_idx=0)
        self.pos_embedding = nn.Embedding(Config.SEQ_LEN, Config.EMBED_DIM)

        self.title_proj = nn.Linear(384, Config.EMBED_DIM)  # assuming MiniLM 384-dim
        self.title_embedding_dict = title_embedding_dict
        self.item_to_category = item_to_category

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)

        self.norm = nn.LayerNorm(Config.EMBED_DIM)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.output = nn.Linear(Config.EMBED_DIM, num_items + 2)

    def forward(self, x, mask):
        device = x.device
        batch_size, seq_len = x.size()
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # item_id and category embeddings
        item_emb = self.item_embedding(x)
        category_ids = [[self.item_to_category.get(item.item(), 0) for item in seq] for seq in x]
        category_tensor = torch.tensor(category_ids, dtype=torch.long).to(device)
        cat_emb = self.category_embedding(category_tensor)

        # title embeddings (precomputed and projected)
        title_vectors = [[self.title_embedding_dict.get(item.item(), torch.zeros(384).tolist()) for item in seq] for seq in x]
        title_tensor = torch.tensor(title_vectors, dtype=torch.float32).to(device)
        title_emb = self.title_proj(title_tensor)

        # position embeddings
        pos_emb = self.pos_embedding(pos_ids)

        # combine all
        x = item_emb + cat_emb + pos_emb + title_emb
        x = self.norm(self.dropout(x))
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.output(x)
