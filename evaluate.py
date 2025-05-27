import torch
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

def evaluate_model(model, test_data, num_items, device, k_values=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
    model.eval()
    # Initialize dictionaries to store recall@k and ndcg@k values
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}

    batch_size = 512  # Use a large batch size to leverage GPU

    with torch.no_grad():
        # Iterate through the test data in batches
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch = test_data[i:i + batch_size]
            
            # Convert batch to tensor and move to device
            input_seqs = torch.LongTensor(batch).to(device)
            
            # Create mask for padding tokens
            mask = (input_seqs == 0)
            
            # Get model outputs
            outputs = model(input_seqs, mask)  
            
            # Get the last item's output for each sequence
            last_outputs = outputs[:, -1, :]   

            # Get the top-k item indices
            topk = torch.topk(last_outputs, k=max(k_values), dim=-1).indices 

            # Iterate through each sequence in the batch
            for idx, seq in enumerate(batch):
                target_item = seq[-1]  # The last item in the sequence is the target
                topk_items = topk[idx].tolist()  # Convert the top-k indices to a list

                # Calculate recall and ndcg for each k value
                for k in k_values:
                    top_k = topk_items[:k]  # Get the top-k items
                    hit = int(target_item in top_k)  # Check if the target item is in top-k
                    recalls[k].append(hit)  # Append hit to recalls list

                    # Create relevance list for ndcg calculation
                    relevance = [1 if item == target_item else 0 for item in top_k]
                    # Calculate ndcg score
                    ndcg = ndcg_score([relevance], [[1] * len(relevance)])
                    ndcgs[k].append(ndcg)  # Append ndcg to ndcgs list

    # Calculate the mean recall@k and ndcg@k values
    metrics = {
        "recall": {k: np.mean(recalls[k]) for k in k_values},
        "ndcg": {k: np.mean(ndcgs[k]) for k in k_values},
    }

    return metrics
