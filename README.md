# Hybrid BERT4Rec Recommender System

This project implements a hybrid version of BERT4Rec for sequential recommendation, integrating both categorical metadata and BERT-based title embeddings to improve item representations.

---

## 📁 Project Structure

```plaintext
Final_project/
├── BERT4Rec_hybrid_title.py       # Full hybrid model with title embeddings
├── config.py                      # Global configuration settings
├── data_preprocessing.py          # Data loader and train/val splitter
├── embed_titles.py                # Precompute BERT title vectors
├── evaluate.py                    # Evaluation (Recall@k, NDCG@k)
├── generate_submission.py         # Generate submission.csv for Kaggle
├── main.py                        # Training/evaluation script
├── plot.py                        # Training curve and metric visualizations
├── train.py                       # Model training loop
├── tune_hyperparameters.py        # Tuning hyperparameters
├── data_set/                      # Input data (train.csv, test.csv, item_meta.csv)
├── preprocessed_data/             # Saved sequences and embeddings
└── results/                       # Model outputs and submissions
```

---

## 🧠 Models

### Supported Architectures

- `category`: BERT4Rec + item ID + category embedding
- `hybrid_title`: BERT4Rec + item ID + category + BERT title embedding

Select model using:

```bash
python main.py --model category
```

or

```bash
python main.py --model hybrid_title
```

> For `hybrid_title`, the title embeddings will be automatically generated.

All results are saved to subfolders in `results/{model_type}/`.

---

## 📋 Requirements

- Python 3.6+
- PyTorch (>= 1.6)
- pandas
- numpy
- matplotlib
- scikit-learn
- sentence_transformers
- tqdm
- optuna

---

## 🛠️ Setup

```bash
pip install torch pandas numpy matplotlib scikit-learn sentence_transformers tqdm optuna
```

### Run on GPU (Optional)

To train this module on a GPU, ensure that `CUDA` and `cuDNN` are installed on your device. Once installed, use the following command to install the appropriate version of PyTorch:

```bash
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu{version_number}
```

Replace `{version_number}` with the version of CUDA installed on your system.

For example, with `CUDA` version `12.8.1` and `cuDNN` version `9.8.0`, the command would be:

```bash
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note:** All training for this project was conducted on a GPU.

---

## 🚀 Usage

### 1. Preprocess Data & Train Model

```bash
python main.py --model hybrid_title
```

### 2. Generate Submission

```bash
python generate_submission.py --model hybrid_title
```

---

## 🧪 Hyperparameter Tuning

`tune_hyperparameters.py` uses [Optuna](https://optuna.org/) to automatically search for the best model hyperparameters (embedding size, number of layers, learning rate, etc.) based on validation NDCG@10.  
Run:

```bash
python tune_hyperparameters.py
```

Best hyperparameters are saved to `results/best_hyperparams.json`.

---

## 📊 Output

Results and logs are saved to `results/hybrid_title/`:

- `submission.csv` — predictions for top-10 items per user
- `model_metrics.json` — final Recall/NDCG@k values
- `model_performance.json` — epoch-wise training/val loss & metrics
- `metrics_at_k.png` — evaluation metrics plot

---

## 🔬 Evaluation Metrics

- **Recall@K**: Was the correct item among the top K?
- **NDCG@K**: Rank-sensitive score for recommended list

---

## 📌 Notes

- Values in `config.py` is the result of running `tune_hyperparameters.py`
- Supports metadata from `item_meta.csv` (main_category, title, description, price, average_rating)
- Trained using masked item prediction (MLM-style BERT loss)

---
