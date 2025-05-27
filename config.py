# File: Final_project/config.py

class Config:
    # Model hyperparameters
    SEQ_LEN = 20
    MASK_PROB = 0.15
    EMBED_DIM = 512
    NUM_HEADS = 4
    NUM_LAYERS = 4
    DROPOUT = 0.2

    # Training hyperparameters
    BATCH_SIZE = 128
    LR = 1e-4
    EPOCHS = 100
    PATIENCE = 5

    # Paths
    MODEL_SAVE_PATH = "results/best_model.pt"
    DATA_DIR = "data_set/train.csv"
    META_FILE = "data_set/item_meta.csv"
    TEST_FILE = "data_set/test.csv"
    PROCESSED_DIR = "preprocessed_data/"
    RESULTS_DIR = "results/"
