import os
import numpy as np
from transformers import T5Tokenizer

DATA_DIR = "./data"
MODEL_NAME = "google-t5/t5-small"

def load_nonempty_lines(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return [l for l in lines if l != ""]

def compute_stats(nl_list, sql_list, tokenizer):
    assert len(nl_list) == len(sql_list)

    # --- KEY: use tokenizer.tokenize (string tokens) ---
    nl_tokens = [tokenizer.tokenize(x) for x in nl_list]
    sql_tokens = [tokenizer.tokenize(x) for x in sql_list]

    nl_lengths = [len(t) for t in nl_tokens]
    sql_lengths = [len(t) for t in sql_tokens]

    nl_vocab = set(tok for seq in nl_tokens for tok in seq)
    sql_vocab = set(tok for seq in sql_tokens for tok in seq)

    return {
        "num_examples": len(nl_list),
        "mean_sentence_length": float(np.mean(nl_lengths)),
        "mean_sql_length": float(np.mean(sql_lengths)),
        "vocab_size_nl": len(nl_vocab),
        "vocab_size_sql": len(sql_vocab),
    }

def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    train_nl = load_nonempty_lines(os.path.join(DATA_DIR, "train.nl"))
    train_sql = load_nonempty_lines(os.path.join(DATA_DIR, "train.sql"))
    dev_nl   = load_nonempty_lines(os.path.join(DATA_DIR, "dev.nl"))
    dev_sql  = load_nonempty_lines(os.path.join(DATA_DIR, "dev.sql"))

    print("=== Train ===")
    print(compute_stats(train_nl, train_sql, tokenizer))

    print("\n=== Dev ===")
    print(compute_stats(dev_nl, dev_sql, tokenizer))

if __name__ == "__main__":
    main()
