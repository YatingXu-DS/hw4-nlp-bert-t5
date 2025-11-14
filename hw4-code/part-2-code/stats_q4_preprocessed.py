import os
import numpy as np
import torch
from transformers import T5TokenizerFast
from load_data import T5Dataset, _clean_sql

DATA_DIR = "./data"
MODEL_NAME = "google-t5/t5-small"

def compute_lengths_and_vocab(tensor_list):
    lengths = [len(x) for x in tensor_list]
    vocab = set()
    for x in tensor_list:
        vocab.update(x.tolist())
    return float(np.mean(lengths)), len(vocab)

def extract_preprocessed_lists(split):

    dataset = T5Dataset(DATA_DIR, split)
    enc_inputs = []
    sql_inputs = []

    for ex in dataset.examples:

        enc_inputs.append(ex["encoder_ids"])

        if ex["decoder_targets"] is not None:
            sql_inputs.append(ex["decoder_targets"])

    return enc_inputs, sql_inputs


def main():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

    # --------------------
    # 🔥 Train
    # --------------------
    train_enc, train_sql = extract_preprocessed_lists("train")

    train_mean_nl, train_vocab_nl = compute_lengths_and_vocab(train_enc)
    train_mean_sql, train_vocab_sql = compute_lengths_and_vocab(train_sql)

    print("========== Table 2: AFTER preprocessing ==========")
    print("\n--- Train ---")
    print("mean_sentence_length:", train_mean_nl)
    print("mean_sql_length:", train_mean_sql)
    print("vocab_size_nl:", train_vocab_nl)
    print("vocab_size_sql:", train_vocab_sql)

    # --------------------
    # 🔥 Dev
    # --------------------
    dev_enc, dev_sql = extract_preprocessed_lists("dev")

    dev_mean_nl, dev_vocab_nl = compute_lengths_and_vocab(dev_enc)
    dev_mean_sql, dev_vocab_sql = compute_lengths_and_vocab(dev_sql)

    print("\n--- Dev ---")
    print("mean_sentence_length:", dev_mean_nl)
    print("mean_sql_length:", dev_mean_sql)
    print("vocab_size_nl:", dev_vocab_nl)
    print("vocab_size_sql:", dev_vocab_sql)


if __name__ == "__main__":
    main()
