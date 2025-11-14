'''
'''
import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch
try:
    import sqlparse
except Exception:
    sqlparse = None

PAD_IDX = 0


def _clean_sql(sql_raw: str) -> str:
    """Clean and normalize SQL with conservative edits.

    Steps:
    1) Try to format with sqlparse (keywords uppercased, comments stripped) if available.
    2) Remove tautologies like (1 = 1) and trailing AND/OR 1 = 1 patterns.
    3) Fix dangling conjunctions (WHERE AND/OR, AND/OR ) ).
    4) Normalize whitespace and comma spacing, ensure trailing semicolon.
    5) Validate by parsing with sqlparse; if parse fails or a crude heuristic detects missing comparison operators
       for some common fields, return the original raw SQL (rollback).
    """
    import re
    sql = sql_raw
    # 1) Basic formatting (do not change identifiers)
    if sqlparse is not None:
        try:
            sql = sqlparse.format(sql, keyword_case='upper', strip_comments=True)
        except Exception:
            sql = sql_raw

    # 2) Remove common tautologies like (1 = 1) or AND 1 = 1
    # remove parenthesized 1=1 first
    sql = re.sub(r"\(\s*1\s*=\s*1\s*\)", " ", sql, flags=re.IGNORECASE)
    # remove patterns like 'AND 1 = 1' or 'OR 1 = 1'
    sql = re.sub(r"\s+(AND|OR)\s+1\s*=\s*1\b", " ", sql, flags=re.IGNORECASE)
    # bare 1=1
    sql = re.sub(r"\b1\s*=\s*1\b", " ", sql, flags=re.IGNORECASE)

    # 3) Fix dangling conjunctions
    sql = re.sub(r"\bWHERE\s+(AND|OR)\b", "WHERE ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+(AND|OR)\s*\)", " )", sql, flags=re.IGNORECASE)

    # 4) Whitespace and punctuation normalization
    sql = re.sub(r"\s+", " ", sql).strip()
    sql = re.sub(r"\s*,\s*", ", ", sql)
    if not sql.endswith(";"):
        sql = sql + ";"

    # 5) Robustness checks: parse and basic operator sanity
    try:
        parsed = None
        if sqlparse is not None:
            parsed = sqlparse.parse(sql)
        if not parsed:
            # parse failed or empty
            return sql_raw
    except Exception:
        return sql_raw

    # crude check: if we see field followed by a number (no operator), consider it suspect
    suspect = re.search(r"\b(arrival_time|departure_time|day_number|month_number|year)\s+\d", sql, re.IGNORECASE)
    if suspect and not re.search(r"\b(arrival_time|departure_time|day_number|month_number|year)\s*(=|<|>|<=|>=|<>|BETWEEN)\b", sql, re.IGNORECASE):
        return sql_raw

    return sql


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
                    '''
                    Skeleton for the class for performing data processing for the T5 model.

                    Some tips for implementation:
                            * You should be using the 't5-small' tokenizer checkpoint to tokenize both
                                the encoder and decoder output. 
                            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
                                T5Tokenizer should serve that purpose.
                            * Class behavior should be different on the test set.
                    '''
                    assert split in {"train", "dev", "test"}
                    self.data_folder = data_folder
                    self.split = split

                    # tokenizer for both encoder and decoder
                    self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

                    # use an extra_id as the initial decoder token (T5 has <extra_id_0>...)
                    self.init_token = '<extra_id_0>'
                    # ensure tokenizer knows about the token (it exists in T5 vocab)
                    self.init_token_id = self.tokenizer.convert_tokens_to_ids(self.init_token)

                    # store processed entries
                    self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        sql_path = os.path.join(data_folder, f"{split}.sql")

        nl_lines = load_lines(nl_path) if os.path.exists(nl_path) else []
        sql_lines = load_lines(sql_path) if os.path.exists(sql_path) else []

        examples = []
        for i, nl in enumerate(nl_lines):
            # --- Natural Language preprocessing ---
            nl_norm = re.sub(r"\s+", " ", nl.strip()).lower()
            # strictly match description
            prompt = f"translate question to SQL: {nl_norm}"

            enc = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=512)
            enc_ids = torch.LongTensor(enc['input_ids'])
            enc_mask = torch.LongTensor(enc.get('attention_mask', [1]*len(enc_ids)))

            if split == 'test' or i >= len(sql_lines):
                dec_inputs = None
                dec_targets = None
                raw_sql = None
            else:
                raw_sql = sql_lines[i].strip()
                sql = _clean_sql(raw_sql)

                # tokenize SQL
                sql_enc = tokenizer(sql, add_special_tokens=True, truncation=True, max_length=512)
                sql_ids = sql_enc['input_ids']

                # --- decoder right-shift (teacher forcing) ---
                pad_id = tokenizer.pad_token_id
                dec_targets = torch.LongTensor(sql_ids)
                # shift right: prepend pad token, remove last element
                dec_inputs = torch.LongTensor([pad_id] + sql_ids[:-1])

            examples.append({
                'encoder_ids': enc_ids,
                'encoder_mask': enc_mask,
                'decoder_inputs': dec_inputs,
                'decoder_targets': dec_targets,
                'init_decoder_input': tokenizer.pad_token_id,
                'raw_nl': nl,
                'raw_sql': raw_sql,
            })
        return examples

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # batch: list of dicts
    encoder_ids = [item['encoder_ids'] for item in batch]
    encoder_masks = [item['encoder_mask'] for item in batch]
    decoder_inputs = [item['decoder_inputs'] for item in batch]
    decoder_targets = [item['decoder_targets'] for item in batch]
    init_decoder_inputs = [item['init_decoder_input'] for item in batch]

    # pad encoder ids
    enc_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    enc_mask_padded = pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    # decoder padding: some entries (e.g., test) may have None
    if any(x is None for x in decoder_inputs):
        # create empty tensors
        dec_in_padded = torch.zeros((len(batch), 0), dtype=torch.long)
        dec_tgt_padded = torch.zeros((len(batch), 0), dtype=torch.long)
    else:
        dec_in_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
        dec_tgt_padded = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)

    init_decoder_inputs = torch.LongTensor(init_decoder_inputs)

    return enc_padded, enc_mask_padded, dec_in_padded, dec_tgt_padded, init_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_ids'] for item in batch]
    encoder_masks = [item['encoder_mask'] for item in batch]
    init_decoder_inputs = [item['init_decoder_input'] for item in batch]

    enc_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    enc_mask_padded = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    init_decoder_inputs = torch.LongTensor(init_decoder_inputs)

    return enc_padded, enc_mask_padded, init_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))

    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))

    test_x = load_lines(os.path.join(data_folder, 'test.nl'))

    return train_x, train_y, dev_x, dev_y, test_x