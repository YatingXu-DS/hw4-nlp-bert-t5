import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW")
    parser.add_argument('--learning_rate', type=float, default=3e-4)         
    parser.add_argument('--weight_decay', type=float, default=0.01)           
    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"])
    parser.add_argument('--num_warmup_epochs', type=int, default=1)          
    parser.add_argument('--max_n_epochs', type=int, default=10)               
    parser.add_argument('--patience_epochs', type=int, default=3)             

    # Data & misc
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--grad_clip', type=float, default=1.0)               
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='t5_finetune_exp')

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    # create AMP scaler if requested and GPU available
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler, scaler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            try:
                import importlib
                _wandb = importlib.import_module('wandb')
                _wandb.log(result_dict, step=epoch)
            except Exception:
                # wandb not available or not configured — skip logging
                pass

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler, scaler=None):
    """
    Single training epoch. Supports optional AMP via `scaler` and gradient clipping using
    `args.grad_clip` (set to 0.0 to disable clipping).
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    use_amp = (scaler is not None)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        non_pad = decoder_targets != PAD_IDX

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=decoder_input,
                )['logits']

                if torch.sum(non_pad).item() > 0:
                    loss = criterion(logits[non_pad], decoder_targets[non_pad])
                else:
                    # no tokens to compute loss on for this batch
                    continue

            # scale, backward, unscale for clipping, step
            scaler.scale(loss).backward()
            # unscale before clipping
            scaler.unscale_(optimizer)
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            if torch.sum(non_pad).item() > 0:
                loss = criterion(logits[non_pad], decoder_targets[non_pad])
            else:
                continue

            loss.backward()
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0

    # collect generated queries
    gen_sqls = []

    # create tokenizer for decoding
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, init_decoder_inputs in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # compute loss on dev set (no backward)
            if decoder_targets.numel() != 0:
                decoder_input = decoder_input.to(DEVICE)
                decoder_targets = decoder_targets.to(DEVICE)

                logits = model(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=decoder_input,
                )['logits']

                non_pad = decoder_targets != PAD_IDX
                if torch.sum(non_pad).item() > 0:
                    loss = criterion(logits[non_pad], decoder_targets[non_pad])
                    num_tokens = torch.sum(non_pad).item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

            # generation (greedy)
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=512,
                do_sample=False,        # greedy decoding
                num_beams=1,            # ensure greedy
            )

            # decode and store
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_sqls.extend([q.strip() for q in decoded])

    # save generated queries and computed records
    save_queries_and_records(gen_sqls, model_sql_path, model_record_path)

    # compute metrics
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)

    # error rate: fraction of generated queries that produced an error message
    if len(model_error_msgs) > 0:
        error_rate = sum(1 for e in model_error_msgs if e) / len(model_error_msgs)
    else:
        error_rate = 0.0

    eval_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    return eval_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()

    # create tokenizer for decoding
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    gen_sqls = []
    with torch.no_grad():
        for encoder_input, encoder_mask, init_decoder_inputs in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=512,
                do_sample=False,        # greedy decoding
                num_beams=1,            # ensure greedy
            )

            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_sqls.extend([q.strip() for q in decoded])

    # Save generated SQLs and their records
    save_queries_and_records(gen_sqls, model_sql_path, model_record_path)
    print(f"Saved {len(gen_sqls)} generated SQL queries to {model_sql_path} and records to {model_record_path}")

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    # if fp16 requested but no GPU, warn and fall back to fp32
    if args.fp16 and not torch.cuda.is_available():
        print("Warning: --fp16 requested but no CUDA available; running in fp32.")
        args.fp16 = False

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss:.6f}, Record F1: {dev_record_f1:.4f}, Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
