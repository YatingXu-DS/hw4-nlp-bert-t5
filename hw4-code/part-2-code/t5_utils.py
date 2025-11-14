import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    try:
        import importlib
        _wandb = importlib.import_module('wandb')
        # Initialize a wandb run with a sensible default project name
        _wandb.init(project=getattr(args, 'wandb_project', 'text2sql'), name=getattr(args, 'experiment_name', None))
        # log args for reproducibility
        if hasattr(args, '__dict__'):
            _wandb.config.update(args.__dict__)
    except Exception:
        # If wandb isn't configured / available, continue without failing
        print("wandb setup failed or unavailable — continuing without wandb logging")

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 't5-small' checkpoint
    or training a T5 model initialized with the 't5-small' config
    from scratch.
    '''
    # If finetune flag is set, load pretrained weights; otherwise create from config
    model_name = 'google-t5/t5-small'
    if getattr(args, 'finetune', False):
        print(f"Loading pretrained T5 model '{model_name}' for finetuning")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        print(f"Initializing T5 model from config '{model_name}' (training from scratch)")
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)

    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    # Save the raw state_dict — small and flexible
    target = os.path.join(checkpoint_dir, 'best.pt' if best else 'last.pt')
    try:
        torch.save(model.state_dict(), target)
        print(f"Saved model state to {target}")
    except Exception as e:
        print(f"Failed to save model to {target}: {e}")

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    # Determine checkpoint directory
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    else:
        model_type = 'ft' if getattr(args, 'finetune', False) else 'scr'
        checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', getattr(args, 'experiment_name', 'experiment'))

    filename = 'best.pt' if best else 'last.pt'
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Initialize model architecture (matching how it was created originally)
    model = initialize_model(args)

    # Load state dict
    state = torch.load(path, map_location=DEVICE)
    # support both raw state_dict and wrapped dicts
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # try strict=False as a fallback (e.g., when training with DataParallel or minor name mismatches)
        print(f"Warning: strict load failed ({e}), retrying with strict=False")
        model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    print(f"Loaded model weights from {path}")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

