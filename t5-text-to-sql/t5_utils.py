import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

try:
    import wandb
except ImportError:
    wandb = None

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

T5_NAME = "google-t5/t5-small"


def setup_wandb(args):
    # wandb is optional in this project; we log metrics via stdout / slurm logs.
    pass


def initialize_model(args):
    '''
    --finetune  : load pretrained T5-small weights for fine-tuning.
    otherwise   : randomly initialize a T5-small with the same config (Extra Credit 2-1).
    '''
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(T5_NAME)
    else:
        config = T5Config.from_pretrained(T5_NAME)
        model = T5ForConditionalGeneration(config)
    model.to(DEVICE)
    return model


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def _checkpoint_subdir(checkpoint_dir, best):
    return os.path.join(checkpoint_dir, "best" if best else "last")


def save_model(checkpoint_dir, model, best):
    target = _checkpoint_subdir(checkpoint_dir, best)
    mkdir(target)
    model.save_pretrained(target)


def load_model_from_checkpoint(args, best):
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    target = _checkpoint_subdir(checkpoint_dir, best)
    model = T5ForConditionalGeneration.from_pretrained(target)
    model.to(DEVICE)
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
        raise NotImplementedError(f"Unknown optimizer_type: {args.optimizer_type}")

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
