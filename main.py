from dataclasses import dataclass, field, asdict, fields
import argparse
import json
from pathlib import Path
from typing import Optional
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm
import wandb

import torch
from cs336_basics.transformer.model import TransformerModel
from cs336_basics.optim.adamw import AdamW
from cs336_basics.optim.schedulers import WarmupCosineScheduler
from cs336_basics.data.dataset import create_dataloader
from cs336_basics.losses.crossentropy import crossentropy
from cs336_basics.utils.checkpoint import save_checkpoint

@dataclass
class TransformerModelTrainingArgs():
    normalization: str = field(default="pre", metadata={'help': 'which form of normalization to use: pre, post, off'})
    activation: str = field(default="swiglu", metadata={'help': 'Activation function in feed forward modules: swiglu, silu'})
    device: str = field(default="cpu", metadata={'help': 'device to use: cpu, cuda, mps'})
    vocab_size: int = field(default=50257, metadata={'help': 'vocabulary size'})
    context_length: int = field(default=1024, metadata={'help': 'maximum sequence length'})
    d_model: int = field(default=768, metadata={'help': 'model dimension'})
    num_layers: int = field(default=12, metadata={'help': 'number of transformer layers'})
    num_heads: int = field(default=12, metadata={'help': 'number of attention heads'})
    d_ff: int = field(default=3072, metadata={'help': 'feed-forward dimension'})
    rope_theta: float = field(default=10000.0, metadata={'help': 'RoPE theta parameter'})
    batch_size: int = field(default=32, metadata={'help': 'training batch size'})
    with_scheduler: bool = field(default=True, metadata={'help': 'use learning rate scheduler'})
    lr: float = field(default=3e-4, metadata={'help': 'learning rate'})
    lr_min: float = field(default=1e-5, metadata={'help': 'minimum learning rate for cosine schedule'})
    lr_warmup_iters: int = field(default=2000, metadata={'help': 'number of warmup iterations'})
    lr_cosine_iters: int = field(default=10000, metadata={'help': 'number of cosine decay iterations'})
    num_iters: int = field(default=10000, metadata={'help': 'total number of training iterations'})
    train_data_path: str = field(default="data/train.bin", metadata={'help': 'path to training data'})
    valid_data_path: str = field(default="data/valid.bin", metadata={'help': 'path to validation data'})
    checkpoint_interval: int = field(default=1000, metadata={'help': 'save checkpoint every N iterations'})
    output_path: str = field(default="outputs/checkpoints", metadata={'help': 'directory to save checkpoints'})
    logs_path: str = field(default="outputs/logs", metadata={'help': 'directory to save logs'})
    with_wandb: bool = field(default=False, metadata={'help': 'enable Weights & Biases logging'})
    wandb_project_name: str = field(default="transformer-training", metadata={'help': 'W&B project name'})

    @classmethod
    def from_json(cls, json_path: str) -> 'TransformerModelTrainingArgs':
        """Load arguments from a JSON configuration file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
        return cls(**config)

    def to_json(self, json_path: str) -> None:
        """Save arguments to a JSON configuration file."""
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_command_line(cls, args: Optional[argparse.Namespace] = None) -> 'TransformerModelTrainingArgs':
        """Parse arguments from command-line, optionally loading base config from JSON."""
        parser = argparse.ArgumentParser(description='Train a Transformer language model')

        # Add config file argument
        parser.add_argument('--config', type=str, default=None,
                          help='path to JSON configuration file (other args override config file values)')

        # Add all dataclass fields as command-line arguments
        for f in fields(cls):
            arg_name = f'--{f.name}'
            field_type = f.type
            help_text = f.metadata.get('help', '')

            # Handle boolean fields specially
            if field_type == bool:
                parser.add_argument(arg_name, action='store_true', default=None, help=help_text)
                parser.add_argument(f'--no_{f.name}', action='store_false', dest=f.name, help=f'disable {f.name}')
            else:
                parser.add_argument(arg_name, type=field_type, default=None, help=help_text)

        # Parse arguments
        if args is None:
            parsed_args = parser.parse_args()
        else:
            parsed_args = args

        # Start with base config from JSON if provided
        if parsed_args.config is not None:
            config = cls.from_json(parsed_args.config)
            config_dict = asdict(config)
        else:
            config_dict = {}

        # Override with command-line arguments (only non-None values)
        for f in fields(cls):
            arg_value = getattr(parsed_args, f.name, None)
            if arg_value is not None:
                config_dict[f.name] = arg_value

        return cls(**config_dict)


def setup_logging(args: TransformerModelTrainingArgs, run_name: str) -> logging.Logger:
    """Setup file logging to logs directory."""
    logs_dir = Path(args.logs_path) / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "training.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Logs directory: {logs_dir}")
    return logger


def load_data(data_path: str) -> np.ndarray:
    """Load .npy data file using memory mapping from current working directory."""
    full_path = Path.cwd() / data_path

    # Resolve symlinks to get the actual file path
    resolved_path = full_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")

    data = np.load(resolved_path, mmap_mode='r')
    return data

def train(args: TransformerModelTrainingArgs):
    """Main training loop."""
    # Create unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    logger = setup_logging(args, run_name)

    logger.info("Training configuration:")
    logger.info(json.dumps(asdict(args), indent=2))

    # Initialize wandb if enabled
    if args.with_wandb:
        try:
            wandb_api_key = ""
            if wandb_api_key:
                wandb.login(key=wandb_api_key)

            # Initialize wandb with full config
            config_dict = asdict(args)
            wandb.init(
                project=args.wandb_project_name,
                name=run_name,
                config=config_dict
            )

            # Log configuration as summary for easy comparison
            for key, value in config_dict.items():
                wandb.run.summary[f"config/{key}"] = value

            logger.info("Weights & Biases logging enabled")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            args.with_wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            args.with_wandb = False

    # Load data
    logger.info(f"Loading training data from {args.train_data_path}")
    train_data = load_data(args.train_data_path)
    logger.info(f"Loading validation data from {args.valid_data_path}")
    valid_data = load_data(args.valid_data_path)

    # Create dataloaders
    get_train_batch = create_dataloader(
        train_data,
        args.context_length,
        args.device
    )
    get_valid_batch = create_dataloader(
        valid_data,
        args.context_length,
        args.device
    )

    # Initialize model
    logger.info("Initializing model")
    model = TransformerModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        normalization=args.normalization
    )
    model.to(args.device)
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr
    )

    # Initialize scheduler if enabled
    scheduler = None
    if args.with_scheduler:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_iters=args.lr_warmup_iters,
            cosine_cycle_iters=args.lr_cosine_iters,
            lr_min=args.lr_min
        )
        logger.info("Learning rate scheduler enabled")

    # Create checkpoint directory with run name
    checkpoint_dir = Path(args.output_path) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints directory: {checkpoint_dir}")

    # Training loop
    logger.info("Starting training")
    model.train()
    iteration = 0
    pb = tqdm(range(args.num_iters))
    while iteration < args.num_iters:
        try:
            # Get next batch
            try:
                x_batch, y_batch = get_train_batch(args.batch_size)
            except StopIteration:
                # Reset iterator when dataset is exhausted
                x_batch, y_batch = get_train_batch(args.batch_size)

            # Forward pass
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = crossentropy(logits, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = args.lr

            # Log metrics
            logger.info(f"Iteration {iteration}/{args.num_iters} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
            pb.update()
            pb.set_postfix_str(f"Iteration {iteration}/{args.num_iters} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")

            if args.with_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": current_lr
                })

            # Validation
            if iteration % 500 == 0 and iteration > 0 or iteration == args.num_iters-1:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for _ in range(150):
                        val_x, val_y = get_valid_batch(args.batch_size)
                        val_logits = model(val_x)
                        val_loss = crossentropy(val_logits, val_y)
                        valid_losses.append(val_loss.item())

                avg_valid_loss = np.mean(valid_losses)
                logger.info(f"Validation Loss: {avg_valid_loss:.4f}")

                if args.with_wandb:
                    wandb.log({
                        "valid/loss": avg_valid_loss
                    })

                model.train()

            # Save checkpoint
            if iteration % args.checkpoint_interval == 0 and iteration > 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
                save_checkpoint(model, optimizer, iteration, str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            iteration += 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error during training at iteration {iteration}: {e}")
            raise

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, iteration, str(final_checkpoint_path))
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")

    if args.with_wandb:
        wandb.finish()

    logger.info("Training completed")


if __name__ == '__main__':
    # Parse arguments from command-line (with optional JSON config)
    args = TransformerModelTrainingArgs.from_command_line()

    # Run training
    train(args)