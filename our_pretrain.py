"""
our_pretrain.py

Pretraining script for Prismatic VLM pretraining in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed training across GPUs. By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - We're loading LLaMa-2 (and possibly other) gated models from HuggingFace (HF Hub); these require an auth_token.
      For LLaMa-2, make sure to first get Meta approval, then fill out the form at the top of the HF LLaMa-2 page:
        => Link: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
        => Generate Token (from `huggingface.co`): Settings / Access Tokens / New "Read" Token
        => Set `cfg.hf_token` to file path with token (as single line text file) or environment variable name

    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_dataset_and_collator
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

# Import methods from single_finetune.py
from pizza_dataset import PizzaDataset
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps logging.Logger
overwatch = initialize_overwatch(__name__)

@dataclass
class PretrainConfig:
    # fmt: off

    vla_path: str = "openvla/openvla-7b"  # 这里直接指定模型的路径
    pizza_dir: Path = Path("/mnt/data-rundong/robot_datasets/pizza/task_04/pizza_dataset_dataset/1.0.0/")  # 数据集路径

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("/mnt/fsx/x-prismatic-vlms/runs")     # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "onyx-vlms"                                # Name of W&B project (default: `prismatic`)
    wandb_entity: Optional[str] = "stanford-voltron"                # Name of W&B entity (default: None)

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    overwatch.info("Prismatic VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    dataset_id = cfg.dataset.dataset_id
    cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # === Load Model and Processor using single_finetune's method ===
    processor = AutoProcessor.from_pretrained(cfg.model.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device_id)

    # Initialize Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load dataset (PizzaDataset as used in single_finetune)
    pizza_data = PizzaDataset(
        cfg.dataset.pizza_dir,
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.model.model_id else VicunaV15ChatPromptBuilder,
    )

    # Data Collator and Loader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        pizza_data,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=model,
        device_id=device_id,
        stage=cfg.stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(pizza_data))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(dataloader, collator, metrics, stage=cfg.stage, seed=cfg.seed)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()
