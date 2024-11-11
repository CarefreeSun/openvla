"""
single_finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

#import pizzadataset
from pizza_dataset import PizzaDataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on

def get_logger(file_path):
    rf_handler = logging.StreamHandler()
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if file_path is not None:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(rf_handler)
    if file_path is not None: logger.addHandler(f_handler)
    
    return logger
    


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    pizza_dir: Path = Path("/mnt/data-rundong/robot_datasets/pizza/task_04/pizza_dataset_dataset/1.0.0/")                        # Path to Pizza dataset directory
    dataset_name: str = "pizza"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("/mnt/data-rundong/openvla/runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("/mnt/data-rundong/openvla/adapter-tmp")                     # Temporary directory for LoRA weights before fusing
    task_id: str = "04"                                       # Task ID for Pizza dataset finetune

    # Fine-tuning Parameters
    epochs: int = 30                                                 # Number of fine-tuning epochs
    batch_size: int = 8                                        # Fine-tuning batch size
    max_steps: int =  318990                                        # Max number of fine-tuning steps
    save_steps: int = 1000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                             # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 16                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under

    # fmt: on
    # cuda Parameters
    device: str = "0"                                               # Device id of target cuda device


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    log_path = os.path.join(cfg.run_root_dir, f"id{cfg.task_id}-lr{cfg.learning_rate}.log")
    logger = get_logger(log_path)
    logger.info(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    # print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    # distributed_state = PartialState()
    device_id = "cuda:" + cfg.device
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    task_id = cfg.task_id
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    # for params in vla.parameters():
    #     torch.nn.utils.clip_grad_value_(params, 0.001)
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    # batch_transform = RLDSBatchTransform(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    pizza_data = PizzaDataset(
        cfg.pizza_dir,
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    # vla_dataset = RLDSDataset(
    #     cfg.data_root_dir,
    #     cfg.dataset_name,
    #     batch_transform,
    #     resize_resolution=tuple(vla.config.image_sizes),
    #     # resize_resolution=tuple(vla.module.config.image_sizes),
    #     shuffle_buffer_size=cfg.shuffle_buffer_size,
    #     image_aug=cfg.image_aug,
    # )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    
    dataset_run_dir = os.path.join(run_dir, task_id)
    os.makedirs(dataset_run_dir, exist_ok=True)
    # save_dataset_statistics(vla_dataset.dataset_statistics, dataset_run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        pizza_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=collator,
        num_workers=4,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    # logger.info(f"Loaded {len(vla_dataset)} samples from {cfg.dataset_name} dataset compose of a dataloader with batch size {cfg.batch_size} and {cfg.grad_accumulation_steps} gradient accumulation steps, total dataloader steps {len(dataloader)}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    for epoch in tqdm.tqdm(range(cfg.epochs)):
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
            # action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if batch_idx % 10 == 0:
                logger.info(f"train_loss: {smoothened_loss}, action_accuracy: {smoothened_action_accuracy}, l1_loss: {smoothened_l1_loss}, step: {gradient_step_idx}")
            

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                
                logger.info(f"Skip saving for now ... ")
                logger.info(f"Current Step: {gradient_step_idx}")
                # logger.info(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                # print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                # # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                # save_dir = adapter_dir if cfg.use_lora else run_dir
                # save_dir = os.path.join(save_dir, task_id)

                # # Save Processor & Weights
                # processor.save_pretrained(os.path.join(save_dir, f'Task_id{task_id}-B{cfg.batch_size}-{cfg.dataset_name}-{cfg.learning_rate}-{gradient_step_idx}'))
                # # vla.module.save_pretrained(os.path.join(save_dir, f'Task_id{task_id}-B{cfg.batch_size}-{cfg.dataset_name}-{cfg.learning_rate}-{gradient_step_idx}'))
                # vla.save_pretrained(os.path.join(save_dir, f'Task_id{task_id}-B{cfg.batch_size}-{cfg.dataset_name}-{cfg.learning_rate}-{gradient_step_idx}'))


                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
        if cfg.use_lora:
            logger.info(f"Saving Model Checkpoint for epoch {epoch}")
            save_dir = adapter_dir if cfg.use_lora else run_dir
            save_dir = os.path.join(save_dir, task_id)
            processor.save_pretrained(os.path.join(save_dir, f'task{task_id}-b{cfg.batch_size}-{cfg.dataset_name}-{cfg.learning_rate}-{epoch}'))
            # vla.module.save_pretrained(os.path.join(save_dir, f'Task_id{task_id}-B{cfg.batch_size}-{cfg.dataset_name}-{cfg.learning_rate}-{gradient_step_idx}'))
            vla.save_pretrained(os.path.join(save_dir, f'task{task_id}-b{cfg.batch_size}-{cfg.dataset_name}-{cfg.learning_rate}-{epoch}'))
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            save_dir = os.path.join(adapter_dir, task_id)
            merged_vla = PeftModel.from_pretrained(base_vla, os.path.join(save_dir, f'task{task_id}-b{cfg.batch_size}-{cfg.dataset_name}-{cfg.learning_rate}-{epoch}'))
            merged_vla = merged_vla.merge_and_unload()
            
            model_param_dir = os.path.join(run_dir, task_id, "merged")
            os.makedirs(model_param_dir, exist_ok=True)
            processor.save_pretrained(model_param_dir)
            merged_vla.save_pretrained(model_param_dir)
        



if __name__ == "__main__":
    #set cuda visible device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    finetune()
    
