import os
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

#import pizzadataset
from pizza_dataset import PizzaDataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_logger(file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    rf_handler = logging.StreamHandler()
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(rf_handler)

    if file_path:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(f_handler)
    
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
    batch_size: int = 4                                           # Fine-tuning batch size
    max_steps: int =  318990                                        # Max number of fine-tuning steps
    save_steps: int = 1000                                          # Interval for checkpoint saving
    learning_rate: float = 1e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 2                             # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 16                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    log_path = os.path.join(cfg.run_root_dir, f"id{cfg.task_id}-lr{cfg.learning_rate}.log")
    logger = get_logger(log_path)
    logger.info(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
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
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    pizza_data = PizzaDataset(
        cfg.pizza_dir,
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

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

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    for epoch in tqdm.tqdm(range(cfg.epochs), desc="Epochs"):
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches"):
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
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                logger.info(f"train_loss: {smoothened_loss}, action_accuracy: {smoothened_action_accuracy}, l1_loss: {smoothened_l1_loss}, step: {gradient_step_idx}")
            

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if cfg.use_lora and distributed_state.is_main_process:
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
        
            # Block on Main Process Checkpointing
            dist.barrier()

if __name__ == "__main__":
    finetune()
