import tensorflow_datasets as tfds

import torch
from torch.utils.data import Dataset
import numpy as np
import os

from prismatic.vla.datasets import RLDSBatchTransform
from tqdm import tqdm

from dataclasses import dataclass

class PizzaDataset(Dataset):
    def __init__(self, data_dir, action_tokenizer, processtokenier, image_transform, prompt_builder_fn):
        self.data_dir = data_dir
        self.batchTransform = RLDSBatchTransform(
            action_tokenizer,
            processtokenier,
            image_transform, 
            prompt_builder_fn)
        
        builder = tfds.builder_from_directory(data_dir)
        
        self.data = []
        ds = builder.as_dataset(split="train")
        for example in tqdm(ds):
            steps = example['steps']
            for window in steps.as_numpy_iterator():
                data_pack = {}
                data_pack["dataset_name"] = "PIZZADATASET"
                data_pack['action'] = [window['action']]
                data_pack["observation"] = {}
                data_pack["observation"]["image_primary"] = [window['observation']['image']]
                data_pack["task"] = {}
                data_pack["task"]["language_instruction"] = window['language_instruction']
                self.data.append(data_pack)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.batchTransform(self.data[idx])
    
class PizzaDatasetGrip(Dataset):
    def __init__(self, data_dir, action_tokenizer, processtokenier, image_transform, prompt_builder_fn, action_threshold = 0.005):
        self.data_dir = data_dir
        self.batchTransform = RLDSBatchTransform(
            action_tokenizer,
            processtokenier,
            image_transform, 
            prompt_builder_fn)
        
        builder = tfds.builder_from_directory(data_dir)
        
        self.data = []
        ds = builder.as_dataset(split="train")
        for example in tqdm(ds):
            steps = example['steps']
            for window in steps.as_numpy_iterator():
                if np.abs(window['action'][6]) >= action_threshold:
                    data_pack = {}
                    data_pack["dataset_name"] = "PIZZADATASET"
                    data_pack['action'] = [window['action']]
                    data_pack["observation"] = {}
                    data_pack["observation"]["image_primary"] = [window['observation']['image']]
                    data_pack["task"] = {}
                    data_pack["task"]["language_instruction"] = window['language_instruction']
                    self.data.append(data_pack)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
    import time
    vla_path = '/home/yunqiliu/vlatune/transfer/openvla_param'
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    data_dir = '/home/yunqiliu/vlatune/transfer/sep_pizza_dataset/task_12/pizza_dataset_dataset/1.0.0/'
    
    dst = PizzaDataset(data_dir, action_tokenizer, processor.tokenizer, processor.image_processor.apply_transform, PurePromptBuilder if "v01" not in vla_path else VicunaV15ChatPromptBuilder)
    print(len(dst))
    time_start = time.time()
    for i in range(len(dst)):
        print(dst[i])
    print("time cost: ", time.time() - time_start)
        
    
    
        