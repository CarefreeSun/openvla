from typing import Union
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import tensorflow_datasets as tfds

from mani_skill.utils.io_utils import load_json
from mani_skill.utils import sapien_utils
from mani_skill.utils import common
from prismatic.vla.datasets import RLDSBatchTransform
from dataclasses import dataclass

class ManiSkillToPizzaDataset(Dataset):
    """
    将 ManiSkill 轨迹数据转换为 PizzaDataset 格式
    """
    
    def __init__(self, dataset_file: str, data_dir: str, action_tokenizer, processor, image_transform, prompt_builder_fn, load_count=-1, success_only=False, device=None) -> None:
        self.data_dir = data_dir
        self.batchTransform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform,
            prompt_builder_fn)
        
        # 加载 ManiSkill 数据集
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        
        self.data = []
        
        # 加载和处理轨迹数据
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            if success_only:
                if not eps.get("success", False):
                    continue
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            eps_len = len(trajectory["actions"])
            
            # 排除最后一个观测
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            
            # 将数据转换为 PizzaDataset 格式
            for idx in range(eps_len - 1):
                data_pack = {}
                data_pack["dataset_name"] = "ManiSkill"
                data_pack["action"] = [trajectory["actions"][idx]]
                data_pack["observation"] = {}
                data_pack["observation"]["image_primary"] = [obs["image"][idx]]  # 假设图像在观测中
                data_pack["task"] = {}
                data_pack["task"]["language_instruction"] = eps.get("language_instruction", "No instruction")  # 假设有语言指令
                
                self.data.append(data_pack)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.batchTransform(self.data[idx])

if __name__ == "__main__":
    from transformers import AutoProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    import time

    # 初始化模型和处理器
    vla_path = '/path/to/vla_param'
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # 初始化数据集
    data_dir = '/path/to/pizza_dataset/'
    dataset_file = '/path/to/mani_skill_data.h5'
    
    dataset = ManiSkillToPizzaDataset(
        dataset_file=dataset_file,
        data_dir=data_dir,
        action_tokenizer=action_tokenizer,
        processor=processor,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
    )
    
    print(len(dataset))
    time_start = time.time()
    
    # 遍历数据集
    for i in range(len(dataset)):
        print(dataset[i])
    
    print("时间消耗:", time.time() - time_start)

