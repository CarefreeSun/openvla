from typing import Union
import re
import gzip
import json
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import torch
import tensorflow_datasets as tfds

from prismatic.vla.datasets import RLDSBatchTransform
from dataclasses import dataclass

def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def index_dict_array(x1, idx: Union[int, slice], inplace=True):
    """Indexes every array in x1 with slice and returns result."""
    if (
        isinstance(x1, np.ndarray)
        or isinstance(x1, list)
        or isinstance(x1, torch.Tensor)
    ):
        return x1[idx]
    elif isinstance(x1, dict):
        if inplace:
            for k in x1.keys():
                x1[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return x1
        else:
            out = dict()
            for k in x1.keys():
                out[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return out

def remove_np_uint16(x: Union[np.ndarray, dict]):
    if isinstance(x, dict):
        for k in x.keys():
            x[k] = remove_np_uint16(x[k])
        return x
    else:
        if x.dtype == np.uint16:
            return x.astype(np.int32)
        return x
    
class ManiSkillToPizzaDataset(Dataset):
    """
    将 ManiSkill 轨迹数据转换为 PizzaDataset 格式
    """
    
    def __init__(self, dataset_file: str, action_tokenizer, processor, image_transform, prompt_builder_fn, load_count=-1, success_only=False, device=None) -> None:
        self.batchTransform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform,
            prompt_builder_fn)
        
        # 加载 ManiSkill 数据集
        self.dataset_file = dataset_file
        # self.device = device
        self.h5data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        
        #并不有用
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        def convert_env_id_to_language_instruction(env_id):
            """
            PlugCharger-v1 -> Plug charger
            """
            # 移除版本号 (-v1, -v2, 等)
            env_id = env_id.split("-")[0]
            # 按大写字母分隔，转为小写并用空格连接
            language_instruction = " ".join([word.lower() for word in re.findall(r'[A-Z][^A-Z]*', env_id)])
            return language_instruction

        self.language_instruction = convert_env_id_to_language_instruction(self.env_id)
        self.data = []
        
        # 加载和处理轨迹数据
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]

            if success_only:
                assert "success" in eps, "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                if not eps.get("success", False):
                    continue
            trajectory = self.h5data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            eps_len = len(trajectory["actions"])
            
            obs = index_dict_array(trajectory["obs"], slice(eps_len))
            obs = remove_np_uint16(obs)

            for idx in range(eps_len - 1):
                data_pack = {}
                data_pack["dataset_name"] = "ManiSkill"
                data_pack["action"] = [trajectory["actions"][idx]]
                data_pack["observation"] = {}
                data_pack["observation"]["image_primary"] = [obs["sensor_data"]["base_camera"]["rgb"]]  # 假设图像在观测中
                data_pack["task"] = {}
                data_pack["task"]["language_instruction"] = self.language_instruction  # 假设有语言指令
            
                self.data.append(data_pack)
                
                # handle data that might optionally be in the trajectory
                '''
                if "rewards" in trajectory:
                    if self.rewards is None:
                        self.rewards = [trajectory["rewards"]]
                    else:
                        self.rewards.append(trajectory["rewards"])
                if "success" in trajectory:
                    if self.success is None:
                        self.success = [trajectory["success"]]
                    else:
                        self.success.append(trajectory["success"])
                if "fail" in trajectory:
                    if self.fail is None:
                        self.fail = [trajectory["fail"]]
                    else:
                        self.fail.append(trajectory["fail"])
                
                if self.rewards is not None:
                    self.rewards = np.concatenate(self.rewards)
                if self.success is not None:
                    self.success = np.concatenate(self.success)
                if self.fail is not None:
                    self.fail = np.concatenate(self.fail)

                self.terminated.append(trajectory["terminated"])
                self.truncated.append(trajectory["truncated"])
                '''
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.batchTransform(self.data[idx])

if __name__ == "__main__":
    from transformers import AutoProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    import time

    vla_path = None #'/home/yunqiliu/vlatune/transfer/openvla_param'
    processor = None #AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = None #ActionTokenizer(processor.tokenizer)
    dataset_file = '/home/v-wenhuitan/hs/simplerEnv/data_maniskill/PlugCharger-v1/motionplanning/trajectory.rgb.pd_joint_pos.cpu.h5'
    
    dataset = ManiSkillToPizzaDataset(
        dataset_file=dataset_file,
        action_tokenizer=action_tokenizer,
        processor=processor,
        image_transform=None, #processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
    )
    
    print(len(dataset))
    time_start = time.time()
    
    # 遍历数据集
    for i in range(len(dataset)):
        print(dataset[i])
    
    print("时间消耗:", time.time() - time_start)

