import pickle
import numpy as np
from pathlib import Path
from pprint import pprint
import yaml
from enum import Enum, auto


class PklType(Enum):
    STATS = auto()
    KEY_INFO = auto()

PKL_TYPE = PklType.STATS


def pkl_to_yaml(input_pkl, output_yaml=None):
    """
    将pkl文件转换为HDF5格式
    
    参数:
        input_pkl: 输入的pkl文件路径
        output_hdf5: 输出的HDF5文件路径(可选)
        compression: 压缩算法('gzip', 'lzf', None)
    """
    input_path = Path(input_pkl)
    if output_yaml is None:
        output_yaml = input_path.with_suffix('.yaml')
    
    # 读取原始pkl数据
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    pprint(data)

    if PKL_TYPE is PklType.KEY_INFO:
        data["all_config"]["start_action"] = data["all_config"]["start_action"].tolist()
        data["all_config"]["start_joint"] = data["all_config"]["start_joint"].tolist()
        data["init_info"]["init_action"] = data["init_info"]["init_action"].tolist()
        data["init_info"]["init_joint"] = data["init_info"]["init_joint"].tolist()
    elif PKL_TYPE is PklType.STATS:
        data["action_mean"] = data["action_mean"].tolist()
        data["action_std"] = data["action_std"].tolist()
        data["example_qpos"] = data["example_qpos"].tolist()
        data["qpos_mean"] = data["qpos_mean"].tolist()
        data["qpos_std"] = data["qpos_std"].tolist()

    # save to yaml
    with open(output_yaml, 'w') as f:
        yaml.dump(data, f)
    print(f"数据已保存到 {output_yaml}")


def yaml_to_pkl(input_yaml, output_pkl=None):
    """
    将yaml文件转换为pkl格式
    
    参数:
        input_yaml: 输入的yaml文件路径
        output_pkl: 输出的pkl文件路径(可选)
    """
    input_path = Path(input_yaml)
    if output_pkl is None:
        output_pkl = str(input_path).replace(".yaml", "_cvt.pkl")
    
    # 读取原始yaml数据
    with open(input_yaml, 'r') as f:
        data = yaml.safe_load(f)

    if PKL_TYPE is PklType.KEY_INFO:
        data["all_config"]["start_action"] = np.array(data["all_config"]["start_action"])
        data["all_config"]["start_joint"] = np.array(data["all_config"]["start_joint"])
        data["init_info"]["init_action"] = np.array(data["init_info"]["init_action"])
        data["init_info"]["init_joint"] = np.array(data["init_info"]["init_joint"])
    elif PKL_TYPE is PklType.STATS:
        data["action_mean"] = np.array(data["action_mean"])
        data["action_std"] = np.array(data["action_std"])
        data["example_qpos"] = np.array(data["example_qpos"])
        data["qpos_mean"] = np.array(data["qpos_mean"])
        data["qpos_std"] = np.array(data["qpos_std"])

    # save to pkl
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

    print(f"数据已保存到 {output_pkl}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将pkl文件转换为HDF5格式")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出的HDF5文件路径(可选)")
    
    args = parser.parse_args()

    if args.input.endswith('.yaml'):
        yaml_to_pkl(args.input, args.output)
    else:
        pkl_to_yaml(args.input, args.output)