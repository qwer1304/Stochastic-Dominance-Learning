from typing import Any
from orbax.checkpoint import PyTreeCheckpointHandler, Checkpointer, CheckpointManager, type_handlers
from flax.training import train_state
from flax import struct
from ml_collections import config_dict as mlc
import numpy as np

@struct.dataclass
class SDTrainState(train_state.TrainState):
    xepoch: Any = None
    xstep: Any = None
    batch_stats: Any = None
    buffer_state: Any = None

class TrainableModel(object):
    def __init__(self, config):
        self.config = config
        
def to_plain_dict(obj):
    if isinstance(obj, mlc.ConfigDict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_plain_dict(v) for v in obj)
    else:
        return obj


def to_config_dict_recursive(d):
    if isinstance(d, dict):
        return mlc.ConfigDict({k: to_config_dict_recursive(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_config_dict_recursive(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(to_config_dict_recursive(v) for v in d)
    else:
        return d

def flatten_and_convert_metrics(metrics_dict, parent_key="", sep="/"):

    def convert(value, key_path):
        if isinstance(value, dict):
            return flatten_and_convert_metrics(value, parent_key=key_path, sep=sep)

        elif isinstance(value, (np.ndarray,)):
            return float(value.item()) if value.ndim == 0 else value.tolist()

        elif hasattr(value, "tolist"):  # covers JAX arrays
            value = value.tolist()
            return float(value) if isinstance(value, (float, int)) else value

        elif isinstance(value, (float, int)):
            return float(value)

        elif isinstance(value, (list, tuple)):
            # Recursively convert each item
            return [convert(v, f"{key_path}[{i}]") for i, v in enumerate(value)]

        raise TypeError(f"Unsupported metric type at key '{key_path}': {type(value)}")

    flat_metrics = {}
    for k, v in metrics_dict.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else k
        converted = convert(v, full_key)
        if isinstance(converted, dict):
            flat_metrics.update(converted)
        else:
            flat_metrics[full_key] = converted

    return flat_metrics
