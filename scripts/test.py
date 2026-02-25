import argparse
import glob
import inspect
import json
import logging
import os
import sys

import pandas as pd
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yaml import safe_load

from data.dataset import BCDataset
from model import GAT, GCN, SAGE
from utils.evaluate import evaluate

MODEL_REGISTRY = {
    "GCN": GCN,
    "GAT": GAT,
    "SAGE": SAGE
}

DEFAULT_EXPERIMENT_PRESETS = {
    "baseline": {
        "graphnorm": False,
        "init_mode": "default",
        "results_root": "results",
        "checkpoint_root": "checkpoints",
    },
    "xavier_only": {
        "graphnorm": False,
        "init_mode": "xavier",
        "results_root": "results_no_graphnorm",
        "checkpoint_root": "checkpoints_no_graphnorm",
    },
    "graphnorm_xavier": {
        "graphnorm": True,
        "init_mode": "xavier",
        "results_root": "results_with_graphnorm",
        "checkpoint_root": "checkpoints_with_graphnorm",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test a model on the BC dataset.")
    parser.add_argument(
        "--testing-config",
        type=str,
        default="config/testing.yaml",
        help="Path to testing config YAML.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level.upper())
    return args


def _resolve_experiment_setting(config):
    experiment_cfg = config.get("experiment", {})
    preset_name = experiment_cfg.get("setting", "baseline")

    presets = {k: dict(v) for k, v in DEFAULT_EXPERIMENT_PRESETS.items()}
    for key, value in (experiment_cfg.get("presets") or {}).items():
        if key not in presets:
            presets[key] = {}
        if isinstance(value, dict):
            presets[key].update(value)

    if preset_name not in presets:
        raise ValueError(
            f"Unknown experiment.setting='{preset_name}'. "
            f"Available presets: {sorted(presets.keys())}"
        )

    selected = presets[preset_name]
    fallback_checkpoint_root = config.get("pretrained", {}).get("checkpoint", "checkpoints")
    fallback_results_root = config.get("results", {}).get("path", "results")
    return {
        "name": preset_name,
        "graphnorm": bool(selected.get("graphnorm", False)),
        "init_mode": str(selected.get("init_mode", "xavier")),
        "results_root": str(selected.get("results_root", fallback_results_root)),
        "checkpoint_root": str(selected.get("checkpoint_root", fallback_checkpoint_root)),
    }


def _resolve_device(config):
    runtime_cfg = config.get("runtime", {})
    gpu_id = runtime_cfg.get("gpu_id", None)
    allowed_gpu_ids = runtime_cfg.get("available_cuda_ids", [0, 1, 2, 3])

    if gpu_id is not None:
        if int(gpu_id) not in [int(x) for x in allowed_gpu_ids]:
            raise ValueError(f"runtime.gpu_id={gpu_id} is not in available_cuda_ids={allowed_gpu_ids}")
        device_str = f"cuda:{int(gpu_id)}"
    else:
        device_str = runtime_cfg.get("device", config.get("device", "cpu"))

    if isinstance(device_str, str) and device_str.startswith("cuda:"):
        gpu_idx = int(device_str.split(":", 1)[1])
        if gpu_idx not in [int(x) for x in allowed_gpu_ids]:
            raise ValueError(f"Requested device '{device_str}' is not in available_cuda_ids={allowed_gpu_ids}")
        if not torch.cuda.is_available():
            logging.warning("CUDA requested (%s) but not available. Falling back to CPU.", device_str)
            return torch.device("cpu")
        if gpu_idx >= torch.cuda.device_count():
            raise ValueError(
                f"Requested device '{device_str}' but only {torch.cuda.device_count()} CUDA devices are visible."
            )
    return torch.device(device_str)

if __name__ == "__main__":
    args = parse_args()
    with open(args.testing_config, "r") as f:
        t_config = safe_load(f)
    with open(args.model_config, "r") as f:
        m_config = safe_load(f)

    logging.info("Starting testing process...")
    logging.info("Testing configuration:\n%s", yaml.dump(t_config, sort_keys=False))
    logging.info("Model configuration:\n%s", yaml.dump(m_config, sort_keys=False))

    dataset = BCDataset(
        type=t_config["dataset"]["type"],
        **t_config["dataset"]["kwargs"]
    )

    task = m_config["model"]["type"]
    if task not in MODEL_REGISTRY:
        logging.error(f"Unsupported model type: {task}")
        sys.exit(1)

    model = MODEL_REGISTRY[task]

    experiment_setting = _resolve_experiment_setting(t_config)
    checkpoint = f"{experiment_setting['checkpoint_root']}/{task}/*_best.pt"
    if len(glob.glob(checkpoint)) == 0:
        raise FileNotFoundError(f"No checkpoint found for task {task} at {checkpoint}")
    elif len(glob.glob(checkpoint)) > 1:
        logging.warning(f"Multiple checkpoints found for task {task}. Using the first one.")
        checkpoint = glob.glob(checkpoint)[0]
    else:
        checkpoint = glob.glob(checkpoint)[0]

    logging.info(f"Checkpoint found: {checkpoint}")
    result_path = f"{experiment_setting['results_root']}/{task}"
    device = _resolve_device(t_config)

    logging.info(f"Using device: {device}")
    logging.info(
        "Experiment setting: %s | graphnorm=%s | init_mode=%s",
        experiment_setting["name"],
        experiment_setting["graphnorm"],
        experiment_setting["init_mode"],
    )
    
    _train_mask, val_mask, test_mask = dataset.get_masks()
    assert not torch.logical_and(val_mask, test_mask).any().item(), "val_mask and test_mask must be disjoint."
    
    percentiles = t_config["evaluation"]["percentiles"]
    data = dataset.to_torch_data().to(device)
    # data.x = data.x[:, 1:94]
    
    with open(f"{result_path}/{task}_training_results.json", "r") as f:
        studies = json.load(f)
    config = studies["Parameters"] 
    config = dict(config)
    config.pop("lr", None)
    config.pop("n_epochs", None)
    config.pop("weight_decay", None)
    graphnorm = bool(studies.get("GraphNorm", experiment_setting["graphnorm"]))
    init_mode = str(studies.get("InitMode", experiment_setting["init_mode"]))

    base_model_kwargs = {
        "edge_index": data.edge_index,
        "in_channels": data.num_features,
        "output_dim": 2,
        "graphnorm": graphnorm,
        "init_mode": init_mode,
        **config
    }
    signature = inspect.signature(model.__init__)
    filtered_model_kwargs = {k: v for k, v in base_model_kwargs.items() if k in signature.parameters}
    dropped_keys = sorted(set(base_model_kwargs.keys()) - set(filtered_model_kwargs.keys()))
    if dropped_keys:
        logging.info("Dropping unsupported model kwargs for %s: %s", task, dropped_keys)

    logging.info(f"Loading model from {checkpoint}")
    model = model(
        **filtered_model_kwargs
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    # deep_train(data, model, train_mask, n_epochs, lr, batch_size=128, loader=None)
    print(model)
    logging.info("Model loaded successfully.")

    logging.info("Starting evaluation...")
    results = evaluate(
        model=model,
        data=data,
        test_mask=test_mask,
        percentile_q_list=percentiles,
        n_samples=100,
        device=device
    )
    auc_list, ap_list, precision_dict, recall_dict, f1_dict = results
    logging.info("Evaluation completed.")
    logging.info("Testing process completed successfully.")

    # Save results TI
    os.makedirs(f"{result_path}", exist_ok=True)
    res_TI = {
        'AUC': auc_list,
        'AP': ap_list
    }
    df_TI = pd.DataFrame(res_TI)
    df_TI.to_csv(os.path.join(result_path, "results_TI.csv"), index=False)

    # save results TD
    res_TD = dict()
    for key in precision_dict.keys():
        res_TD['Precision_' + str(key)] = precision_dict[key]
        res_TD['Recall_' + str(key)] = recall_dict[key]
        res_TD['F1_Score_' + str(key)] = f1_dict[key]
    df_TD = pd.DataFrame(res_TD)
    df_TD.to_csv(os.path.join(result_path, "results_TD.csv"), index=False)
