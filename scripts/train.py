import argparse
import glob
import inspect
import json
import logging
import os
import sys
import warnings

import optuna
import torch
import yaml
from yaml import safe_load

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import BCDataset
from model import GAT, GCN, SAGE
from utils.objectives import GNN_features, objective_gnn

warnings.filterwarnings("ignore")

MODEL_REGISTRY = {
    "GCN": GCN,
    "GAT": GAT,
    "SAGE": SAGE,
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
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the BC dataset.")
    parser.add_argument(
        "--training-config",
        type=str,
        default="config/training.yaml",
        help="Path to training config YAML.",
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
        help="Set the logging level.",
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
    return {
        "name": preset_name,
        "graphnorm": bool(selected.get("graphnorm", False)),
        "init_mode": str(selected.get("init_mode", "xavier")),
        "results_root": str(selected.get("results_root", "results")),
        "checkpoint_root": str(selected.get("checkpoint_root", "checkpoints")),
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


def _build_model_from_hparams(model_cls, graph, hparams, experiment_options):
    required_keys = ("hidden_dim", "embedding_dim", "num_layers", "dropout")
    missing = [k for k in required_keys if k not in hparams]
    if missing:
        raise KeyError(f"Missing required model hyperparameters: {missing}")

    base_kwargs = {
        "edge_index": graph.edge_index,
        "in_channels": graph.num_features,
        "hidden_dim": hparams["hidden_dim"],
        "embedding_dim": hparams["embedding_dim"],
        "output_dim": 2,
        "num_layers": hparams["num_layers"],
        "dropout": hparams["dropout"],
        "graphnorm": experiment_options["graphnorm"],
        "init_mode": experiment_options["init_mode"],
        "aggregator": hparams.get("aggregator", "mean"),
    }
    signature = inspect.signature(model_cls.__init__)
    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in signature.parameters}
    return model_cls(**filtered_kwargs)


if __name__ == "__main__":
    args = parse_args()
    with open(args.training_config, "r") as f:
        t_config = safe_load(f)
    with open(args.model_config, "r") as f:
        m_config = safe_load(f)

    logging.info("Starting training process...")
    logging.info("Training configuration:\n%s", yaml.dump(t_config, sort_keys=False))
    logging.info("Model configuration:\n%s", yaml.dump(m_config, sort_keys=False))

    seed = int(t_config["training"].get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = BCDataset(
        type=t_config["dataset"]["type"],
        **t_config["dataset"]["kwargs"],
    )

    experiment_setting = _resolve_experiment_setting(t_config)
    device = _resolve_device(t_config)
    logging.info("Using device: %s", device)
    logging.info(
        "Experiment setting: %s | graphnorm=%s | init_mode=%s",
        experiment_setting["name"],
        experiment_setting["graphnorm"],
        experiment_setting["init_mode"],
    )

    split_ratio = t_config["training"].get("ratio", [0.8, 0.1, 0.1])
    if isinstance(split_ratio, (list, tuple)) and len(split_ratio) >= 2 and (split_ratio[0] + split_ratio[1]) > 0:
        val_ratio = float(split_ratio[1]) / float(split_ratio[0] + split_ratio[1])
    else:
        val_ratio = 0.2

    train_mask, val_mask, test_mask = dataset.get_masks(
        seed=seed,
        val_ratio=val_ratio,
        temporal_split=True,
    )
    assert not torch.logical_and(val_mask, test_mask).any().item(), "val_mask and test_mask must be disjoint."
    logging.info(
        "Mask split | train=%d val=%d test=%d",
        int(train_mask.sum().item()),
        int(val_mask.sum().item()),
        int(test_mask.sum().item()),
    )
    logging.info("Sanity check passed: validation and test masks are disjoint.")

    data = dataset.to_torch_data().to(device)

    task = m_config["model"]["type"]
    if task not in MODEL_REGISTRY:
        logging.error("Unsupported model type: %s", task)
        sys.exit(1)

    model = MODEL_REGISTRY[task]
    logging.info("Selected model: %s", task)

    def wrapped_objective(trial: optuna.Trial):
        return objective_gnn(
            trial,
            model_cls=model,
            graph=data,
            masks=(train_mask, val_mask, test_mask),
            device=device,
            result_path=f"{experiment_setting['checkpoint_root']}/{task}",
            weight_decay=t_config["optimizer"]["weight_decay"],
            log_test_metric=True,
            graphnorm=experiment_setting["graphnorm"],
            init_mode=experiment_setting["init_mode"],
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        wrapped_objective,
        n_trials=t_config["training"]["n_trials"],
    )

    best_trial = study.best_trial
    params = dict(best_trial.params)
    best_val_score = float(best_trial.value)
    logging.info("Best parameters: %s", params)
    logging.info("Best validation AUPRC (Optuna objective): %.6f", best_val_score)
    logging.info("Best-trial debug test AUPRC: %s", best_trial.user_attrs.get("test_score"))
    if best_trial.user_attrs.get("objective_source") != "val":
        raise AssertionError("Best trial objective_source must be 'val'.")

    final_hparams = dict(params)
    lr = final_hparams.pop("lr", t_config["optimizer"].get("lr", 1e-3))
    n_epochs = int(final_hparams.pop("n_epochs", t_config["training"].get("epochs", 256)))
    weight_decay = final_hparams.pop("weight_decay", t_config["optimizer"].get("weight_decay", 5e-4))

    final_model = _build_model_from_hparams(model, data, final_hparams, experiment_setting)
    final_train_mask = torch.logical_or(train_mask, val_mask).detach()
    final_result = GNN_features(
        graph=data,
        model=final_model,
        lr=lr,
        n_epochs=n_epochs,
        train_mask=final_train_mask,
        val_mask=None,
        test_mask=test_mask,
        device=device,
        weight_decay=weight_decay,
        selection_split="none",
        evaluate_test=True,
    )
    final_test_score = final_result["test_score"]
    logging.info("Final held-out test AUPRC (single post-Optuna evaluation): %s", final_test_score)

    result_dir = f"{experiment_setting['results_root']}/{task}"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{task}_training_results.json", "w") as f:
        json.dump(
            {
                "Task": task,
                "Parameters": params,
                "AUC_PRC": best_val_score,
                "BestValidationAUPRC": best_val_score,
                "FinalTestAUPRC": final_test_score,
                "ObjectiveMetric": "validation_auprc",
                "ExperimentSetting": experiment_setting["name"],
                "GraphNorm": experiment_setting["graphnorm"],
                "InitMode": experiment_setting["init_mode"],
                "Study": study.trials_dataframe().to_dict(orient="records"),
            },
            f,
            indent=4,
            default=str,
        )

    checkpoint_dir = os.path.join(experiment_setting["checkpoint_root"], task)
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint = os.path.join(checkpoint_dir, f"{task.lower()}_best.pt")
    torch.save(final_model.state_dict(), final_checkpoint)

    for fpath in glob.glob(os.path.join(checkpoint_dir, f"{task.lower()}_trial_*.pt")):
        if os.path.normpath(fpath).endswith(".pt"):
            os.remove(fpath)

    logging.info("Saved final checkpoint: %s", final_checkpoint)
    logging.info("✅ Results saved successfully.")
    logging.info("✅ Training phase completed successfully.")
