import torch
import os
import sys
import json
import glob
import logging
import yaml
import argparse
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import BCDataset
from yaml import safe_load
from model import GCN
from utils.evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test a model on the BC dataset.")
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

if __name__ == "__main__":
    args = parse_args()
    with open("config/testing.yaml", "r") as f:
        t_config = safe_load(f)
    with open("config/model.yaml", "r") as f:
        m_config = safe_load(f)

    logging.info("Starting testing process...")
    logging.info("Testing configuration:\n%s", yaml.dump(t_config, sort_keys=False))
    logging.info("Model configuration:\n%s", yaml.dump(m_config, sort_keys=False))

    dataset = BCDataset(
        type=t_config["dataset"]["type"],
        **t_config["dataset"]["kwargs"]
    )

    task = m_config["model"]["type"]
    checkpoint = f"{t_config['pretrained']['checkpoint']}/{task}/*_best.pt"
    if len(glob.glob(checkpoint)) == 0:
        raise FileNotFoundError(f"No checkpoint found for task {task} at {checkpoint}")
    elif len(glob.glob(checkpoint)) > 1:
        logging.warning(f"Multiple checkpoints found for task {task}. Using the first one.")
        checkpoint = glob.glob(checkpoint)[0]
    else:
        checkpoint = glob.glob(checkpoint)[0]

    logging.info(f"Checkpoint found: {checkpoint}")
    result_path = f"{t_config['results']['path']}/{task}"
    device = t_config["device"]
    if isinstance(device, str):
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")
    train_mask, val_mask, test_mask = dataset.get_masks()
    percentiles = t_config["evaluation"]["percentiles"]
    data = dataset.to_torch_data().to(device)
    with open(f"{t_config['results']['path']}/{task}/{task}_training_results.json", "r") as f:
        studies = json.load(f)
    config = studies["Parameters"] 
    config.pop("lr", None)
    config.pop("n_epochs", None)

    if task == "GCN":
        model = GCN(
            edge_index=data.edge_index,
            in_channels=data.num_node_features,
            output_dim=2,
            **config
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {task}")
    logging.info(f"Loading model from {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logging.info("Model loaded successfully.")

    logging.info("Starting evaluation...")
    results = evaluate(
        model=model,
        data=data,
        test_mask=test_mask,
        percentile_q_list=percentiles,
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