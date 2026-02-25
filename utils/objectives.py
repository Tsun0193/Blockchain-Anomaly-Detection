import os
import inspect

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torch_geometric.data import Data


def GNN_features(
    graph: Data,
    model: nn.Module,
    lr: float,
    n_epochs: int,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    train_mask: torch.Tensor = None,
    val_mask: torch.Tensor = None,
    test_mask: torch.Tensor = None,
    selection_split: str = "val",
    evaluate_test: bool = True,
    **kwargs
):
    device = kwargs.get('device', 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    graph = graph.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kwargs.get('weight_decay', 5e-4))
    criterion = nn.CrossEntropyLoss()  # Define loss function.
    
    train_losses = []
    val_scores = []
    val_losses = []

    if val_mask is not None and test_mask is not None:
        has_overlap = torch.logical_and(val_mask.bool(), test_mask.bool()).any().item()
        assert not has_overlap, "Sanity check failed: val_mask and test_mask must be disjoint."
    
    def train_epoch():
        model.train()
        total_loss = 0.0

        if train_loader is None:
            optimizer.zero_grad()
            y_hat = model(graph.x, graph.edge_index.to(device))
            loss = criterion(y_hat[train_mask], graph.y[train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()
        else:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                y_hat = model(batch.x, batch.edge_index.to(device))[: batch.batch_size]
                y_true = batch.y[: batch.batch_size]
                loss = criterion(y_hat, y_true)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_loader)

    def evaluate(loader, mask):
        model.eval()
        if loader is None:
            if mask is None or mask.sum().item() == 0:
                return None, None
            with torch.no_grad():
                y_hat = model(graph.x, graph.edge_index.to(device))
                logits = y_hat[mask]
                labels = graph.y[mask]
                probs = logits.softmax(dim=1)
                loss_val = criterion(logits, labels).item()
        else:
            all_probs = []
            all_labels = []
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    y_hat = model(batch.x, batch.edge_index)[: batch.batch_size]
                    labels = batch.y[: batch.batch_size]
                    total_loss += criterion(y_hat, labels).item() * labels.size(0)
                    total_count += labels.size(0)
                    all_probs.append(y_hat.softmax(dim=1).detach().cpu())
                    all_labels.append(labels.detach().cpu())
            if total_count == 0:
                return None, None
            loss_val = total_loss / total_count
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)

        if labels.numel() == 0:
            return None, None

        try:
            ap = average_precision_score(labels.numpy(), probs.numpy()[:, 1])
        except Exception:
            preds = probs.argmax(dim=1)
            ap = (preds == labels).sum().item() / labels.size(0)
        return ap, loss_val

    for _ in range(n_epochs):
        loss_train = train_epoch()
        train_losses.append(loss_train)

        if (val_mask is not None) or (val_loader is not None):
            ap_val, loss_val = evaluate(val_loader, val_mask)
        else:
            ap_val = None
            loss_val = None
        val_scores.append(ap_val)
        val_losses.append(loss_val)

    ap_test = None
    if evaluate_test and ((test_mask is not None and test_mask.sum().item() > 0) or (test_loader is not None)):
        ap_test, _ = evaluate(test_loader, test_mask)

    valid_val_scores = [score for score in val_scores if score is not None]
    best_val_score = max(valid_val_scores) if len(valid_val_scores) > 0 else None

    if selection_split == "val":
        if best_val_score is None:
            raise ValueError("Validation metric is required for Optuna objective, but no val split is available.")
        score = best_val_score
        objective_source = "val"
    elif selection_split == "test":
        if ap_test is None:
            raise ValueError("selection_split='test' requires available test metrics.")
        score = ap_test
        objective_source = "test"
    elif selection_split in (None, "none"):
        score = None
        objective_source = "none"
    else:
        raise ValueError(f"Unsupported selection_split: {selection_split}")
    
    # Optional inline plotting
    plot_path = kwargs.get('plot_path', None)
    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        epochs = list(range(1, n_epochs + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), tight_layout=True)
        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss")
        ax1.set_title("Training Loss vs. Epoch")

        ax2.plot(epochs, val_scores, label="Validation Score")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation AP/Acc")
        ax2.set_title("Validation Score vs. Epoch")

        fig.savefig(plot_path)
        plt.close(fig)

    history = {
        "train_loss": train_losses,
        "val_score": val_scores,
        "val_loss": val_losses,
        "best_val_score": best_val_score,
    }

    return {
        "score": score,
        "objective_source": objective_source,
        "val_score": best_val_score,
        "test_score": ap_test,
        "history": history
    }


def objective_gnn(trial, model_cls, model_kwargs=None, **kwargs):
    """
    Universal Optuna objective for GNN models (GCN, GAT, etc.)
    Returns validation score, and logs history into trial.user_attrs.
    """
    def _get(name, suggest_fn):
        return kwargs[name] if name in kwargs else suggest_fn()

    graph = kwargs['graph']
    result_path = kwargs.get('result_path', 'results/gnn')
    os.makedirs(result_path, exist_ok=True)

    # Suggested hyperparameters
    hidden_dim     = _get('hidden_dim',     lambda: trial.suggest_int('hidden_dim', 128, 256))
    embedding_dim  = _get('embedding_dim',  lambda: trial.suggest_int('embedding_dim', 64, 128))
    num_layers     = _get('num_layers',     lambda: trial.suggest_int('num_layers', 1, 3))
    lr             = _get('lr',             lambda: trial.suggest_float('lr', 2e-6, 1e-3, log=True))
    n_epochs       = _get('n_epochs',       lambda: trial.suggest_int('n_epochs', 128, 512))
    dropout        = _get('dropout',        lambda: trial.suggest_float('dropout', 0.08, 0.64, log=True))
    weight_decay   = _get('weight_decay',   lambda: trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True))
    aggregator     = _get('aggregator',     lambda: trial.suggest_categorical('aggregator', ['mean', 'max']))
    graphnorm      = bool(kwargs.get('graphnorm', False))
    init_mode      = kwargs.get('init_mode', 'xavier')

    # Static model config
    model_kwargs = model_kwargs or {}
    base_kwargs = {
        "edge_index": graph.edge_index,
        "in_channels": graph.num_features,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "output_dim": 2,
        "num_layers": num_layers,
        "dropout": dropout,
        "graphnorm": graphnorm,
        "init_mode": init_mode,
        "aggregator": aggregator,
        **model_kwargs,
    }
    signature = inspect.signature(model_cls.__init__)
    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in signature.parameters}
    model = model_cls(**filtered_kwargs)

    result = GNN_features(
        graph=graph,
        model=model,
        lr=lr,
        n_epochs=n_epochs,
        train_mask=kwargs['masks'][0],
        val_mask=kwargs['masks'][1],
        test_mask=kwargs['masks'][2],
        device=kwargs.get('device', 'cpu'),
        weight_decay=weight_decay,
        selection_split="val",
        evaluate_test=kwargs.get("log_test_metric", True)
    )

    model_path = os.path.join(result_path, f"{model_cls.__name__.lower()}_trial_{trial.number}.pt")
    torch.save(model.state_dict(), model_path)

    trial.set_user_attr("model_state_path", model_path)
    trial.set_user_attr("history", result["history"])
    trial.set_user_attr("val_score", result["val_score"])
    trial.set_user_attr("test_score", result["test_score"])
    trial.set_user_attr("objective_source", result["objective_source"])

    assert result["objective_source"] == "val", "Optuna objective must be based on validation metric."
    assert result["score"] == result["val_score"], "Objective return must match validation score."

    return result["score"]
