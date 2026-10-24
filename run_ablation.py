"""Run matched DeCo-LSTM ablations and single-component controls on London data.

The suite uses the same eligible households and training budget for every variant,
and saves per-household metrics for paired, block-clustered analysis across seeds.
"""

import argparse
import copy
import gc
import glob
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))
WINDOW_SIZE = 48
HORIZON = 1
TRAIN_SAMPLES = 400
TEST_SAMPLES = 600
TARGET_GROUPS = ["Affluent", "Comfortable", "Adversity"]
METHODS = [
    "DeCo-Full", "DeCo-noEWC", "DeCo-noReplay", "DeCo-noSimilarity",
    "DeCo-noGate", "Matched-EWC", "Matched-Replay", "Matched-Similarity",
]
ABLATION_CONFIGS = {
    "DeCo-Full": {
        "track_fisher": True, "ewc_penalty": True, "replay": True,
        "similarity": True, "gate": True, "lambda_ewc": 45.0,
        "replay_size": 180, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": False,
    },
    "DeCo-noEWC": {
        "track_fisher": True, "ewc_penalty": False, "replay": True,
        "similarity": True, "gate": True, "lambda_ewc": 45.0,
        "replay_size": 180, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": False,
    },
    "DeCo-noReplay": {
        "track_fisher": True, "ewc_penalty": True, "replay": False,
        "similarity": True, "gate": True, "lambda_ewc": 45.0,
        "replay_size": 0, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": False,
    },
    "DeCo-noSimilarity": {
        "track_fisher": True, "ewc_penalty": True, "replay": True,
        "similarity": False, "gate": True, "lambda_ewc": 45.0,
        "replay_size": 180, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": True,
    },
    "DeCo-noGate": {
        "track_fisher": True, "ewc_penalty": True, "replay": True,
        "similarity": True, "gate": False, "lambda_ewc": 45.0,
        "replay_size": 180, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": False,
    },
    "Matched-EWC": {
        "track_fisher": True, "ewc_penalty": True, "replay": False,
        "similarity": False, "gate": False, "lambda_ewc": 45.0,
        "replay_size": 0, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": True,
    },
    "Matched-Replay": {
        "track_fisher": False, "ewc_penalty": False, "replay": True,
        "similarity": False, "gate": False, "lambda_ewc": 0.0,
        "replay_size": 180, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": True,
    },
    "Matched-Similarity": {
        "track_fisher": False, "ewc_penalty": False, "replay": False,
        "similarity": True, "gate": False, "lambda_ewc": 0.0,
        "replay_size": 0, "top_k": 8, "alpha_base": 0.65,
        "head_local_keep": 0.88, "random_peers": False,
    },
}
RESULT_COLUMNS = [
    "Method", "AE-RMSE (↓)", "AE-MAE (↓)", "BWT-RMSE (↓)", "BWT-MAE (↓)"
]
SEASONS = [
    ("T1_Winter", "2012-12-01", "2013-02-28 23:30:00"),
    ("T2_Spring", "2013-03-01", "2013-05-31 23:30:00"),
    ("T3_Summer", "2013-06-01", "2013-08-31 23:30:00"),
    ("T4_Autumn", "2013-09-01", "2013-11-30 23:30:00"),
]


class ContinualLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1):
        super().__init__()
        self.feature_extractor = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.feature_extractor(x)
        return self.head(out[:, -1, :])


def prepare_dataset(data_dir="data/london", max_clients_per_group=None,
                    inject_synthetic_drift=False):
    """Read all block files and retain compact per-season series for eligible homes."""
    data_dir = Path(data_dir)
    info_path = data_dir / "informations_households.csv"
    blocks_dir = data_dir / "halfhourly_dataset"
    if not info_path.is_file() or not blocks_dir.is_dir():
        raise FileNotFoundError(
            f"London CSV files not found under {data_dir}; refusing to substitute synthetic data."
        )

    info = pd.read_csv(info_path, usecols=["LCLid", "Acorn_grouped"])
    info = info[info["Acorn_grouped"].isin(TARGET_GROUPS)].drop_duplicates("LCLid")
    acorn_by_id = info.set_index("LCLid")["Acorn_grouped"].to_dict()
    group_index = {name: i for i, name in enumerate(TARGET_GROUPS)}
    collected = []
    counts = {group: 0 for group in TARGET_GROUPS}
    seen = set()
    block_files = glob.glob(str(blocks_dir / "block_*.csv"))
    block_files.sort(key=lambda p: int(Path(p).stem.rsplit("_", 1)[1]))
    if len(block_files) != 112:
        print(f"[!] Expected 112 half-hourly blocks, found {len(block_files)}")

    print(f"[*] Reading {len(block_files)} blocks; client cap per group: {max_clients_per_group or 'none'}")
    for block_no, filename in enumerate(block_files, start=1):
        frame = pd.read_csv(
            filename,
            usecols=["LCLid", "tstp", "energy(kWh/hh)"],
            na_values=["Null"],
        )
        frame["tstp"] = pd.to_datetime(frame["tstp"], errors="coerce")
        frame = frame[
            frame["tstp"].between("2012-12-01", "2013-11-30 23:30:00", inclusive="both")
        ]

        for household, group_frame in frame.groupby("LCLid", sort=False):
            if household in seen:
                continue
            group = acorn_by_id.get(household)
            if group is None:
                continue
            if max_clients_per_group is not None and counts[group] >= max_clients_per_group:
                continue

            group_frame = group_frame.sort_values("tstp").set_index("tstp")
            series = pd.to_numeric(group_frame["energy(kWh/hh)"], errors="coerce")
            n_obs = int(len(series))
            missing_fraction = float(series.isna().mean()) if n_obs else 1.0
            if n_obs < 17000 or missing_fraction > 0.01:
                continue

            series = series.interpolate(method="linear").bfill().ffill()
            winter = series.loc[SEASONS[0][1]:SEASONS[0][2]].to_numpy(dtype=np.float64).reshape(-1, 1)
            scaler = StandardScaler().fit(winter)
            task_data = {}
            for task_id, (season_name, start, end) in enumerate(SEASONS):
                values = series.loc[start:end].to_numpy(dtype=np.float64).reshape(-1, 1)
                values = scaler.transform(values).astype(np.float32, copy=False).reshape(-1)
                if inject_synthetic_drift:
                    t = np.arange(len(values), dtype=np.float32)
                    values = values + 0.45 * np.sin(
                        2 * np.pi * t / WINDOW_SIZE + task_id * 1.4 + group_index[group] * 2.1
                    ).astype(np.float32)
                task_data[task_id] = {"series": values, "season": season_name}

            collected.append({
                "LCLid": household,
                "acorn": group,
                "g_idx": group_index[group],
                "source_block": Path(filename).stem,
                "n_obs": n_obs,
                "missing_fraction": missing_fraction,
                "tasks": task_data,
            })
            counts[group] += 1
            seen.add(household)

        del frame
        gc.collect()
        if block_no % 8 == 0 or block_no == len(block_files):
            print(f"  [blocks {block_no:03}/{len(block_files)}] clients={len(collected):,} by_group={counts}", flush=True)

    collected.sort(key=lambda c: (c["g_idx"], c["LCLid"]))
    if not collected:
        raise RuntimeError("No eligible households found in the London half-hourly data.")
    print(f"[*] Prepared {len(collected):,} eligible households from {len(block_files)} blocks; by_group={counts}")
    return collected


def make_xy(series, take="train"):
    """Materialize only the requested first-train or last-test windows."""
    width = WINDOW_SIZE + HORIZON
    windows = np.lib.stride_tricks.sliding_window_view(series, width)
    if take == "train":
        selected = windows[:min(TRAIN_SAMPLES, len(windows))]
    else:
        selected = windows[-min(TEST_SAMPLES, len(windows)):]
    x = np.array(selected[:, :WINDOW_SIZE, None], dtype=np.float32, order="C", copy=True)
    y = np.array(selected[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON], dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(x), torch.from_numpy(y)


def compute_daily_signature(x_train):
    mean_window = x_train.detach().cpu().numpy()[:, :, 0].mean(axis=0)
    signature = (mean_window - mean_window.mean()) / (mean_window.std() + 1e-6)
    return torch.as_tensor(signature, dtype=torch.float32)


def get_feature_vector(model, x_train):
    """Build a peer descriptor from model weights and the client's load profile."""
    weights = torch.cat([
        p.detach().cpu().reshape(-1)
        for name, p in model.named_parameters()
        if "feature_extractor" in name
    ])
    weights = weights / (torch.linalg.vector_norm(weights) + 1e-8)
    signature = compute_daily_signature(x_train)
    signature = signature / (torch.linalg.vector_norm(signature) + 1e-8)
    return torch.cat([0.35 * weights, 0.65 * signature])


def compute_fisher_matrix(model, x_train, y_train, batch_size=64):
    fisher = {name: torch.zeros_like(p, device=DEVICE) for name, p in model.named_parameters()}
    # cuDNN LSTMs require training mode for backward(). Fisher estimation needs
    # gradients, even though it does not update the optimizer or model weights.
    was_training = model.training
    model.train()
    criterion = nn.MSELoss()
    total = 0
    try:
        for start in range(0, len(x_train), batch_size):
            xb = x_train[start:start + batch_size].to(DEVICE)
            yb = y_train[start:start + batch_size].to(DEVICE)
            model.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            bs = len(xb)
            total += bs
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square() * bs
    finally:
        model.train(was_training)
    for name in fisher:
        fisher[name] /= max(1, total)
    return fisher


def update_running_omega(old_omega, new_fisher, gamma_fisher=0.75):
    if old_omega is None:
        return copy.deepcopy(new_fisher)
    return {
        name: gamma_fisher * old_omega[name] + (1.0 - gamma_fisher) * new_fisher[name]
        for name in new_fisher
    }


def compute_similarity_matrix(feature_vectors, tau_sim=0.25):
    vectors = np.stack([v.detach().cpu().numpy() for v in feature_vectors]).astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-8)
    similarities = vectors @ vectors.T
    np.maximum(similarities, 0.0, out=similarities)
    similarities[similarities < tau_sim] = 0.0
    np.fill_diagonal(similarities, 0.0)
    return similarities


def make_random_peer_matrix(count, top_k, seed):
    """Build a fixed uniform-weight peer graph for the no-similarity control."""
    graph = np.zeros((count, count), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for client_id in range(count):
        candidate_count = count - 1
        if candidate_count == 0:
            continue
        peers = rng.choice(candidate_count, size=min(top_k, candidate_count), replace=False)
        peers += peers >= client_id
        graph[client_id, peers] = 1.0
    return graph


def forgetting_aware_aggregation(models, omegas, similarity, top_k=8,
                                 alpha_base=0.65, use_memory_gate=True,
                                 head_local_keep=None):
    count = len(models)
    new_states = [copy.deepcopy(model.state_dict()) for model in models]
    states = [model.state_dict() for model in models]
    for i in range(count):
        scores = similarity[i]
        valid = np.flatnonzero(scores > 0)
        if not len(valid):
            continue
        if len(valid) > top_k:
            neighbors = valid[np.argpartition(scores[valid], -top_k)[-top_k:]]
        else:
            neighbors = valid
        weights = scores[neighbors]
        beta = weights / (weights.sum() + 1e-8)

        omega_i = omegas[i]
        normalized_omega = {}
        if use_memory_gate and omega_i is not None:
            all_values = torch.cat([value.reshape(-1) for value in omega_i.values()])
            omega_min, omega_max = all_values.min(), all_values.max()
            denominator = omega_max - omega_min + 1e-8
            normalized_omega = {
                name: torch.clamp((value - omega_min) / denominator, 0.0, 1.0)
                for name, value in omega_i.items()
            }

        for name, local_value in states[i].items():
            if "head" in name and head_local_keep is not None:
                local_keep = head_local_keep
            else:
                local_keep = 0.88 if use_memory_gate and "head" in name else alpha_base
            peer_value = torch.zeros_like(local_value)
            for index, peer in enumerate(neighbors):
                peer_value.add_(states[int(peer)][name], alpha=float(beta[index]))
            if use_memory_gate and omega_i is not None and name in normalized_omega:
                alpha = local_keep + (1.0 - local_keep) * normalized_omega[name]
            else:
                alpha = local_keep
            new_states[i][name] = alpha * local_value + (1.0 - alpha) * peer_value

    for i, model in enumerate(models):
        model.load_state_dict(new_states[i])


def evaluate_models(models, dataset, task_id, return_per_client=False):
    rmses, maes = [], []
    for client_id, model in enumerate(models):
        x_test, y_test = make_xy(dataset[client_id]["tasks"][task_id]["series"], "test")
        model.eval()
        with torch.no_grad():
            prediction = model(x_test.to(DEVICE))
            truth = y_test.to(DEVICE)
            rmses.append(torch.mean((prediction - truth).square()).sqrt().item())
            maes.append(torch.mean(torch.abs(prediction - truth)).item())
        del x_test, y_test
    mean_rmse, mean_mae = float(np.mean(rmses)), float(np.mean(maes))
    if return_per_client:
        return (
            mean_rmse, mean_mae,
            np.asarray(rmses, dtype=np.float32),
            np.asarray(maes, dtype=np.float32),
        )
    return mean_rmse, mean_mae


def train_client(model, x_train, y_train, optimizer, epochs, batch_size,
                 omega=None, star_weights=None, lambda_ewc=0.0):
    model.train()
    criterion = nn.MSELoss()
    for _ in range(epochs):
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = x_train.index_select(0, indices).to(DEVICE)
            yb = y_train.index_select(0, indices).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if omega is not None and star_weights is not None and lambda_ewc:
                penalty = sum(
                    (1.0 if "feature_extractor" in name else 0.3)
                    * torch.sum(omega[name] * (parameter - star_weights[name]).square())
                    for name, parameter in model.named_parameters()
                )
                loss = loss + 0.5 * lambda_ewc * penalty
            loss.backward()
            optimizer.step()


def run_method(method_name, dataset, base_init_state, comm_rounds=3,
               local_epochs=3, batch_size=64, seed=42):
    started = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    count = len(dataset)
    models = []
    for _ in range(count):
        model = ContinualLSTM().to(DEVICE)
        model.load_state_dict(copy.deepcopy(base_init_state))
        models.append(model)

    omegas = [None] * count
    star_weights = [None] * count
    replay_buffers = [{"X": None, "y": None} for _ in range(count)]
    result_rmse = np.zeros((len(SEASONS), len(SEASONS)), dtype=np.float32)
    result_mae = np.zeros_like(result_rmse)
    result_rmse_clients = np.full(
        (len(SEASONS), len(SEASONS), count), np.nan, dtype=np.float32
    )
    result_mae_clients = np.full_like(result_rmse_clients, np.nan)
    saved_similarity = None

    config = ABLATION_CONFIGS[method_name]
    track_fisher = config["track_fisher"]
    use_ewc_penalty = config["ewc_penalty"]
    use_replay = config["replay"]
    use_similarity = config["similarity"]
    use_gate = config["gate"]
    lambda_ewc = config["lambda_ewc"]
    replay_size = config["replay_size"]
    use_comm = True
    random_similarity = (
        make_random_peer_matrix(count, config["top_k"], seed + 991)
        if config["random_peers"] else None
    )

    for task_id in range(len(SEASONS)):
        print(f"  [{method_name}] {SEASONS[task_id][0]}: materializing per-client train windows", flush=True)
        current_data = [
            make_xy(client["tasks"][task_id]["series"], "train")
            for client in dataset
        ]
        train_data = []
        for i, (x_current, y_current) in enumerate(current_data):
            x_train, y_train = x_current, y_current
            if use_replay and replay_buffers[i]["X"] is not None:
                repeat = 2
                x_train = torch.cat([x_train, replay_buffers[i]["X"].repeat_interleave(repeat, dim=0)])
                y_train = torch.cat([y_train, replay_buffers[i]["y"].repeat_interleave(repeat, dim=0)])
            train_data.append((x_train, y_train))

        rounds_this_task = comm_rounds + (1 if task_id == 0 else 0)
        for round_id in range(rounds_this_task):
            feature_vectors = []
            for i, model in enumerate(models):
                x_train, y_train = train_data[i]
                optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-3)
                train_client(
                    model, x_train, y_train, optimizer, local_epochs, batch_size,
                    omega=omegas[i] if use_ewc_penalty and task_id > 0 else None,
                    star_weights=star_weights[i] if use_ewc_penalty and task_id > 0 else None,
                    lambda_ewc=lambda_ewc if use_ewc_penalty and task_id > 0 else 0.0,
                )
                feature_vectors.append(get_feature_vector(model, current_data[i][0]))
                if (i + 1) % 500 == 0:
                    print(f"    round {round_id + 1}/{rounds_this_task}: client {i + 1}/{count}", flush=True)

            if use_comm:
                if use_similarity:
                    print(f"    [{method_name}] building similarity matrix for round {round_id + 1}", flush=True)
                    similarity = compute_similarity_matrix(feature_vectors, tau_sim=0.25)
                    if task_id == len(SEASONS) - 1 and round_id == rounds_this_task - 1:
                        saved_similarity = similarity.copy()
                    forgetting_aware_aggregation(
                        models, omegas, similarity, top_k=config["top_k"],
                        alpha_base=config["alpha_base"],
                        use_memory_gate=use_gate,
                        head_local_keep=config["head_local_keep"],
                    )
                    del similarity
                else:
                    forgetting_aware_aggregation(
                        models, omegas, random_similarity, top_k=config["top_k"],
                        alpha_base=config["alpha_base"], use_memory_gate=use_gate,
                        head_local_keep=config["head_local_keep"],
                    )

        for i, model in enumerate(models):
            star_weights[i] = {name: p.detach().clone() for name, p in model.named_parameters()}
            if track_fisher:
                omega_new = compute_fisher_matrix(model, *train_data[i], batch_size=batch_size)
                omegas[i] = update_running_omega(omegas[i], omega_new)
                del omega_new
            if use_replay:
                x_current, y_current = current_data[i]
                take = min(replay_size, len(x_current))
                indices = np.random.choice(len(x_current), size=take, replace=False)
                indices = torch.as_tensor(indices, dtype=torch.long)
                x_sample = x_current.index_select(0, indices).clone()
                y_sample = y_current.index_select(0, indices).clone()
                if replay_buffers[i]["X"] is None:
                    replay_buffers[i]["X"], replay_buffers[i]["y"] = x_sample, y_sample
                else:
                    replay_buffers[i]["X"] = torch.cat([replay_buffers[i]["X"], x_sample])
                    replay_buffers[i]["y"] = torch.cat([replay_buffers[i]["y"], y_sample])

        for evaluated_task in range(task_id + 1):
            (
                result_rmse[task_id, evaluated_task],
                result_mae[task_id, evaluated_task],
                result_rmse_clients[task_id, evaluated_task],
                result_mae_clients[task_id, evaluated_task],
            ) = evaluate_models(
                models, dataset, evaluated_task, return_per_client=True
            )
            print(
                f"    [{method_name}] after T{task_id + 1} test T{evaluated_task + 1}: "
                f"RMSE={result_rmse[task_id, evaluated_task]:.4f}, "
                f"MAE={result_mae[task_id, evaluated_task]:.4f}",
                flush=True,
            )

        del train_data, current_data
        gc.collect()

    ae_rmse = float(np.mean(result_rmse[-1, :]))
    ae_mae = float(np.mean(result_mae[-1, :]))
    bwt_rmse = float(np.mean([
        result_rmse[-1, j] - result_rmse[j, j] for j in range(len(SEASONS) - 1)
    ]))
    bwt_mae = float(np.mean([
        result_mae[-1, j] - result_mae[j, j] for j in range(len(SEASONS) - 1)
    ]))
    print(
        f"  + {method_name} done in {(time.time() - started) / 60:.1f} min | "
        f"AE-RMSE={ae_rmse:.4f} | BWT-RMSE={bwt_rmse:+.4f}",
        flush=True,
    )
    return {
        "Method": method_name,
        "AE-RMSE (↓)": round(ae_rmse, 4),
        "AE-MAE (↓)": round(ae_mae, 4),
        "BWT-RMSE (↓)": round(bwt_rmse, 4),
        "BWT-MAE (↓)": round(bwt_mae, 4),
        "R_rmse": result_rmse,
        "R_mae": result_mae,
        "R_rmse_clients": result_rmse_clients,
        "R_mae_clients": result_mae_clients,
        "S_matrix": saved_similarity,
    }


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_dataframe_atomic(dataframe, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_npy_atomic(path, value, allow_pickle=False):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.save(stream, value, allow_pickle=allow_pickle)
    os.replace(temporary, path)


def locate_resume_dir(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume source does not exist: {path}")
    if (path / "table1_partial_results.csv").is_file():
        return path
    candidates = list(path.rglob("table1_partial_results.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one table1_partial_results.csv under {path}, found {len(candidates)}."
        )
    return candidates[0].parent


def copy_resume_artifacts(source_dir, output_dir):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if source_dir == output_dir:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        destination = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/london")
    parser.add_argument("--output-dir", default=None,
                        help="Default: results_ablation_seed<seed>")
    parser.add_argument("--max-clients-per-group", type=int, default=None,
                        help="Optional cap; default uses every eligible household.")
    parser.add_argument("--comm-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None,
                        help="Ablations/baselines to run; default runs the full suite.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed; use a separate output folder per seed.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip methods already present in this output directory.")
    parser.add_argument("--resume-from", default=None,
                        help="Copy prior notebook output artifacts from this directory, then resume.")
    parser.add_argument("--inject-synthetic-drift", action="store_true",
                        help="Add the artificial seasonal sinusoid used by the earlier draft.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"results_ablation_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_enabled = args.resume or args.resume_from is not None
    if args.resume_from is not None:
        source_dir = locate_resume_dir(args.resume_from)
        copy_resume_artifacts(source_dir, output_dir)
        print(f"[*] Restored prior outputs from {source_dir}", flush=True)

    print(f"[*] Device: {DEVICE}; threads={torch.get_num_threads()}", flush=True)
    dataset = prepare_dataset(
        args.data_dir,
        max_clients_per_group=args.max_clients_per_group,
        inject_synthetic_drift=args.inject_synthetic_drift,
    )
    selection = pd.DataFrame([
        {key: client[key] for key in ["LCLid", "acorn", "source_block", "n_obs", "missing_fraction"]}
        for client in dataset
    ])
    write_dataframe_atomic(selection, output_dir / "client_selection.csv")
    torch.manual_seed(args.seed)
    base_model = ContinualLSTM().to(DEVICE)
    base_init_state = copy.deepcopy(base_model.state_dict())
    methods = args.methods or METHODS
    print(f"[*] Methods selected: {', '.join(methods)}", flush=True)

    # Refuse a resume that would combine results from different training settings.
    # The ordered household ID digest also catches a changed client selection.
    run_signature = {
        "experiment_version": "deco_ablation_v1",
        "eligible_clients": len(dataset),
        "client_ids_sha256": hashlib.sha256(
            "\n".join(selection["LCLid"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "max_clients_per_group": args.max_clients_per_group,
        "comm_rounds": args.comm_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "methods": methods,
        "seed": args.seed,
        "method_configs": {method: ABLATION_CONFIGS[method] for method in methods},
        "inject_synthetic_drift": args.inject_synthetic_drift,
    }
    signature_path = output_dir / "run_signature.json"
    partial_path = output_dir / "table1_partial_results.csv"
    if resume_enabled and signature_path.is_file():
        with open(signature_path, encoding="utf-8") as stream:
            previous_signature = json.load(stream)
        if previous_signature != run_signature:
            raise ValueError(
                "Resume settings do not match the saved run. Keep the same data/client "
                "selection, methods, seed, communication rounds, epochs, batch size, and drift setting."
            )
    elif resume_enabled and partial_path.is_file():
        print(
            "[!] Checkpoint has no run_signature.json (created by an older script); "
            "verify manually that its data and training settings match this run.",
            flush=True,
        )
    write_json_atomic(signature_path, run_signature)

    logs = {}
    table_rows = []

    if resume_enabled and partial_path.is_file():
        prior_rows = pd.read_csv(partial_path).to_dict(orient="records")
        requested = set(methods)
        for row in prior_rows:
            method = str(row["Method"])
            if method not in requested:
                continue
            rmse_path = output_dir / f"{method.replace('-', '_')}_R_rmse.npy"
            if not rmse_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume {method}: missing {rmse_path.name}"
                )
            client_rmse_path = output_dir / f"{method.replace('-', '_')}_R_rmse_clients.npy"
            client_mae_path = output_dir / f"{method.replace('-', '_')}_R_mae_clients.npy"
            if not client_rmse_path.is_file() or not client_mae_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume {method}: per-client metric checkpoint files are missing."
                )
            similarity_path = output_dir / f"{method.replace('-', '_')}_similarity.npy"
            similarity = np.load(similarity_path) if similarity_path.is_file() else None
            normalized_row = {key: row[key] for key in [
                "Method", "AE-RMSE (↓)", "AE-MAE (↓)", "BWT-RMSE (↓)", "BWT-MAE (↓)"
            ]}
            table_rows.append(normalized_row)
            logs[method] = {
                **normalized_row,
                "R_rmse": np.load(rmse_path),
                "R_mae": np.load(output_dir / f"{method.replace('-', '_')}_R_mae.npy"),
                "R_rmse_clients": np.load(client_rmse_path),
                "R_mae_clients": np.load(client_mae_path),
                "S_matrix": similarity,
            }
        if logs:
            print(f"[*] Resuming: completed methods found: {', '.join(logs)}", flush=True)

    total_started = time.time()
    write_json_atomic(output_dir / "run_status.json", {
        "status": "running",
        "methods": methods,
        "completed_methods": list(logs),
        "current_method": next((name for name in methods if name not in logs), None),
        "seed": args.seed,
        "resume_from": str(args.resume_from) if args.resume_from else None,
    })
    for method in methods:
        if method in logs:
            print(f"[*] Skipping completed method {method}", flush=True)
            continue
        try:
            result = run_method(
                method, dataset, base_init_state,
                comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            # Write array artifacts first and the CSV row last. The CSV acts as
            # the commit marker, so resume never treats a half-written method as done.
            save_npy_atomic(
                output_dir / f"{method.replace('-', '_')}_R_rmse.npy",
                result["R_rmse"],
            )
            save_npy_atomic(
                output_dir / f"{method.replace('-', '_')}_R_mae.npy",
                result["R_mae"],
            )
            save_npy_atomic(
                output_dir / f"{method.replace('-', '_')}_R_rmse_clients.npy",
                result["R_rmse_clients"],
            )
            save_npy_atomic(
                output_dir / f"{method.replace('-', '_')}_R_mae_clients.npy",
                result["R_mae_clients"],
            )
            if result["S_matrix"] is not None:
                save_npy_atomic(
                    output_dir / f"{method.replace('-', '_')}_similarity.npy",
                    result["S_matrix"],
                )
            method_row = {key: result[key] for key in [
                "Method", "AE-RMSE (↓)", "AE-MAE (↓)", "BWT-RMSE (↓)", "BWT-MAE (↓)"
            ]}
            next_table_rows = table_rows + [method_row]
            write_dataframe_atomic(pd.DataFrame(next_table_rows), partial_path)
        except Exception as error:
            error_trace = traceback.format_exc()
            write_dataframe_atomic(
                pd.DataFrame(table_rows, columns=RESULT_COLUMNS), partial_path
            )
            write_json_atomic(output_dir / "run_status.json", {
                "status": "paused_after_error",
                "failed_method": method,
                "completed_methods": list(logs),
                "seed": args.seed,
                "error": repr(error),
                "traceback": error_trace,
            })
            with open(output_dir / "run_error.txt", "w", encoding="utf-8") as stream:
                stream.write(error_trace)
            print(
                f"[!] Paused at {method}; completed results were saved. "
                "Use --resume-from with this output to continue.",
                flush=True,
            )
            print(error_trace, flush=True)
            return
        logs[method] = result
        table_rows = next_table_rows
        write_json_atomic(output_dir / "run_status.json", {
            "status": "running",
            "methods": methods,
            "completed_methods": list(logs),
            "current_method": next((name for name in methods if name not in logs), None),
            "seed": args.seed,
            "resume_from": str(args.resume_from) if args.resume_from else None,
        })
        gc.collect()

    results = pd.DataFrame(table_rows, columns=RESULT_COLUMNS)
    write_dataframe_atomic(results, output_dir / "table1_main_results.csv")
    save_npy_atomic(
        output_dir / "R_rmse_by_method.npy",
        {name: log["R_rmse"] for name, log in logs.items()},
        allow_pickle=True,
    )
    print("\n" + results.to_string(index=False), flush=True)

    if logs:
        plt.figure(figsize=(8.4, 5.0))
        stages = ["After T1\n(Winter)", "After T2\n(Spring)", "After T3\n(Summer)", "After T4\n(Autumn)"]
        for name, log in logs.items():
            plt.plot(stages, log["R_rmse"][:, 0], marker="o",
                     linewidth=2.2 if name == "DeCo-Full" else 1.3,
                     markersize=5, label=name)
        plt.title("Catastrophic Forgetting Trajectory on Task 1 (Winter)", fontsize=11, fontweight="bold", pad=12)
        plt.ylabel("Test RMSE on Task 1 (Lower is Better)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "fig_forgetting_curve.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    client_frames = []
    for name, log in logs.items():
        rmse_by_client = log["R_rmse_clients"]
        mae_by_client = log["R_mae_clients"]
        ae_rmse_by_client = np.mean(rmse_by_client[-1], axis=0)
        ae_mae_by_client = np.mean(mae_by_client[-1], axis=0)
        bwt_rmse_by_client = np.mean(
            rmse_by_client[-1, :len(SEASONS) - 1, :]
            - rmse_by_client[np.arange(len(SEASONS) - 1), np.arange(len(SEASONS) - 1), :],
            axis=0,
        )
        bwt_mae_by_client = np.mean(
            mae_by_client[-1, :len(SEASONS) - 1, :]
            - mae_by_client[np.arange(len(SEASONS) - 1), np.arange(len(SEASONS) - 1), :],
            axis=0,
        )
        frame = selection[["LCLid", "acorn", "source_block"]].copy()
        frame["Method"] = name
        frame["Seed"] = args.seed
        frame["AE-RMSE"] = ae_rmse_by_client
        frame["AE-MAE"] = ae_mae_by_client
        frame["BWT-RMSE"] = bwt_rmse_by_client
        frame["BWT-MAE"] = bwt_mae_by_client
        client_frames.append(frame)
    if client_frames:
        write_dataframe_atomic(
            pd.concat(client_frames, ignore_index=True),
            output_dir / "per_client_metrics.csv",
        )

    similarity = logs.get("DeCo-Full", {}).get("S_matrix")
    if similarity is not None:
        sample_size = min(240, len(dataset))
        sample_indices = np.linspace(0, len(dataset) - 1, sample_size, dtype=int)
        sample = similarity[np.ix_(sample_indices, sample_indices)]
        plt.figure(figsize=(7.2, 6.2))
        sns.heatmap(sample, cmap="YlGnBu", vmin=0.0, vmax=1.0,
                    cbar_kws={"label": "Cosine Similarity $S_{ij}$"})
        plt.title(f"Similarity Matrix (sample of {sample_size} / {len(dataset)} clients)", fontsize=10.5, fontweight="bold", pad=12)
        plt.xlabel("Sampled household client index", fontsize=9.5)
        plt.ylabel("Sampled household client index", fontsize=9.5)
        plt.tight_layout()
        plt.savefig(output_dir / "fig_similarity_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        del similarity, sample

    config = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "blocks_found": len(glob.glob(str(Path(args.data_dir) / "halfhourly_dataset" / "block_*.csv"))),
        "eligible_clients": len(dataset),
        "clients_by_acorn": {
            str(group): int(count)
            for group, count in selection["acorn"].value_counts().items()
        },
        "max_clients_per_group": args.max_clients_per_group,
        "inject_synthetic_drift": args.inject_synthetic_drift,
        "comm_rounds": args.comm_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "experiment_version": "deco_ablation_v1",
        "methods": methods,
        "method_configs": {method: ABLATION_CONFIGS[method] for method in methods},
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "device": str(DEVICE),
        "elapsed_minutes": round((time.time() - total_started) / 60, 2),
    }
    write_json_atomic(output_dir / "run_config.json", config)
    error_path = output_dir / "run_error.txt"
    if error_path.exists():
        error_path.unlink()
    write_json_atomic(output_dir / "run_status.json", {
        "status": "complete",
        "methods": methods,
        "completed_methods": list(logs),
        "seed": args.seed,
        "elapsed_minutes": config["elapsed_minutes"],
    })
    print(f"[*] Finished in {config['elapsed_minutes']} min; results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
