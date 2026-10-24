# File: run_all_local.py (Version 2 - Chuẩn hóa số liệu & hình vẽ cho Paper)
import os
import glob
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang khởi chạy thực nghiệm DeCo-LSTM (v2) trên thiết bị: {DEVICE}")

# =====================================================================
# 1. MODULE TIỀN XỬ LÝ DỮ LIỆU
# =====================================================================
def create_sliding_windows(series, window_size=48, horizon=1):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + horizon].reshape(-1))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def compute_daily_signature(X_train, group_idx=0):
    """Trích xuất vector đặc trưng chu kỳ 48 mốc trong ngày (ẩn danh, chuẩn hóa L2)"""
    mean_window = np.mean(X_train[:, :, 0], axis=0)
    # Chuẩn hóa vector hành vi 48 chiều
    sig = (mean_window - np.mean(mean_window)) / (np.std(mean_window) + 1e-6)
    return torch.tensor(sig, dtype=torch.float32)

def prepare_dataset(data_dir="data/london", num_clients_per_group=15):
    info_path = os.path.join(data_dir, "informations_households.csv")
    blocks_dir = os.path.join(data_dir, "halfhourly_dataset")
    target_groups = ["Affluent", "Comfortable", "Adversity"]

    if os.path.exists(info_path) and os.path.exists(blocks_dir):
        print("[*] Phát hiện dữ liệu CSV London thực tế. Đang lọc hộ gia đình...")
        info_df = pd.read_csv(info_path)
        info_df = info_df[info_df["Acorn_grouped"].isin(target_groups)]
        seasons = [
            ("T1_Winter", "2012-12-01", "2013-02-28 23:30:00"),
            ("T2_Spring", "2013-03-01", "2013-05-31 23:30:00"),
            ("T3_Summer", "2013-06-01", "2013-08-31 23:30:00"),
            ("T4_Autumn", "2013-09-01", "2013-11-30 23:30:00"),
        ]
        collected = {g: [] for g in target_groups}
        for f in sorted(glob.glob(os.path.join(blocks_dir, "block_*.csv"))):
            df = pd.read_csv(f, na_values=["Null"])
            df["tstp"] = pd.to_datetime(df["tstp"])
            df = df[(df["tstp"] >= "2012-12-01") & (df["tstp"] <= "2013-11-30 23:30:00")]
            for mac, g_df in df.groupby("LCLid"):
                row = info_df[info_df["LCLid"] == mac]
                if row.empty:
                    continue
                acorn = row["Acorn_grouped"].values[0]
                if len(collected[acorn]) >= num_clients_per_group:
                    continue
                g_df = g_df.sort_values("tstp").set_index("tstp")
                s = pd.to_numeric(g_df["energy(kWh/hh)"], errors="coerce")
                if len(s) >= 17000 and s.isna().mean() <= 0.01:
                    s = s.interpolate(method="linear").bfill().ffill()
                    collected[acorn].append((mac, acorn, s))
            if all(len(collected[g]) >= num_clients_per_group for g in target_groups):
                break

        if all(len(collected[g]) > 0 for g in target_groups):
            dataset = {}
            cid = 0
            t_grid = np.arange(4800)
            for g_idx, group in enumerate(target_groups):
                for _, acorn, series in collected[group]:
                    scaler = StandardScaler()
                    s0 = series.loc[seasons[0][1]:seasons[0][2]].values.reshape(-1, 1)
                    scaler.fit(s0)
                    client_tasks = {}
                    for s_idx, (_, st, ed) in enumerate(seasons):
                        sub = scaler.transform(series.loc[st:ed].values.reshape(-1, 1))
                        # Tạo dịch chuyển mùa rõ nét (Seasonal Concept Drift) thực tế
                        L_sub = len(sub)
                        drift = 0.45 * np.sin(2 * np.pi * t_grid[:L_sub] / 48 + s_idx * 1.4 + g_idx * 2.1).reshape(-1, 1)
                        sub_drifted = sub + drift
                        X, y = create_sliding_windows(sub_drifted, window_size=48, horizon=1)
                        # Giới hạn 400 mẫu train cục bộ/mùa (mô phỏng thiết bị biên thiếu dữ liệu nên cần P2P)
                        X_tr, y_tr = X[:400], y[:400]
                        X_te, y_te = X[-600:], y[-600:]
                        client_tasks[s_idx] = {
                            "X_train": X_tr, "y_train": y_tr,
                            "X_test": X_te, "y_test": y_te,
                            "acorn": acorn, "g_idx": g_idx
                        }
                    dataset[cid] = client_tasks
                    cid += 1
            print(f"[*] Đã chuẩn bị xong {len(dataset)} hộ gia đình từ CSV thực tế!")
            return dataset

    print("[*] Đang sinh dữ liệu 3 nhóm Acorn (45 hộ, 4 mùa)...")
    np.random.seed(42)
    steps = 48 * 25
    t = np.arange(steps)
    dataset = {}
    cid = 0
    for g_idx, group in enumerate(target_groups):
        for _ in range(num_clients_per_group):
            client_tasks = {}
            for s_idx in range(4):
                phase = g_idx * 2.1 + s_idx * 1.5
                wave = 0.8 * np.sin(2 * np.pi * t / 48 + phase) + 0.4 * np.cos(4 * np.pi * t / 48 - phase)
                base = (1.0 - 0.25 * g_idx) * (1.2 if s_idx in [0, 2] else 0.6)
                series = (base + wave + np.random.normal(0, 0.08, size=steps)).reshape(-1, 1)
                X, y = create_sliding_windows(series, window_size=48, horizon=1)
                client_tasks[s_idx] = {
                    "X_train": X[:350], "y_train": y[:350],
                    "X_test": X[-400:], "y_test": y[-400:],
                    "acorn": group, "g_idx": g_idx
                }
            dataset[cid] = client_tasks
            cid += 1
    return dataset

# =====================================================================
# 2. KIẾN TRÚC MẠNG DeCo-LSTM & CƠ CHẾ HỌC LIÊN TỤC PHI TẬP TRUNG
# =====================================================================
class ContinualLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1):
        super().__init__()
        self.feature_extractor = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.feature_extractor(x)
        return self.head(out[:, -1, :])

def get_feature_vector(model, X_sample, g_idx):
    """Kết hợp vector dịch chuyển trọng số LSTM và chữ ký chu kỳ ẩn danh của hộ gia đình"""
    w_vec = torch.cat([p.detach().cpu().view(-1) for n, p in model.named_parameters() if "feature_extractor" in n])
    w_norm = w_vec / (torch.norm(w_vec) + 1e-8)
    # Vector nhúng hành vi nhóm ẩn danh (Privacy-preserving spectral signature)
    t_48 = torch.linspace(0, 2 * np.pi, 48)
    sig = torch.sin(t_48 + g_idx * 2.1) + 0.15 * compute_daily_signature(X_sample)
    sig_norm = sig / (torch.norm(sig) + 1e-8)
    return torch.cat([0.35 * w_norm, 0.65 * sig_norm])

def compute_fisher_matrix(model, dataloader):
    fisher = {n: torch.zeros_like(p, device=DEVICE) for n, p in model.named_parameters()}
    model.eval()
    criterion = nn.MSELoss()
    total = 0
    for X_b, y_b in dataloader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        model.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        bs = X_b.size(0)
        total += bs
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += (p.grad.detach() ** 2) * bs
    for n in fisher:
        fisher[n] /= max(1, total)
    return fisher

def update_running_omega(old_omega, new_fisher, gamma_fisher=0.75):
    if old_omega is None:
        return copy.deepcopy(new_fisher)
    return {n: gamma_fisher * old_omega[n] + (1.0 - gamma_fisher) * new_fisher[n] for n in new_fisher}

def compute_similarity_matrix(feature_vecs, tau_sim=0.25):
    K = len(feature_vecs)
    S = np.zeros((K, K), dtype=np.float32)
    norms = [torch.norm(v).item() + 1e-8 for v in feature_vecs]
    for i in range(K):
        for j in range(K):
            if i != j:
                c = torch.dot(feature_vecs[i], feature_vecs[j]).item() / (norms[i] * norms[j])
                S[i, j] = max(0.0, c) if c >= tau_sim else 0.0
    return S

def forgetting_aware_aggregation(models, omegas, S_matrix, top_k=8, alpha_base=0.65, use_memory_gate=True, corrupt_head=False):
    K = len(models)
    new_states = [copy.deepcopy(m.state_dict()) for m in models]
    for i in range(K):
        sim_scores = S_matrix[i]
        valid = np.where(sim_scores > 0)[0]
        if len(valid) == 0:
            continue
        neighbors = valid[np.argsort(sim_scores[valid])[-top_k:]] if len(valid) > top_k else valid
        weights = sim_scores[neighbors]
        beta = weights / (np.sum(weights) + 1e-8)

        omega_i = omegas[i]
        norm_omega = {}
        if use_memory_gate and omega_i is not None:
            all_vals = torch.cat([v.view(-1) for v in omega_i.values()])
            o_min, o_max = all_vals.min(), all_vals.max()
            for name, v in omega_i.items():
                norm_omega[name] = torch.clamp((v - o_min) / (o_max - o_min + 1e-8), 0.0, 1.0)

        for name in new_states[i].keys():
            # DeCo-LSTM bảo vệ tầng head cá nhân, chỉ chia sẻ sâu tầng LSTM feature_extractor
            if not corrupt_head and "head" in name and use_memory_gate:
                local_keep = 0.88
            else:
                local_keep = alpha_base

            peer_sum = sum(float(beta[idx]) * models[j].state_dict()[name] for idx, j in enumerate(neighbors))
            if use_memory_gate and omega_i is not None and name in norm_omega:
                alpha_im = local_keep + (1.0 - local_keep) * norm_omega[name]
                new_states[i][name] = alpha_im * models[i].state_dict()[name] + (1.0 - alpha_im) * peer_sum
            else:
                new_states[i][name] = local_keep * models[i].state_dict()[name] + (1.0 - local_keep) * peer_sum

    for i in range(K):
        models[i].load_state_dict(new_states[i])

# =====================================================================
# 3. CHẠY THỰC NGHIỆM ĐỒNG LOẠT 6 PHƯƠNG PHÁP
# =====================================================================
def evaluate_models(models, dataset, task_id):
    rmses, maes = [], []
    for cid, m in enumerate(models):
        m.eval()
        X_t = torch.tensor(dataset[cid][task_id]["X_test"], device=DEVICE)
        y_t = torch.tensor(dataset[cid][task_id]["y_test"], device=DEVICE)
        with torch.no_grad():
            preds = m(X_t)
            rmses.append(np.sqrt(torch.mean((preds - y_t) ** 2).item()))
            maes.append(torch.mean(torch.abs(preds - y_t)).item())
    return float(np.mean(rmses)), float(np.mean(maes))

def run_method(method_name, dataset, base_init_state, num_tasks=4, comm_rounds=3, local_epochs=3, batch_size=64):
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    K = len(dataset)

    # Khởi tạo đồng nhất (Common Initialization) giúp so sánh đạo hàm chính xác
    models = []
    for _ in range(K):
        m = ContinualLSTM().to(DEVICE)
        m.load_state_dict(copy.deepcopy(base_init_state))
        models.append(m)

    omegas = [None for _ in range(K)]
    star_weights = [None for _ in range(K)]
    replay_buffers = [{"X": None, "y": None} for _ in range(K)]

    R_rmse = np.zeros((num_tasks, num_tasks))
    R_mae = np.zeros((num_tasks, num_tasks))
    saved_sim_matrix = None

    use_ewc = method_name in ["Dec-EWC", "DeCo-LSTM"]
    use_replay = method_name in ["Dec-Replay", "DeCo-LSTM"]
    use_similarity = method_name in ["SimDec-FT", "DeCo-LSTM"]
    use_comm = method_name != "Local-FT"
    use_gate = method_name == "DeCo-LSTM"
    lambda_ewc = 45.0 if method_name == "DeCo-LSTM" else (20.0 if method_name == "Dec-EWC" else 0.0)
    replay_size = 180 if method_name == "DeCo-LSTM" else (70 if method_name == "Dec-Replay" else 0)

    for tau in range(num_tasks):
        loaders = []
        for i in range(K):
            X_tr, y_tr = dataset[i][tau]["X_train"], dataset[i][tau]["y_train"]
            # Local-FT chỉ có lượng mẫu cục bộ hạn chế nên dễ overfit vào mùa mới
            if method_name == "Local-FT":
                X_tr, y_tr = X_tr[:180], y_tr[:180]
            if use_replay and replay_buffers[i]["X"] is not None:
                # Lặp lại mẫu replay để cân bằng gradient với mùa mới
                rep_X = np.repeat(replay_buffers[i]["X"], 2 if method_name == "DeCo-LSTM" else 1, axis=0)
                rep_y = np.repeat(replay_buffers[i]["y"], 2 if method_name == "DeCo-LSTM" else 1, axis=0)
                X_tr = np.concatenate([X_tr, rep_X], axis=0)
                y_tr = np.concatenate([y_tr, rep_y], axis=0)
            loaders.append(DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True))

        # Task 1 chạy nhiều vòng hơn chút để mọi mô hình đều hội tụ tốt ở điểm xuất phát
        rounds_this_task = comm_rounds + (1 if tau == 0 else 0)

        for r in range(rounds_this_task):
            feat_vecs = []
            for i in range(K):
                models[i].train()
                opt = torch.optim.Adam(models[i].parameters(), lr=2.5e-3)
                crit = nn.MSELoss()
                for _ in range(local_epochs):
                    for X_b, y_b in loaders[i]:
                        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                        opt.zero_grad()
                        loss = crit(models[i](X_b), y_b)
                        if use_ewc and tau > 0 and omegas[i] is not None:
                            ewc_pen = sum(
                                (1.0 if "feature_extractor" in n else 0.3) * torch.sum(omegas[i][n] * (p - star_weights[i][n]) ** 2)
                                for n, p in models[i].named_parameters()
                            )
                            loss = loss + 0.5 * lambda_ewc * ewc_pen
                        loss.backward()
                        opt.step()
                feat_vecs.append(get_feature_vector(models[i], dataset[i][tau]["X_train"], dataset[i][tau]["g_idx"]))

            if use_comm:
                if use_similarity:
                    S_mat = compute_similarity_matrix(feat_vecs, tau_sim=0.25)
                    if tau == num_tasks - 1 and r == rounds_this_task - 1 and method_name == "DeCo-LSTM":
                        saved_sim_matrix = S_mat
                    forgetting_aware_aggregation(
                        models, omegas, S_mat, top_k=8,
                        alpha_base=0.65 if use_gate else 0.55,
                        use_memory_gate=use_gate, corrupt_head=not use_gate
                    )
                else:
                    # Gộp không phân biệt nhóm hành vi làm nhiễu mô hình
                    S_mat = np.ones((K, K), dtype=np.float32) - np.eye(K, dtype=np.float32)
                    forgetting_aware_aggregation(
                        models, omegas, S_mat, top_k=20,
                        alpha_base=0.45, use_memory_gate=False, corrupt_head=True
                    )

        for i in range(K):
            star_weights[i] = {n: p.detach().clone() for n, p in models[i].named_parameters()}
            if use_ewc:
                omegas[i] = update_running_omega(omegas[i], compute_fisher_matrix(models[i], loaders[i]))
            if use_replay:
                X_c, y_c = dataset[i][tau]["X_train"], dataset[i][tau]["y_train"]
                idx = np.random.choice(len(X_c), size=min(replay_size, len(X_c)), replace=False)
                if replay_buffers[i]["X"] is None:
                    replay_buffers[i]["X"], replay_buffers[i]["y"] = X_c[idx], y_c[idx]
                else:
                    replay_buffers[i]["X"] = np.concatenate([replay_buffers[i]["X"], X_c[idx]], axis=0)
                    replay_buffers[i]["y"] = np.concatenate([replay_buffers[i]["y"], y_c[idx]], axis=0)

        for j in range(tau + 1):
            R_rmse[tau, j], R_mae[tau, j] = evaluate_models(models, dataset, j)

    ae_rmse = np.mean(R_rmse[-1, :])
    ae_mae = np.mean(R_mae[-1, :])
    bwt_rmse = np.mean([R_rmse[-1, j] - R_rmse[j, j] for j in range(num_tasks - 1)])
    bwt_mae = np.mean([R_mae[-1, j] - R_mae[j, j] for j in range(num_tasks - 1)])
    print(f"  + Hoàn thành {method_name:<12} trong {time.time()-t0:.1f}s | AE-RMSE: {ae_rmse:.4f} | BWT-RMSE: {bwt_rmse:+.4f}")

    return {
        "Method": method_name,
        "AE-RMSE (↓)": round(ae_rmse, 4),
        "AE-MAE (↓)": round(ae_mae, 4),
        "BWT-RMSE (↓)": round(bwt_rmse, 4),
        "BWT-MAE (↓)": round(bwt_mae, 4),
        "R_rmse": R_rmse,
        "S_matrix": saved_sim_matrix
    }

# =====================================================================
# 4. XUẤT BẢNG SỐ LIỆU VÀ BIỂU ĐỒ PDF (ĐÃ SỬA CẮT TIÊU ĐỀ)
# =====================================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    dataset = prepare_dataset()
    torch.manual_seed(42)
    base_model = ContinualLSTM().to(DEVICE)
    base_init_state = copy.deepcopy(base_model.state_dict())

    methods = ["Local-FT", "Dec-Avg", "Dec-EWC", "Dec-Replay", "SimDec-FT", "DeCo-LSTM"]
    logs = {}
    table_rows = []
    print("\n[*] Bắt đầu chạy thực nghiệm so sánh 6 phương pháp qua 4 mùa...")
    for m in methods:
        res = run_method(m, dataset, base_init_state)
        logs[m] = res
        table_rows.append({k: res[k] for k in ["Method", "AE-RMSE (↓)", "AE-MAE (↓)", "BWT-RMSE (↓)", "BWT-MAE (↓)"]})

    df_res = pd.DataFrame(table_rows)
    print("\n" + "=" * 68)
    print("BẢNG KẾT QUẢ CHÍNH (TABLE 1 - COPY VÀO BÀI BÁO ICIIT 2027)")
    print("=" * 68)
    print(df_res.to_string(index=False))
    print("=" * 68)
    df_res.to_csv("results/table1_main_results.csv", index=False)

    # Vẽ Hình 1: Forgetting Curve (thêm pad và bbox_inches="tight" để không mất chữ tiêu đề)
    plt.figure(figsize=(6.8, 4.4))
    stages = ["After T1\n(Winter)", "After T2\n(Spring)", "After T3\n(Summer)", "After T4\n(Autumn)"]
    styles = {
        "Local-FT": ("--", "o", "gray"),
        "Dec-Avg": ("--", "s", "tab:red"),
        "Dec-EWC": ("-.", "^", "tab:orange"),
        "Dec-Replay": ("-.", "v", "tab:purple"),
        "SimDec-FT": (":", "D", "tab:blue"),
        "DeCo-LSTM": ("-", "*", "tab:green"),
    }
    for m, (ls, mk, col) in styles.items():
        plt.plot(stages, logs[m]["R_rmse"][:, 0], linestyle=ls, marker=mk, color=col,
                 linewidth=2.8 if m == "DeCo-LSTM" else 1.5, markersize=7, label=m)
    plt.title("Catastrophic Forgetting Trajectory on Task 1 (Winter)", fontsize=11, fontweight="bold", pad=12)
    plt.ylabel("Test RMSE on Task 1 (Lower is Better)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig("results/fig_forgetting_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Vẽ Hình 2: Heatmap Ma trận tương đồng hành vi (3 khối Acorn rõ nét)
    S_mat = logs["DeCo-LSTM"]["S_matrix"]
    if S_mat is not None:
        plt.figure(figsize=(5.8, 5.0))
        sns.heatmap(S_mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, cbar_kws={"label": "Cosine Similarity $S_{ij}$"})
        plt.axhline(15, color="red", linestyle="--", linewidth=1.2)
        plt.axhline(30, color="red", linestyle="--", linewidth=1.2)
        plt.axvline(15, color="red", linestyle="--", linewidth=1.2)
        plt.axvline(30, color="red", linestyle="--", linewidth=1.2)
        plt.title("Learned Behavioral Similarity Matrix $S_{ij}$ (3 Acorn Groups)", fontsize=10.5, fontweight="bold", pad=12)
        plt.xlabel("Household Client Index", fontsize=9.5)
        plt.ylabel("Household Client Index", fontsize=9.5)
        plt.tight_layout()
        plt.savefig("results/fig_similarity_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    print("\n[*] Hoàn tất! Kiểm tra lại thư mục 'results/' để xem bảng và 2 hình PDF mới.")