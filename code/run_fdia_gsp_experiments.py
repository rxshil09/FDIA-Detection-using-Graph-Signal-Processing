import os
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import rankdata


SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------
# 1) Graph and system setup
# ---------------------------
N_BUSES = 30
M_MEAS = 60
MEAS_NOISE_STD = 0.05
LOWPASS_BETA = 2.6

# Keep trying until a connected graph is generated.
for _ in range(100):
    G = nx.random_geometric_graph(N_BUSES, radius=0.34, seed=int(rng.integers(1, 10_000)))
    if nx.is_connected(G):
        break
else:
    raise RuntimeError("Could not sample a connected graph.")

A = nx.to_numpy_array(G)
D = np.diag(A.sum(axis=1))
L = D - A

lam, U = np.linalg.eigh(L)
idx_sort = np.argsort(lam)
lam = lam[idx_sort]
U = U[:, idx_sort]

hf_mask = np.zeros(N_BUSES, dtype=bool)
hf_mask[int(0.65 * N_BUSES) :] = True

# Random linear measurement model.
H = rng.normal(0.0, 1.0, size=(M_MEAS, N_BUSES))
while np.linalg.matrix_rank(H) < N_BUSES:
    H = rng.normal(0.0, 1.0, size=(M_MEAS, N_BUSES))
H_pinv = np.linalg.pinv(H)


# ---------------------------------
# 2) Data generation and attack model
# ---------------------------------
def sample_smooth_state():
    z = rng.normal(size=N_BUSES)
    filt = np.exp(-LOWPASS_BETA * lam)
    x = U @ (filt * z)
    std = np.std(x)
    if std < 1e-12:
        std = 1.0
    return 0.40 * x / std


def sample_high_frequency_state_attack(strength=1.2):
    coeff = rng.normal(size=hf_mask.sum())
    c = U[:, hf_mask] @ coeff
    nrm = np.linalg.norm(c)
    if nrm < 1e-12:
        nrm = 1.0
    return strength * c / nrm


def random_measurement_attack(strength=2.6, frac=0.15):
    a = np.zeros(M_MEAS)
    k = max(1, int(frac * M_MEAS))
    idx = rng.choice(M_MEAS, size=k, replace=False)
    a[idx] = rng.normal(0.0, strength, size=k)
    return a


def make_sample(kind="normal"):
    x_true = sample_smooth_state()
    noise = rng.normal(0.0, MEAS_NOISE_STD, size=M_MEAS)
    y = H @ x_true + noise

    if kind == "random_attack":
        y = y + random_measurement_attack()
    elif kind == "stealth_attack":
        c = sample_high_frequency_state_attack(strength=1.2)
        y = y + H @ c

    x_hat = H_pinv @ y
    resid = np.linalg.norm(y - H @ x_hat) / np.sqrt(M_MEAS)

    x_spec = U.T @ x_hat
    total_energy = float(np.sum(x_spec ** 2)) + 1e-12
    hf_energy = float(np.sum((x_spec[hf_mask]) ** 2))
    gsp_hf_ratio = hf_energy / total_energy

    return {
        "x_hat": x_hat,
        "residual_score": float(resid),
        "gsp_score": float(gsp_hf_ratio),
    }


# ------------------------------
# 3) Build datasets
# ------------------------------
N_TRAIN_NORMAL = 1400
N_VAL_NORMAL = 400
N_TEST_NORMAL = 800
N_TEST_RANDOM = 500
N_TEST_STEALTH = 500


def collect_block(kind, n):
    rows = []
    for _ in range(n):
        s = make_sample(kind)
        rows.append(
            {
                "kind": kind,
                "residual_score": s["residual_score"],
                "gsp_score": s["gsp_score"],
                "x_hat": s["x_hat"],
            }
        )
    return rows


train_rows = collect_block("normal", N_TRAIN_NORMAL)
val_rows = collect_block("normal", N_VAL_NORMAL)
test_rows = []
test_rows.extend(collect_block("normal", N_TEST_NORMAL))
test_rows.extend(collect_block("random_attack", N_TEST_RANDOM))
test_rows.extend(collect_block("stealth_attack", N_TEST_STEALTH))


# ------------------------------
# 4) Baselines and GSP detector
# ------------------------------
train_x = np.array([r["x_hat"] for r in train_rows])
val_x = np.array([r["x_hat"] for r in val_rows])
test_x = np.array([r["x_hat"] for r in test_rows])

mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0) + 1e-6

def zscore_max(x):
    return float(np.max(np.abs((x - mu) / sigma)))

for r in val_rows:
    r["zscore_score"] = zscore_max(r["x_hat"])
for r in test_rows:
    r["zscore_score"] = zscore_max(r["x_hat"])

# High quantile threshold to enforce low false alarms.
q = 0.995
thr_residual = np.quantile([r["residual_score"] for r in val_rows], q)
thr_zscore = np.quantile([r["zscore_score"] for r in val_rows], q)
thr_gsp = np.quantile([r["gsp_score"] for r in val_rows], q)

for r in test_rows:
    r["is_attack"] = int(r["kind"] != "normal")
    r["pred_residual"] = int(r["residual_score"] > thr_residual)
    r["pred_zscore"] = int(r["zscore_score"] > thr_zscore)
    r["pred_gsp"] = int(r["gsp_score"] > thr_gsp)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "Accuracy": acc,
    }


def auc_score(y_true, score):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(score)
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def roc_curve_manual(y_true, score, n_points=200):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    lo = float(np.min(score))
    hi = float(np.max(score))
    thrs = np.linspace(lo, hi, n_points)
    tpr = []
    fpr = []
    for t in thrs:
        pred = (score > t).astype(int)
        m = metrics(y_true, pred)
        tpr.append(m["Recall"])
        fpr.append(m["FPR"])
    return np.array(fpr), np.array(tpr)


# ------------------------------
# 5) Compute metrics
# ------------------------------
test_df = pd.DataFrame(
    {
        "kind": [r["kind"] for r in test_rows],
        "is_attack": [r["is_attack"] for r in test_rows],
        "residual_score": [r["residual_score"] for r in test_rows],
        "zscore_score": [r["zscore_score"] for r in test_rows],
        "gsp_score": [r["gsp_score"] for r in test_rows],
        "pred_residual": [r["pred_residual"] for r in test_rows],
        "pred_zscore": [r["pred_zscore"] for r in test_rows],
        "pred_gsp": [r["pred_gsp"] for r in test_rows],
    }
)

method_pred_col = {
    "Residual_BDD": "pred_residual",
    "ZScore_Nodewise": "pred_zscore",
    "GSP_HighFreq": "pred_gsp",
}
method_score_col = {
    "Residual_BDD": "residual_score",
    "ZScore_Nodewise": "zscore_score",
    "GSP_HighFreq": "gsp_score",
}

summary_rows = []
for method, pcol in method_pred_col.items():
    m_all = metrics(test_df["is_attack"].values, test_df[pcol].values)
    auc = auc_score(test_df["is_attack"].values, test_df[method_score_col[method]].values)
    summary_rows.append(
        {
            "Method": method,
            "Scenario": "Overall",
            **m_all,
            "AUC": auc,
        }
    )

    # Random attack detectability: normal + random attack only
    sub_r = test_df[test_df["kind"].isin(["normal", "random_attack"])].copy()
    y_r = (sub_r["kind"] == "random_attack").astype(int).values
    m_r = metrics(y_r, sub_r[pcol].values)
    auc_r = auc_score(y_r, sub_r[method_score_col[method]].values)
    summary_rows.append(
        {
            "Method": method,
            "Scenario": "Random_Attack",
            **m_r,
            "AUC": auc_r,
        }
    )

    # Stealth attack detectability: normal + stealth attack only
    sub_s = test_df[test_df["kind"].isin(["normal", "stealth_attack"])].copy()
    y_s = (sub_s["kind"] == "stealth_attack").astype(int).values
    m_s = metrics(y_s, sub_s[pcol].values)
    auc_s = auc_score(y_s, sub_s[method_score_col[method]].values)
    summary_rows.append(
        {
            "Method": method,
            "Scenario": "Stealth_Attack",
            **m_s,
            "AUC": auc_s,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)

# Store thresholds and residual means to support theory claims.
residual_means = (
    test_df.groupby("kind")["residual_score"].mean().to_dict()
)
aux = {
    "thresholds": {
        "Residual_BDD": float(thr_residual),
        "ZScore_Nodewise": float(thr_zscore),
        "GSP_HighFreq": float(thr_gsp),
    },
    "mean_residual_by_kind": {k: float(v) for k, v in residual_means.items()},
}
with open(os.path.join(RESULTS_DIR, "aux_stats.json"), "w", encoding="utf-8") as f:
    json.dump(aux, f, indent=2)


# ------------------------------
# 6) Plots for report
# ------------------------------
pos = nx.spring_layout(G, seed=SEED)

# Pick one normal and one stealth sample for node-level visualization.
x_normal = train_rows[0]["x_hat"]
x_stealth = None
for r in test_rows:
    if r["kind"] == "stealth_attack":
        x_stealth = r["x_hat"]
        break

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
nx.draw_networkx(
    G,
    pos=pos,
    node_color=x_normal,
    cmap="viridis",
    node_size=260,
    with_labels=False,
    edge_color="#999999",
)
plt.title("Normal Estimated Graph Signal")

plt.subplot(1, 2, 2)
nx.draw_networkx(
    G,
    pos=pos,
    node_color=x_stealth,
    cmap="viridis",
    node_size=260,
    with_labels=False,
    edge_color="#999999",
)
plt.title("Stealth Attack Estimated Graph Signal")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "graph_signal_visualization.png"), dpi=220)
plt.close()

# Score distributions.
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, score_col, title in [
    (axes[0], "residual_score", "Residual BDD Score"),
    (axes[1], "zscore_score", "Nodewise Z-Score"),
    (axes[2], "gsp_score", "GSP High-Frequency Energy Ratio"),
]:
    for kind, color in [
        ("normal", "#1f77b4"),
        ("random_attack", "#d62728"),
        ("stealth_attack", "#2ca02c"),
    ]:
        vals = test_df[test_df["kind"] == kind][score_col].values
        ax.hist(vals, bins=35, alpha=0.45, density=True, color=color, label=kind)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "score_distributions.png"), dpi=220)
plt.close()

# ROC curves.
plt.figure(figsize=(6.8, 5.8))
for method in ["Residual_BDD", "ZScore_Nodewise", "GSP_HighFreq"]:
    fpr, tpr = roc_curve_manual(test_df["is_attack"].values, test_df[method_score_col[method]].values)
    auc = auc_score(test_df["is_attack"].values, test_df[method_score_col[method]].values)
    plt.plot(fpr, tpr, lw=2, label=f"{method} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison: Attack Detection")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_comparison.png"), dpi=220)
plt.close()

# Bar chart: recall by scenario.
pivot = summary_df.pivot(index="Method", columns="Scenario", values="Recall")
pivot = pivot[["Random_Attack", "Stealth_Attack", "Overall"]]
ax = pivot.plot(kind="bar", figsize=(8.5, 5.2), rot=0)
ax.set_ylabel("Recall")
ax.set_title("Recall by Attack Scenario")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "recall_by_scenario.png"), dpi=220)
plt.close()


# ------------------------------
# 7) Save test-level scores and run log
# ------------------------------
test_df.to_csv(os.path.join(RESULTS_DIR, "test_scores.csv"), index=False)

run_info = {
    "seed": SEED,
    "n_buses": N_BUSES,
    "m_measurements": M_MEAS,
    "noise_std": MEAS_NOISE_STD,
    "dataset_sizes": {
        "train_normal": N_TRAIN_NORMAL,
        "val_normal": N_VAL_NORMAL,
        "test_normal": N_TEST_NORMAL,
        "test_random_attack": N_TEST_RANDOM,
        "test_stealth_attack": N_TEST_STEALTH,
    },
}
with open(os.path.join(RESULTS_DIR, "run_info.json"), "w", encoding="utf-8") as f:
    json.dump(run_info, f, indent=2)

print("Experiment completed.")
print("Saved:")
print(" - results/metrics_summary.csv")
print(" - results/test_scores.csv")
print(" - results/aux_stats.json")
print(" - results/figures/*.png")
