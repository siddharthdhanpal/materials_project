#!/usr/bin/env python3
# sample_size_eval.py
import os, json, math, random, shutil, datetime
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Paths & constants (edit these if needed)
# -----------------------------
EXP_012_DIR = Path("/home/ubuntu/projects/densitySpec/density_Spec_new_graphs/exp_012")  # where exp_012 lives
OUT_ROOT    = Path("/home/ubuntu/projects/densitySpec/density_Spec_new_graphs/exp_012_sample_size")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATA_CUBE      = "/home/ubuntu/datasets/new_graphs/stacked_cubes.npy"
DATA_CUBE_ELF  = "/home/ubuntu/datasets/new_graphs/stacked_cubes_elf.npy"
DATA_SPEC      = "/home/ubuntu/datasets/new_graphs/tddft_allspec.npy"

TEST_SIZE = 48               # keep the last 48 samples for test (same as your 740/train setup)
TRAIN_SIZES = [360, 180]     # requested experiments
BATCH_SIZE = 16
EPOCHS = 200                 # matches your summary runs
LR = 5e-5
WEIGHT_DECAY = 1e-6
LOSS_TYPE = "l1"             # as used in your best runs
SEED = 0                     # same shuffle seed as before
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset & Losses (same as your training)
# -----------------------------
class CubeSpectrumDataset(Dataset):
    def __init__(self, cube_data, spec_data):
        self.cube_data = torch.tensor(cube_data, dtype=torch.float32)
        self.spec_data = torch.tensor(spec_data, dtype=torch.float32)
    def __len__(self): return len(self.cube_data)
    def __getitem__(self, idx): return self.cube_data[idx], self.spec_data[idx]

class CorrelationLoss(nn.Module):
    def forward(self, y_pred, y_true):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - cost

LOSS_MAP = {
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
}
# -----------------------------
# Load exp_012 model definition or fallback to local definition
# -----------------------------
def import_model_from_exp_dir(exp_dir: Path):
    cfg = None
    cfg_path = exp_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    # try to import model_def.py (created by the sweep script)
    model_mod = None
    model_path = exp_dir / "model_def.py"
    if model_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_def", model_path)
        model_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_mod)

    # if both exist, great; otherwise, build a local equivalent of exp_012
    if model_mod is not None and cfg is not None:
        def build(input_shape, out_features):
            m = model_mod.build_model(**cfg)
            # patch IO sizes if needed
            if hasattr(m, "input_shape"):
                m.input_shape = input_shape
            # Recreate linear with correct out_features when re-built
            # (exp_012 used out_features=2000; we set to actual spec length)
            if hasattr(m, "linear") and hasattr(m, "flatten"):
                with torch.no_grad():
                    x = torch.rand(1, *input_shape)
                    y = m._forward_features(x)
                    lin_in = m.flatten(y).shape[1]
                m.linear = nn.Linear(lin_in, out_features)
            return m
        return build, cfg
    else:
        # Fallback: parameterised model with exp_012 settings (7 layers, k=5, base=7, dropout=0.1)
        class CNNModel(nn.Module):
            def __init__(self, input_shape, out_features, num_layers=7, kernel_size=5, base_filters=7, dropout_p=0.1):
                super().__init__()
                self.in_ch = 2
                self.input_shape = input_shape
                self.out_features = out_features
                self.num_layers = num_layers
                self.kernel_size = kernel_size
                self.base_filters = base_filters
                self.dropout_p = float(dropout_p)

                self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
                self.avgpool = nn.AvgPool3d(kernel_size=2, padding=1)

                blocks = []
                ks = kernel_size
                pad_same = int((ks - 1) / 2)

                def channel_mult(i):
                    if i == 1:  return 1
                    if i == 2:  return 2
                    if i == 3:  return 4
                    if i == 4:  return 8
                    if i == 5:  return 16
                    return 32

                in_c = self.in_ch
                for i in range(1, num_layers + 1):
                    out_c = self.base_filters * channel_mult(i)
                    out_c = min(out_c, self.base_filters * 32)
                    conv = nn.Conv3d(in_c, out_c, kernel_size=ks, padding=pad_same, padding_mode='zeros')
                    gn = nn.GroupNorm(out_c, out_c)
                    block = nn.Sequential(conv, gn, nn.ReLU(inplace=True))
                    blocks.append(block)
                    in_c = out_c

                self.blocks = nn.ModuleList(blocks)
                self.drop_mid = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
                self.drop_end = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
                self.drop_fc = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
                self.flatten = nn.Flatten()

                with torch.no_grad():
                    x = torch.rand(1, *self.input_shape)
                    mid_index = max(1, math.floor(self.num_layers * 0.6))
                    y = x
                    for j, block in enumerate(self.blocks, start=1):
                        y = block(y)
                        y = self.avgpool(self.maxpool(y))
                        if j == mid_index: y = self.drop_mid(y)
                    y = self.drop_end(y)
                    lin_in = self.flatten(y).shape[1]
                self.linear = nn.Linear(lin_in, self.out_features)
                self.softmax = nn.Softmax(dim=1)

            def _forward_features(self, x):
                mid_index = max(1, math.floor(self.num_layers * 0.6))
                for j, block in enumerate(self.blocks, start=1):
                    x = block(x)
                    x = self.avgpool(self.maxpool(x))
                    if j == mid_index: x = self.drop_mid(x)
                x = self.drop_end(x)
                return x

            def forward(self, x):
                x = self._forward_features(x)
                x = self.flatten(x)
                x = self.drop_fc(x)
                x = self.linear(x)
                x = self.softmax(x)
                return x

        def build(input_shape, out_features):
            return CNNModel(input_shape=input_shape, out_features=out_features)
        fallback_cfg = {"num_layers":7,"kernel_size":5,"base_filters":7,"dropout_p":0.1}
        return build, fallback_cfg

# -----------------------------
# Train / Eval
# -----------------------------
def train_and_evaluate(exp_name, train_idx, val_idx, test_idx, cubes, specs):
    out_dir = OUT_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # datasets/loaders
    train_ds = CubeSpectrumDataset(cubes[train_idx], specs[train_idx])
    val_ds   = CubeSpectrumDataset(cubes[val_idx], specs[val_idx])
    test_ds  = CubeSpectrumDataset(cubes[test_idx], specs[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # model
    input_shape = (cubes.shape[1], cubes.shape[2], cubes.shape[3], cubes.shape[4])  # (2, D, H, W)
    out_features = specs.shape[1]
    build_model, used_cfg = import_model_from_exp_dir(EXP_012_DIR)
    model = build_model(input_shape, out_features).to(DEVICE)

    # losses/optim
    criterion = LOSS_MAP[LOSS_TYPE]
    corr_loss = CorrelationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_path = out_dir / "best_model.pth"

    # log file
    (out_dir / "train_log.csv").write_text("epoch,train_loss,val_loss,train_corr,val_corr\n")

    for epoch in range(EPOCHS):
        # train
        model.train()
        tr_loss_sum = 0.0
        tr_corr_sum = 0.0
        tr_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * xb.size(0)
            tr_corr_sum += (1.0 - corr_loss(pred, yb).item()) * yb.size(0)
            tr_count += yb.size(0)
        tr_loss = tr_loss_sum / len(train_loader.dataset)
        tr_corr = tr_corr_sum / max(1, tr_count)

        # val
        model.eval()
        va_loss_sum = 0.0
        va_corr_sum = 0.0
        va_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                va_loss_sum += loss.item() * xb.size(0)
                va_corr_sum += (1.0 - corr_loss(pred, yb).item()) * yb.size(0)
                va_count += yb.size(0)
        va_loss = va_loss_sum / len(val_loader.dataset)
        va_corr = va_corr_sum / max(1, va_count)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"[{exp_name}] ✓ epoch {epoch+1} best val loss {va_loss:.6f}")

        with (out_dir / "train_log.csv").open("a") as f:
            f.write(f"{epoch+1},{tr_loss:.6f},{va_loss:.6f},{tr_corr:.6f},{va_corr:.6f}\n")

        print(f"[{exp_name}] {epoch+1}/{EPOCHS}  train {tr_loss:.6f}  val {va_loss:.6f}  trCorr {tr_corr:.4f}  vaCorr {va_corr:.4f}")

    # load best and evaluate on TEST (48 samples)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    test_l1 = 0.0
    test_corr_sum = 0.0
    n = 0
    mae_sum = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            l1 = LOSS_MAP["l1"](pred, yb).item()
            test_l1 += l1 * xb.size(0)
            # correlation (same definition as training)
            test_corr_sum += (1.0 - CorrelationLoss()(pred, yb).item()) * yb.size(0)
            # MAE explicitly
            mae_sum += torch.mean(torch.abs(pred - yb)).item() * xb.size(0)
            n += xb.size(0)

    test_loss = test_l1 / n
    test_corr = test_corr_sum / n
    test_mae  = mae_sum / n

    # save summary row
    row = {
        "exp_name": exp_name,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss_type": LOSS_TYPE,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_corr": test_corr,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    return row

# -----------------------------
# Data loading & split (same shuffle; test = last 48)
# -----------------------------
def load_and_prepare_data():
    cube_data = np.load(DATA_CUBE)
    cube_data_2 = np.load(DATA_CUBE_ELF)
    spec_data = np.load(DATA_SPEC)

    cube_data = np.concatenate((cube_data, cube_data_2), axis=1)   # concat channels
    spec_data = gaussian_filter1d(spec_data, sigma=10)             # same smoothing
    cube_data = cube_data / np.max(cube_data)                      # normalize

    # same shuffle as before
    np.random.seed(SEED)
    idx = np.arange(len(cube_data))
    np.random.shuffle(idx)
    cube_data = cube_data[idx]
    spec_data = spec_data[idx]
    return cube_data, spec_data

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load data
    cubes, specs = load_and_prepare_data()
    N = len(cubes)
    assert N >= TEST_SIZE + max(TRAIN_SIZES), f"Dataset too small for requested splits. N={N}"

    # fixed test set (same 48 as before: last 48 after the same shuffle)
    test_idx = np.arange(N - TEST_SIZE, N)

    # results CSV
    results_csv = OUT_ROOT / "sample_size_results.csv"
    if not results_csv.exists():
        results_csv.write_text("exp_name,train_size,val_size,test_size,epochs,batch_size,lr,weight_decay,loss_type,best_val_loss,test_loss,test_mae,test_corr,timestamp\n")

    # run both experiments
    for tr_size in TRAIN_SIZES:
        exp_name = f"train_{tr_size}_test_48"
        # validation is everything between train end and test start
        val_start = tr_size
        val_end = N - TEST_SIZE
        assert val_end > val_start, "No room left for validation before the fixed test set."
        val_idx = np.arange(val_start, val_end)
        train_idx = np.arange(0, tr_size)

        row = train_and_evaluate(exp_name, train_idx, val_idx, test_idx, cubes, specs)

        with results_csv.open("a") as f:
            f.write(",".join([
                row["exp_name"],
                str(row["train_size"]),
                str(row["val_size"]),
                str(row["test_size"]),
                str(row["epochs"]),
                str(row["batch_size"]),
                str(row["lr"]),
                str(row["weight_decay"]),
                row["loss_type"],
                f'{row["best_val_loss"]:.6f}',
                f'{row["test_loss"]:.6f}',
                f'{row["test_mae"]:.6f}',
                f'{row["test_corr"]:.6f}',
                row["timestamp"],
            ]) + "\n")

        print(f"\n[{exp_name}] DONE  |  test_loss={row['test_loss']:.6f}  test_mae={row['test_mae']:.6f}  test_corr={row['test_corr']:.6f}")
        print(f"Results appended to: {results_csv}")
        print(f"Artifacts saved under: {OUT_ROOT/exp_name}\n")
