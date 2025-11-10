#!/usr/bin/env python3
# exp012_varlen_eval.py
import os, json, math, datetime, shutil
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Paths & constants -----------------------------
EXP_012_DIR = Path("/home/ubuntu/projects/densitySpec/density_Spec_new_graphs/exp_012")
OUT_ROOT    = Path("/home/ubuntu/projects/densitySpec/density_Spec_new_graphs/exp_012_varlen")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATA_CUBE     = "/home/ubuntu/datasets/new_graphs/stacked_cubes.npy"
DATA_CUBE_ELF = "/home/ubuntu/datasets/new_graphs/stacked_cubes_elf.npy"
DATA_SPEC     = "/home/ubuntu/datasets/new_graphs/tddft_allspec.npy"

TARGET_LENGTHS = [600, 800, 1000]  # predict these many points
TEST_SIZE = 48                      # keep same fixed test set
TRAIN_SIZE = 740                    # use all 740 samples for training (as before)

BATCH_SIZE = 16
EPOCHS = 200
LR = 5e-5
WEIGHT_DECAY = 1e-6
LOSS_TYPE = "l1"
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

# ----------------------------- Dataset & Losses -----------------------------
class CubeSpectrumDataset(Dataset):
    def __init__(self, cubes, specs):
        self.cubes = torch.tensor(cubes, dtype=torch.float32)
        self.specs = torch.tensor(specs, dtype=torch.float32)
    def __len__(self): return len(self.cubes)
    def __getitem__(self, i): return self.cubes[i], self.specs[i]

class CorrelationLoss(nn.Module):
    def forward(self, y_pred, y_true):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - cost

LOSS_MAP = {"l1": nn.L1Loss(), "mse": nn.MSELoss()}

# ----------------------------- Import exp_012 model -----------------------------
def import_model_from_exp012(exp_dir: Path):
    cfg = json.loads((exp_dir / "config.json").read_text()) if (exp_dir / "config.json").exists() else None
    model_mod = None
    if (exp_dir / "model_def.py").exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_def", exp_dir / "model_def.py")
        model_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_mod)

    if model_mod is not None and cfg is not None:
        def build(input_shape, out_features):
            m = model_mod.build_model(**cfg)
            # ensure input_shape is correct for dry-run sizing
            if hasattr(m, "input_shape"):
                m.input_shape = input_shape
            # re-create linear layer with new out_features
            with torch.no_grad():
                x = torch.rand(1, *input_shape)
                y = m._forward_features(x)
                lin_in = m.flatten(y).shape[1]
            m.linear = nn.Linear(lin_in, out_features)
            return m
        return build, cfg

    # Fallback: a parametric clone of exp_012
    class CNNModel(nn.Module):
        def __init__(self, input_shape, out_features, num_layers=7, kernel_size=5, base_filters=7, dropout_p=0.1):
            super().__init__()
            self.in_ch = input_shape[0]
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
                blocks.append(nn.Sequential(conv, gn, nn.ReLU(inplace=True)))
                in_c = out_c

            self.blocks = nn.ModuleList(blocks)
            self.drop_mid = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
            self.drop_end = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
            self.drop_fc = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
            self.flatten = nn.Flatten()
            with torch.no_grad():
                x = torch.rand(1, *self.input_shape)
                y = self._forward_features(x)
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

# ----------------------------- Data prep -----------------------------
def load_and_prepare():
    cubes = np.load(DATA_CUBE)
    cubes2 = np.load(DATA_CUBE_ELF)
    specs = np.load(DATA_SPEC)

    # concat channels (like your original)
    cubes = np.concatenate((cubes, cubes2), axis=1)

    # smooth
    specs = gaussian_filter1d(specs, sigma=10)

    # normalize cubes
    cubes = cubes / np.max(cubes)

    # same shuffle
    np.random.seed(SEED)
    idx = np.arange(len(cubes))
    np.random.shuffle(idx)
    return cubes[idx], specs[idx]

def renormalize_slice(specs_full, length):
    """Take first 'length' points and renormalize to sum=1 (ε-protected)."""
    sl = specs_full[:, :length].copy()
    sl = np.clip(sl, 1e-14, None)
    s = np.sum(sl, axis=1, keepdims=True)
    sl = sl / s
    return sl

# ----------------------------- Train/Eval -----------------------------
def train_one_length(length, cubes, specs_full, builder):
    import json, datetime
    from pathlib import Path
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    out_dir = OUT_ROOT / f"len_{length}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "length": length, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "loss_type": LOSS_TYPE,
        "seed": SEED
    }, indent=2))

    # targets: slice and renormalize to sum=1
    specs = renormalize_slice(specs_full, length)

    N = len(cubes)
    assert N >= TRAIN_SIZE + TEST_SIZE, "Dataset too small for requested split."
    test_idx  = np.arange(N - TEST_SIZE, N)
    train_idx = np.arange(0, TRAIN_SIZE)
    val_idx   = np.arange(TRAIN_SIZE, N - TEST_SIZE)  # may be empty

    train_ds = CubeSpectrumDataset(cubes[train_idx], specs[train_idx])
    test_ds  = CubeSpectrumDataset(cubes[test_idx],  specs[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # If val set is empty, we'll skip it and use train loss for checkpointing/logging
    use_val = len(val_idx) > 0
    if use_val:
        val_ds = CubeSpectrumDataset(cubes[val_idx], specs[val_idx])
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_shape = (cubes.shape[1], cubes.shape[2], cubes.shape[3], cubes.shape[4])
    out_features = length
    model = builder(input_shape, out_features).to(DEVICE)

    criterion = LOSS_MAP[LOSS_TYPE]
    corr_loss = CorrelationLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_path = out_dir / "best_model.pth"
    (out_dir / "train_log.csv").write_text("epoch,train_loss,val_loss,train_corr,val_corr\n")

    for ep in range(EPOCHS):
        # ---- train ----
        model.train()
        tr_loss_sum = tr_corr_sum = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
            tr_loss_sum += loss.item() * xb.size(0)
            tr_corr_sum += (1.0 - corr_loss(pred, yb).item()) * yb.size(0)
            tr_n += yb.size(0)

        tr_loss = tr_loss_sum / len(train_loader.dataset)
        tr_corr = tr_corr_sum / max(1, tr_n)

        # ---- val (optional) ----
        if use_val:
            model.eval()
            va_loss_sum = va_corr_sum = 0.0
            va_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    va_loss_sum += loss.item() * xb.size(0)
                    va_corr_sum += (1.0 - corr_loss(pred, yb).item()) * yb.size(0)
                    va_n += yb.size(0)
            va_loss = va_loss_sum / len(val_loader.dataset)
            va_corr = va_corr_sum / max(1, va_n)
            ckpt_metric = va_loss
        else:
            # no validation set: use training metrics for checkpointing/logging
            va_loss = tr_loss
            va_corr = tr_corr
            ckpt_metric = tr_loss

        if ckpt_metric < best_val:
            best_val = ckpt_metric
            torch.save(model.state_dict(), best_path)
            print(f"[len={length}] ✓ epoch {ep+1} best {'val' if use_val else 'train'} {ckpt_metric:.6f}")

        with (out_dir / "train_log.csv").open("a") as f:
            f.write(f"{ep+1},{tr_loss:.6f},{va_loss:.6f},{tr_corr:.6f},{va_corr:.6f}\n")

        print(f"[len={length}] {ep+1}/{EPOCHS}  train {tr_loss:.6f}  "
              f"{'val' if use_val else 'train'} {va_loss:.6f}  trCorr {tr_corr:.4f}  "
              f"{'vaCorr' if use_val else 'trCorr'} {va_corr:.4f}")

    # ---- test on fixed last-48 ----
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    test_l1 = test_corr_sum = mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            test_l1 += LOSS_MAP["l1"](pred, yb).item() * xb.size(0)
            test_corr_sum += (1.0 - corr_loss(pred, yb).item()) * yb.size(0)
            mae_sum += torch.mean(torch.abs(pred - yb)).item() * xb.size(0)
            n += xb.size(0)

    return {
        "length": length,
        "best_val_loss": best_val,
        "test_loss": test_l1 / n,
        "test_mae": mae_sum / n,
        "test_corr": test_corr_sum / n,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    # data
    cubes, specs_full = load_and_prepare()

    # builder from exp_012
    build_model, cfg = import_model_from_exp012(EXP_012_DIR)

    # results file
    results_csv = OUT_ROOT / "results_varlen.csv"
    if not results_csv.exists():
        results_csv.write_text("length,best_val_loss,test_loss,test_mae,test_corr,epochs,batch_size,lr,weight_decay,timestamp\n")

    # run lengths
    for L in TARGET_LENGTHS:
        res = train_one_length(L, cubes, specs_full, build_model)
        with results_csv.open("a") as f:
            f.write(",".join([
                str(res["length"]),
                f'{res["best_val_loss"]:.6f}',
                f'{res["test_loss"]:.6f}',
                f'{res["test_mae"]:.6f}',
                f'{res["test_corr"]:.6f}',
                str(res["epochs"]),
                str(res["batch_size"]),
                str(res["lr"]),
                str(res["weight_decay"]),
                res["timestamp"],
            ]) + "\n")
        print(f"[len={L}] DONE | test_loss={res['test_loss']:.6f}  test_corr={res['test_corr']:.6f}")

