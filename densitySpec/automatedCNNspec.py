#!/usr/bin/env python3
# sweep_runner.py
import os
import json
import math
import random
import shutil
import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Global config
# -----------------------------
PROJECT_SAVE_ROOT = Path("density_Spec_new_graphs")
NUM_EXPERIMENTS = 50
EPOCHS = 200
BATCH_SIZE = 16
LR = 5e-5
WEIGHT_DECAY = 1e-6
LOSS_TYPE = "l1"  # keep your default; change if you want
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG_SEED = 0  # reproducible splits

DATA_CUBE = "/home/ubuntu/datasets/new_graphs/stacked_cubes.npy"
DATA_CUBE_ELF = "/home/ubuntu/datasets/new_graphs/stacked_cubes_elf.npy"
DATA_SPEC = "/home/ubuntu/datasets/new_graphs/tddft_allspec.npy"

# -----------------------------
# Dataset
# -----------------------------
class CubeSpectrumDataset(Dataset):
    def __init__(self, cube_data, spec_data):
        self.cube_data = torch.tensor(cube_data, dtype=torch.float32)
        self.spec_data = torch.tensor(spec_data, dtype=torch.float32)

    def __len__(self):
        return len(self.cube_data)

    def __getitem__(self, idx):
        return self.cube_data[idx], self.spec_data[idx]

# -----------------------------
# Losses (as in your script)
# -----------------------------
class CorrelationMAELoss(nn.Module):
    def forward(self, y_pred, y_true):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        cost -= 5e1 * torch.mean(torch.abs(y_pred - y_true))
        return 1 - cost

class CorrelationLoss(nn.Module):
    def forward(self, y_pred, y_true):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - cost

class KLLoss(nn.Module):
    def forward(self, y_pred, y_true, eps=1e-7):
        y_pred = torch.clamp(y_pred, eps, 1.0)
        y_true = torch.clamp(y_true, eps, 1.0)
        return torch.sum(y_true * torch.log(y_true / y_pred), dim=1).mean()

class JSLoss(nn.Module):
    def forward(self, y_pred, y_true, eps=1e-7):
        y_pred = torch.clamp(y_pred, eps, 1.0)
        y_true = torch.clamp(y_true, eps, 1.0)
        m = 0.5 * (y_pred + y_true)
        kl1 = torch.sum(y_true * torch.log(y_true / m), dim=1)
        kl2 = torch.sum(y_pred * torch.log(y_pred / m), dim=1)
        return 0.5 * (kl1 + kl2).mean()

class CosineLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = F.normalize(y_pred, p=2, dim=1)
        y_true = F.normalize(y_true, p=2, dim=1)
        return 1 - torch.sum(y_pred * y_true, dim=1).mean()

LOSS_MAP = {
    "corrmae": CorrelationMAELoss(),
    "corr": CorrelationLoss(),
    "kl": KLLoss(),
    "js": JSLoss(),
    "cosine": CosineLoss(),
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "smoothl1": nn.SmoothL1Loss(),
    "corr+kl": lambda p, t: CorrelationLoss()(p, t) + 0.1 * KLLoss()(p, t),
    "corr+js": lambda p, t: CorrelationLoss()(p, t) + 0.1 * JSLoss()(p, t),
    "corr+cosine": lambda p, t: CorrelationLoss()(p, t) + 0.1 * CosineLoss()(p, t),
}

# -----------------------------
# Parametric CNN (same style)
# Conv3d → GN → ReLU → MaxPool3d → AvgPool3d repeated
# then Flatten → Dropout → Linear → Softmax
# -----------------------------
class CNNModel(nn.Module):
    """
    Same style as your provided architecture, but parameterised:
      - num_layers: 4..10 blocks
      - kernel_size: odd 3..15
      - base_filters: 2..8 (doubles up to 32*base and then capped)
      - dropout_p: float in {0.0, 0.1, 0.2, 0.3, 0.4}
    Dropout is placed (if p>0) after ~60% of blocks, after last block, and before the FC layer,
    mirroring your example (commented mid-drop, then active end + pre-FC).
    """
    def __init__(self, num_layers=10, kernel_size=7, base_filters=8, dropout_p=0.35, input_shape=(2, 165, 169, 155), out_features=2000):
        super().__init__()
        assert 4 <= num_layers <= 10
        assert kernel_size >= 3 and kernel_size <= 15
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.dropout_p = float(dropout_p)
        self.in_ch = 2
        self.input_shape = input_shape
        self.out_features = out_features

        # pooling layers (fixed, as in your style)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=2, padding=1)

        # build conv blocks
        blocks = []
        ks = self.kernel_size
        dilation = 1
        pad_same = int((ks - 1) / 2)  # we sample odd ks to keep shape consistent like your code

        # channel plan: filters, 2f, 4f, 8f, 16f, 32f, then stay at 32f
        def channel_mult(i):
            # i: 1-based block index
            if i == 1:  return 1
            if i == 2:  return 2
            if i == 3:  return 4
            if i == 4:  return 8
            if i == 5:  return 16
            return 32

        in_c = self.in_ch
        for i in range(1, num_layers + 1):
            out_c = self.base_filters * channel_mult(i)
            # cap to avoid exploding channels (keeps "style" identical to your 32*filters maximum)
            out_c = min(out_c, self.base_filters * 32)

            conv = nn.Conv3d(in_c, out_c, kernel_size=ks, padding=pad_same, dilation=dilation, padding_mode='zeros')
            gn = nn.GroupNorm(out_c, out_c)
            block = nn.Sequential(conv, gn, nn.ReLU(inplace=True))
            blocks.append(block)
            in_c = out_c

        self.blocks = nn.ModuleList(blocks)

        # Dropouts: mid-stack, end-stack, and pre-FC (Identity if p == 0)
        self.drop_mid = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
        self.drop_end = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
        self.drop_fc = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()

        self.flatten = nn.Flatten()
        # infer linear in-features by a dry run
        lin_in = self._infer_linear_in()
        self.linear = nn.Linear(lin_in, self.out_features)
        self.softmax = nn.Softmax(dim=1)

    def _infer_linear_in(self):
        with torch.no_grad():
            x = torch.rand(1, *self.input_shape)
            x = self._forward_features(x)
            x = self.flatten(x)
            return x.shape[1]

    def _forward_features(self, x):
        mid_index = max(1, math.floor(self.num_layers * 0.6))
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            x = self.avgpool(self.maxpool(x))
            if i == mid_index:
                x = self.drop_mid(x)
        x = self.drop_end(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.drop_fc(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

# -----------------------------
# Checkpoint utility
# -----------------------------
class CustomCheckpoint:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = Path(save_dir)
        self.best_val_loss = float('inf')
        self.best_model_path = self.save_dir / "best_model.pth"

    def __call__(self, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"✓ Saved best model at epoch {epoch+1} with val loss {val_loss:.6f}")

# -----------------------------
# Utils
# -----------------------------
def ensure_root():
    PROJECT_SAVE_ROOT.mkdir(parents=True, exist_ok=True)

def odd_between(a, b):
    """Return a random odd integer in [a, b]."""
    lo = a if a % 2 == 1 else a + 1
    hi = b if b % 2 == 1 else b - 1
    choices = list(range(lo, hi + 1, 2))
    return random.choice(choices)

def sample_config(idx):
    """Sample one experiment config within the requested ranges."""
    num_layers = random.randint(4, 10)
    kernel_size = odd_between(3, 15)  # keep odd to preserve your "same-ish" padding style
    base_filters = random.randint(2, 8)
    dropout_p = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])
    cfg = {
        "exp_id": f"exp_{idx:03d}",
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "base_filters": base_filters,
        "dropout_p": dropout_p,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss_type": LOSS_TYPE,
        "seed": RNG_SEED,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    return cfg

def write_model_file(exp_dir, cfg):
    """
    Write a portable model_def.py alongside weights so future loading is trivial.
    """
    text = f'''# Auto-generated model definition for {cfg["exp_id"]}
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_layers={cfg["num_layers"]}, kernel_size={cfg["kernel_size"]},
                 base_filters={cfg["base_filters"]}, dropout_p={cfg["dropout_p"]},
                 input_shape=(2, 165, 169, 155), out_features=2000):
        super().__init__()
        assert 4 <= num_layers <= 10
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.dropout_p = float(dropout_p)
        self.in_ch = 2
        self.input_shape = input_shape
        self.out_features = out_features

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=2, padding=1)

        blocks = []
        ks = self.kernel_size
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
            x = self._forward_features(x)
            lin_in = self.flatten(x).shape[1]

        self.linear = nn.Linear(lin_in, self.out_features)
        self.softmax = nn.Softmax(dim=1)

    def _forward_features(self, x):
        mid_index = max(1, int(self.num_layers * 0.6))
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            x = self.avgpool(self.maxpool(x))
            if i == mid_index:
                x = self.drop_mid(x)
        x = self.drop_end(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.drop_fc(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

def build_model(**cfg):
    return CNNModel(
        num_layers=cfg.get("num_layers", {cfg["num_layers"]}),
        kernel_size=cfg.get("kernel_size", {cfg["kernel_size"]}),
        base_filters=cfg.get("base_filters", {cfg["base_filters"]}),
        dropout_p=cfg.get("dropout_p", {cfg["dropout_p"]}),
    )
'''
    (exp_dir / "model_def.py").write_text(text)

def copy_this_script(exp_dir):
    # Best-effort copy of the runner itself for provenance
    try:
        this_path = Path(__file__).resolve()
        shutil.copy2(this_path, exp_dir / "sweep_runner.py")
    except Exception:
        # __file__ may be undefined in some environments; ignore quietly
        pass

def append_summary_row(row_dict):
    summary_path = PROJECT_SAVE_ROOT / "summary.csv"
    header = [
        "exp_id","num_layers","kernel_size","base_filters","dropout_p",
        "best_val_loss","best_val_corr","best_mae",
        "epochs","batch_size","lr","weight_decay","loss_type","seed","timestamp","exp_dir"
    ]
    exists = summary_path.exists()
    with summary_path.open("a") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        vals = [
            row_dict.get("exp_id"),
            str(row_dict.get("num_layers")),
            str(row_dict.get("kernel_size")),
            str(row_dict.get("base_filters")),
            str(row_dict.get("dropout_p")),
            f'{row_dict.get("best_val_loss", float("nan")):.6f}',
            f'{row_dict.get("best_val_corr", float("nan")):.6f}',
            f'{row_dict.get("best_mae", float("nan")):.6f}',
            str(row_dict.get("epochs")),
            str(row_dict.get("batch_size")),
            str(row_dict.get("lr")),
            str(row_dict.get("weight_decay")),
            row_dict.get("loss_type"),
            str(row_dict.get("seed")),
            row_dict.get("timestamp"),
            str(row_dict.get("exp_dir")),
        ]
        f.write(",".join(vals) + "\n")

# -----------------------------
# Training / Evaluation
# -----------------------------
def run_one_experiment(cfg, data, targets):
    exp_dir = PROJECT_SAVE_ROOT / cfg["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # persist config
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # write model_def.py for later re-load
    write_model_file(exp_dir, cfg)

    # copy this sweep file
    copy_this_script(exp_dir)

    # set seeds for split reproducibility
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # split
    train_cubes, val_cubes = data[:740], data[740:]
    train_spectra, val_spectra = targets[:740], targets[740:]

    train_ds = CubeSpectrumDataset(train_cubes, train_spectra)
    val_ds = CubeSpectrumDataset(val_cubes, val_spectra)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # model
    model = CNNModel(
        num_layers=cfg["num_layers"],
        kernel_size=cfg["kernel_size"],
        base_filters=cfg["base_filters"],
        dropout_p=cfg["dropout_p"],
        input_shape=(2, data.shape[2], data.shape[3], data.shape[4]),
        out_features=targets.shape[1],
    ).to(DEVICE)

    # losses
    criterion = LOSS_MAP[cfg["loss_type"]]
    corr_loss = CorrelationLoss()

    # optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    checkpoint = CustomCheckpoint(model, exp_dir)

    # logging file
    log_path = exp_dir / "train_log.csv"
    with log_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,mae,train_corr,val_corr\n")

    best_val_corr = -1e9
    best_mae = 1e9

    for epoch in range(cfg["epochs"]):
        # train
        model.train()
        train_loss_sum = 0.0
        tr_corr_sum = 0.0
        tr_count = 0
        for cubes, spectra in train_loader:
            cubes, spectra = cubes.to(DEVICE), spectra.to(DEVICE)
            optimizer.zero_grad()
            preds = model(cubes)
            loss = criterion(preds, spectra)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * cubes.size(0)
            tr_corr_sum += (1.0 - corr_loss(preds, spectra).item()) * spectra.size(0)
            tr_count += spectra.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        avg_tr_corr = tr_corr_sum / max(1, tr_count)

        # val
        model.eval()
        val_loss_sum = 0.0
        va_corr_sum = 0.0
        va_count = 0
        preds_list, targs_list = [], []
        with torch.no_grad():
            for cubes, spectra in val_loader:
                cubes, spectra = cubes.to(DEVICE), spectra.to(DEVICE)
                out = model(cubes)
                loss = criterion(out, spectra)
                val_loss_sum += loss.item() * cubes.size(0)
                va_corr_sum += (1.0 - corr_loss(out, spectra).item()) * spectra.size(0)
                va_count += spectra.size(0)
                preds_list.append(out.cpu().numpy())
                targs_list.append(spectra.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader.dataset)
        preds_np = np.concatenate(preds_list)
        targs_np = np.concatenate(targs_list)
        mae = float(np.mean(np.abs(preds_np - targs_np)))
        avg_val_corr = va_corr_sum / max(1, va_count)

        checkpoint(epoch, val_loss)
        # keep "best" tracking by correlation too (secondary)
        if avg_val_corr > best_val_corr:
            best_val_corr = avg_val_corr
            best_mae = min(best_mae, mae)

        # log
        with log_path.open("a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{mae:.6f},{avg_tr_corr:.6f},{avg_val_corr:.6f}\n")

        print(f"[{cfg['exp_id']}] Epoch {epoch+1}/{cfg['epochs']}  "
              f"Train {train_loss:.6f} | Val {val_loss:.6f} | MAE {mae:.6f} | "
              f"TrCorr {avg_tr_corr:.4f} | VaCorr {avg_val_corr:.4f}")

    # Final: record best (by val loss via checkpoint + best corr tracked)
    result = {
        "exp_id": cfg["exp_id"],
        "num_layers": cfg["num_layers"],
        "kernel_size": cfg["kernel_size"],
        "base_filters": cfg["base_filters"],
        "dropout_p": cfg["dropout_p"],
        "best_val_loss": float(checkpoint.best_val_loss),
        "best_val_corr": float(best_val_corr),
        "best_mae": float(best_mae),
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "loss_type": cfg["loss_type"],
        "seed": cfg["seed"],
        "timestamp": cfg["timestamp"],
        "exp_dir": str(exp_dir.resolve()),
    }

    # also save a tiny README
    (exp_dir / "README.txt").write_text(
        "This directory contains:\n"
        "- config.json (the exact hyperparameters)\n"
        "- model_def.py (architecture class and build_model helper)\n"
        "- best_model.pth (best weights by validation loss)\n"
        "- train_log.csv (per-epoch metrics)\n"
        "- sweep_runner.py (a copy of the sweep)\n"
    )

    return result

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_root()

    # Data loading & preprocessing (exactly like your script)
    cube_data = np.load(DATA_CUBE)
    spec_data = np.load(DATA_SPEC)

    cube_data_2 = np.load(DATA_CUBE_ELF)
    
    # normalize cubes
    cube_data = cube_data / np.max(cube_data)
    cube_data_2 = cube_data / np.max(cube_data_2)
    
    # concatenate features
    cube_data = np.concatenate((cube_data, cube_data_2), axis=1)

    # smooth spectra
    spec_data = gaussian_filter1d(spec_data, sigma=10)


    # shuffle in unison (seeded)
    np.random.seed(RNG_SEED)
    idx = np.arange(len(cube_data))
    np.random.shuffle(idx)
    cube_data = cube_data[idx]
    spec_data = spec_data[idx]

    # Torch performance tweaks
    torch.backends.cudnn.benchmark = True

    # Create experiments
    all_results = []
    for i in range(NUM_EXPERIMENTS):
        cfg = sample_config(i + 1)
        print(f"\n=== Starting {cfg['exp_id']} ===")
        print(cfg)
        res = run_one_experiment(cfg, cube_data, spec_data)
        append_summary_row(res)
        all_results.append(res)

    # Also save a JSON summary for convenience
    (PROJECT_SAVE_ROOT / "summary.json").write_text(json.dumps(all_results, indent=2))
    print("\nAll experiments complete. Summary written to:")
    print(f"- {PROJECT_SAVE_ROOT / 'summary.csv'}")
    print(f"- {PROJECT_SAVE_ROOT / 'summary.json'}")

if __name__ == "__main__":
    main()
