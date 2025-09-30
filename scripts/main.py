from torch_geometric.nn.models import SchNet
from torch_geometric.nn import global_mean_pool
import torch, torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import add_self_loops
import os, glob, random
import numpy as np
import torch
from ase.io import read
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for saving files
import matplotlib.pyplot as plt
from model import SchNetSpectrum_attpoolv0, DimeNetPPSpectrum
import csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CorrelationLoss(nn.Module):
    def forward(self, y_pred, y_true):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        #cost -= torch.mean(torch.abs(y_pred - y_true))
        return 1 - cost

class KLLoss(nn.Module):
    def forward(self, y_pred, y_true, eps=1e-12):
        y_pred = torch.clamp(y_pred, eps, 1.0)
        y_true = torch.clamp(y_true, eps, 1.0)
        return torch.sum(y_true * torch.log(y_true / y_pred), dim=1).mean()

class JSLoss(nn.Module):
    def forward(self, y_pred, y_true, eps=1e-12):
        y_pred = torch.clamp(y_pred, eps, 1.0)
        y_true = torch.clamp(y_true, eps, 1.0)
        m = 0.5 * (y_pred + y_true)
        kl1 = torch.sum(y_true * torch.log(y_true / m), dim=1)
        kl2 = torch.sum(y_pred * torch.log(y_pred / m), dim=1)
        return 0.5 * (kl1 + kl2).mean()


class _DistUtils:
    @staticmethod
    def _to_batch_last(x: torch.Tensor) -> torch.Tensor:
        # Ensure float and [B, N] by adding batch dim if needed
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _normalize_nonneg(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # Clamp negatives (spectra can be tiny negative after nets/noise)
        x = torch.clamp(x, min=0)
        return x / (x.sum(dim=-1, keepdim=True) + eps)

class WassersteinLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = _DistUtils._to_batch_last(y_pred)
        y_true = _DistUtils._to_batch_last(y_true)

        if y_pred.shape[-1] != y_true.shape[-1]:
            raise ValueError(f"Pred/true bins must match; got {y_pred.shape} vs {y_true.shape}")

        y_pred = _DistUtils._normalize_nonneg(y_pred)
        y_true = _DistUtils._normalize_nonneg(y_true)

        cdf_pred = torch.cumsum(y_pred, dim=-1)
        cdf_true = torch.cumsum(y_true, dim=-1)

        wasserstein = torch.abs(cdf_pred - cdf_true).sum(dim=-1)  # [B]
        return wasserstein.mean()



def load_dataset(root_dir, usecols=1):
    xyz_paths  = sorted(glob.glob(os.path.join(root_dir, "*.xyz")))
    spec_paths = sorted(glob.glob(os.path.join(root_dir, "a_*.dat")))
    assert len(xyz_paths) == len(spec_paths) == 43

    spec_len = len(np.loadtxt(spec_paths[0], dtype=np.float32, usecols=usecols))
    data_list = []
    for xyz, sp in zip(xyz_paths, spec_paths):
        atoms = read(xyz)
        z  = torch.tensor([a.number for a in atoms], dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        y = np.loadtxt(sp, dtype=np.float32, usecols=usecols)[:spec_len//2]
        y/=np.sum(y)
        y  = torch.tensor(y, dtype=torch.float32)  # shape [spec_len]
        data_list.append(Data(z=z, pos=pos, y=y))
    return data_list, spec_len//2

def split_dataset(data_list, n_val=6, seed=42):
    rnd = random.Random(seed)
    rnd.shuffle(data_list)
    return data_list[n_val:], data_list[:n_val]


class CustomIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class SchNetSpectrum(nn.Module):
    def __init__(self, out_dim, hidden=128, interactions=6,
                 gaussians=50, cutoff=6.0):
        super().__init__()
        self.core = SchNet(hidden_channels=hidden,
                           num_filters=hidden,
                           num_interactions=interactions,
                           num_gaussians=gaussians,
                           cutoff=cutoff,
                           readout='add')  # anything non-None
        # bypass graph-level parts
        self.core.readout = CustomIdentity()
        self.core.lin1    = nn.Identity()
        self.core.lin2    = nn.Identity()
        
        fc_layers = [nn.Linear(hidden, out_dim), nn.Softmax(dim=1)]
        self.head = nn.Sequential(*fc_layers)
    def forward(self, batch):
        h_nodes = self.core(batch.z, batch.pos, batch.batch)   # [N, hidden]
        g_emb   = global_mean_pool(h_nodes, batch.batch)       # [B, hidden]
        return self.head(g_emb)                                # [B, out_dim]
        
        
def _pearson_r(a, b, eps=1e-12):
    """
    a, b: 1D arrays/tensors of same length
    returns scalar Pearson correlation in [-1, 1]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + eps
    return float((a*b).sum() / denom)

def plot_predictions(true_specs, pred_specs, save_path, max_samples=6):
    """
    true_specs, pred_specs: lists (or arrays) of shape [B, L]; both same length.
    Plots up to max_samples and titles each subplot with Pearson r.
    """
    n_samples = min(len(true_specs), len(pred_specs), max_samples)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        t = np.asarray(true_specs[i])
        p = np.asarray(pred_specs[i])
        r = _pearson_r(t, p)

        axes[i].plot(t, label="True Spectrum", color="black")
        axes[i].plot(p, label="Predicted Spectrum", color="red", linestyle="--")
        axes[i].set_ylabel("Intensity")
        axes[i].legend()
        axes[i].set_title(f"Sample {i+1}  •  Pearson r = {r:.3f}")

    axes[-1].set_xlabel("Frequency Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved spectra comparison plot to {save_path}")

def run_training(root, epochs=10000, batch_size=1, lr=8e-4, usecols=1, save_dir="checkpoints", cutoff=6.0, interactions=6, hidden=8, gaussians=10):
    save_dir = os.path.join(
        save_dir, f"schnet_att_cut{cutoff:g}_int{interactions}_hid{hidden}_g{gaussians}"
    )
    os.makedirs(save_dir, exist_ok=True)

    data_list, spec_len = load_dataset(root, usecols=usecols)
    train_set, val_set  = split_dataset(data_list, n_val=6)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    model = SchNetSpectrum_attpoolv0(
        out_dim=spec_len,
        hidden=hidden,
        interactions=interactions,
        gaussians=gaussians,
        cutoff=cutoff
    ).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss() #CorrelationLoss()#WassersteinLoss() #nn.L1Loss() #JSLoss()  # or nn.MSELoss()
    corr_loss = CorrelationLoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_model.pt")

    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        tr_loss, n_tr = 0.0, 0
        ploss_tr = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch)          # [B, spec_len]
            loss = loss_fn(pred, batch.y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch.num_graphs
            n_tr    += batch.num_graphs
            ploss_tr+= 1.0 - corr_loss(pred, batch.y).item() * batch.num_graphs

        # ---- val ----
        model.eval()
        va_loss, n_va = 0.0, 0
        ploss_va = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred  = model(batch)
                loss  = loss_fn(pred, batch.y)
                va_loss += loss.item() * batch.num_graphs
                n_va    += batch.num_graphs
                ploss_va+= 1.0 - corr_loss(pred, batch.y).item() * batch.num_graphs

        avg_val_loss = va_loss / n_va

        print(f"Epoch {ep:03d} | Train MSE {tr_loss/n_tr:.5e} | Train Corr {ploss_tr/n_tr:.5e} | "
              f"Val MSE {avg_val_loss:.5e} | Val Corr {ploss_va/n_va:.5e}")

        # ---- save best model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            best_val_corr = ploss_va/n_va
            print(f"✅ Best model updated (Val Loss: {best_val_loss:.5e}) saved at {best_model_path}")
            

    print(f"\nTraining complete. Best model saved at {best_model_path} with Val Loss: {best_val_loss:.5e}")
    with open(os.path.join(save_dir, "results.csv"), "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([cutoff, interactions, hidden, gaussians, best_val_loss, best_val_corr])

# ------------------ Load Best Model & Plot Predictions -------------------
    print("🔄 Loading best model for evaluation...")
    # Load best model and evaluate on validation set
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    

    true_specs, pred_specs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            pred  = model(batch)                     # [B, spec_len]
            y     = _DistUtils._to_batch_last(batch.y)
            for i in range(pred.shape[0]):
                pred_specs.append(pred[i].detach().cpu().numpy())
                true_specs.append(y[i].detach().cpu().numpy())
    
    # Ensure equal lengths
    assert len(true_specs) == len(pred_specs), \
        f"Length mismatch: {len(true_specs)} true vs {len(pred_specs)} pred"
    
    plot_predictions(true_specs, pred_specs, save_path=os.path.join(save_dir, "predictions.png"))


if __name__ == "__main__":
    root = "/home/ubuntu/datasets/hydrocarbons"
    data_list, spec_len = load_dataset(root)
    loader = DataLoader(data_list[:2], batch_size=2)
    batch  = next(iter(loader)).to(DEVICE)


    # then train:
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the best model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['dimenetpp', 'schnet_att'])
    parser.add_argument('--epochs', type=int, default=3500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cutoff', type=float, default=6.0)
    parser.add_argument('--interactions', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--gaussians', type=int, default=50)
    args = parser.parse_args()
    
    # build model
    if args.model == 'dimenetpp':
        model = DimeNetPPSpectrum(out_dim=spec_len).to(DEVICE)
    else:
        model = SchNetSpectrum_attpoolv0(out_dim=spec_len).to(DEVICE)

    with torch.no_grad():
        y = model(batch)
    print("Spec len:", spec_len, " → output:", y.shape)
    

    run_training(
        root,
        epochs=10000,
        batch_size=1,
        lr=args.lr,
        usecols=1,
        save_dir=args.save_dir,
        cutoff=args.cutoff,
        interactions=args.interactions,
        hidden=args.hidden,
        gaussians=args.gaussians,
    )

