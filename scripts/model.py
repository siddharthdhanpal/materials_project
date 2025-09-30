from torch_geometric.nn.models import SchNet
import torch, torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    GlobalAttention
)
from torch_geometric.nn.models import DimeNetPlusPlus



class CustomIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class SchNetSpectrum(nn.Module):
    def __init__(self, out_dim, hidden=8, interactions=6,
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


class SchNetSpectrum_attpoolv0(nn.Module):
    def __init__(self, out_dim, hidden=128, interactions=6,
                 gaussians=50, cutoff=6.0):
        super().__init__()
        self.core = SchNet(hidden_channels=hidden,
                           num_filters=hidden,
                           num_interactions=interactions,
                           num_gaussians=gaussians,
                           cutoff=cutoff,
                           readout='add')  # anything non-None
        
        self.att_gate = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
                           
        # bypass graph-level parts
        self.core.readout = CustomIdentity()
        self.core.lin1    = nn.Identity()
        self.core.lin2    = nn.Identity()
        self.att_pool = GlobalAttention(self.att_gate)

        fc_layers = [nn.Linear(4*hidden, out_dim), nn.Softmax(dim=1)]
        self.head = nn.Sequential(*fc_layers)
    def forward(self, batch):
        h_nodes = self.core(batch.z, batch.pos, batch.batch)   # [N, hidden]
        # ─── Multiple graph-level poolings ─────────────────────────────────────
        g_mean = global_mean_pool(h_nodes, batch.batch)
        g_max  = global_max_pool(h_nodes, batch.batch)
        g_add  = global_add_pool(h_nodes, batch.batch)
        g_att  = self.att_pool(h_nodes, batch.batch)          # attention-weighted sum
        combined = torch.cat([g_mean, g_max, g_add, g_att], dim=1)

        return self.head(combined)                                # [B, out_dim]
        

class DimeNetPPSpectrum(nn.Module):
    """
    Graph-level spectrum predictor using DimeNet++ as the core.
    We set out_channels=hidden so the core returns a graph embedding [B, hidden],
    then apply the same Softmax head you use for SchNet to get a normalized spectrum.
    """
    def __init__(
        self,
        out_dim: int,
        hidden: int = 128,
        interactions: int = 4,   # map to num_blocks in DimeNet++
        gaussians: int = 6,      # map to num_radial in DimeNet++
        cutoff: float = 6.0,
        num_spherical: int = 7,  # standard DimeNet++ setting
        envelope_exponent: int = 5,
        max_num_neighbors: int = 32,
    ):
        super().__init__()

        # DimeNet++ returns [B, out_channels]. We want a graph embedding first,
        # so set out_channels=hidden and add our own spectrum head after.
        self.core = DimeNetPlusPlus(
            hidden_channels=hidden,
            out_channels=hidden,           # <-- graph embedding
            num_blocks=interactions,       # reuse your "interactions" name
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=num_spherical,
            num_radial=gaussians,          # reuse your "gaussians" name
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            act='swish',
            output_initializer='zeros',
        )

        # final spectrum head (same style as your SchNetSpectrum)
        self.head = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.Softmax(dim=1)              # you’re using JS/KL losses ⇒ normalize
        )

    def forward(self, batch):
        # DimeNet++ builds its own radius graph & triplets internally
        g_emb = self.core(batch.z, batch.pos, batch.batch)  # [B, hidden]
        return self.head(g_emb)                              # [B, out_dim]

