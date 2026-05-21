import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import GRUCell, Linear

from torch_geometric.nn import (
    GATv2Conv,
    PNAConv, 
    global_add_pool, 
    global_mean_pool,
    global_max_pool
)
from src import data

DEFAULT_AGGREGATORS = ("mean", "min", "max", "std")
DEFAULT_SCALERS = ("identity", "amplification", "attenuation")

def make_global_fragment_ids(batch, fragment_id):
    """
    Converts local fragment IDs into compact global fragment IDs.

    Example:
        batch:       [0, 0, 0, 1, 1]
        fragment_id: [0, 0, 1, 0, 1]

    Returns:
        atom_to_frag: [0, 0, 1, 2, 3]
        frag_to_mol: [0, 0, 1, 1]
    """
    fragment_id = fragment_id.long()
    batch = batch.long()

    # Need a multiplier larger than any local fragment id.
    max_local_frag = int(fragment_id.max().item()) + 1

    # Unique key for each molecule-fragment pair.
    frag_key = batch * max_local_frag + fragment_id

    # atom_to_frag maps each atom to a compact global fragment index.
    unique_keys, atom_to_frag = torch.unique(
        frag_key,
        sorted=True,
        return_inverse=True,
    )

    # Recover which molecule each global fragment belongs to.
    frag_to_mol = unique_keys // max_local_frag

    return atom_to_frag, frag_to_mol

def build_fragment_complete_graph(frag_to_mol):
    """
    Builds directed fragment-fragment edges within each molecule.

    If a molecule has fragments [0, 1, 2], this creates:
        0 -> 1, 0 -> 2
        1 -> 0, 1 -> 2
        2 -> 0, 2 -> 1

    Self-loops can be handled by GATv2Conv(add_self_loops=True).
    """
    device = frag_to_mol.device
    edges = []

    num_mols = int(frag_to_mol.max().item()) + 1 if frag_to_mol.numel() > 0 else 0

    for mol_idx in range(num_mols):
        frags = (frag_to_mol == mol_idx).nonzero(as_tuple=False).view(-1)

        if frags.numel() <= 1:
            continue

        src = frags.repeat_interleave(frags.numel())
        dst = frags.repeat(frags.numel())

        # Remove self edges here. Let GAT add self-loops if desired.
        mask = src != dst

        edge_index = torch.stack([src[mask], dst[mask]], dim=0)
        edges.append(edge_index)

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.cat(edges, dim=1)

class FragComboPNA(nn.Module):

    def __init__(
            self,
            in_channels=9,
            edge_dim=3,
            hidden_dim=64,
            aggregators=DEFAULT_AGGREGATORS,
            scalers=DEFAULT_SCALERS,
            deg=None,
            towers=1,
            out_dim=64,
            num_layers=2,
            num_timesteps=2,
            dropout: float=0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.aggregators = aggregators
        self.scalers = scalers
        self.deg = deg
        self.towers = towers
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_dim)

        self.gat = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout,
                            add_self_loops=False, # add_self_loops=False since GRU will handle self-loops
                            negative_slope=0.01) # negative_slope=0.01 for leakyReLU, to suppress negative values
        self.gru = GRUCell(hidden_dim, hidden_dim)

        # GAT / PNA Layers
        self.atom_gats = nn.ModuleList() 
        self.atom_pnas = nn.ModuleList()
        self.atom_lins = nn.ModuleList()
        self.atom_grus = nn.ModuleList()
        for _ in range(num_layers - 1):
            gat = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout,
                             add_self_loops=False, negative_slope=0.01)

            pna = PNAConv(in_channels=hidden_dim,
                          out_channels=hidden_dim, 
                          edge_dim=edge_dim, 
                          aggregators=self.aggregators,
                          scalers=self.scalers,
                          deg=self.deg,
                        )
            
            lin = Linear(2 * hidden_dim, hidden_dim)
            
            gru = GRUCell(hidden_dim, hidden_dim)

            self.atom_gats.append(gat)
            self.atom_pnas.append(pna)
            self.atom_lins.append(lin)
            self.atom_grus.append(gru)  

        self.mol_gat = GATv2Conv(hidden_dim, hidden_dim, dropout=dropout,
                                  add_self_loops=False, negative_slope=0.01)
        # self.mol_gat.explain = False # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_dim, hidden_dim)

        # Fragment attention 
        self.frag_gat = GATv2Conv(hidden_dim, hidden_dim, dropout=dropout,
                                  add_self_loops=False, negative_slope=0.01)
        self.frag_gru = GRUCell(hidden_dim, hidden_dim)
        self.frag_lin = Linear(hidden_dim, hidden_dim)

        self.lin2 = Linear(2 * hidden_dim, hidden_dim)
        self.lin3 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gat.reset_parameters()
        self.gru.reset_parameters()
        for gat, pna, lin, gru in zip(self.atom_gats, self.atom_pnas, self.atom_lins, self.atom_grus):
            gat.reset_parameters()
            pna.reset_parameters()
            lin.reset_parameters()
            gru.reset_parameters()
        self.mol_gat.reset_parameters()
        self.mol_gru.reset_parameters()
        self.frag_gat.reset_parameters()
        self.frag_gru.reset_parameters()
        self.frag_lin.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_attr, fragment_id, batch = data.x, data.edge_index, data.edge_attr, data.fragment_id, data.batch

        # Initial atom embedding
        x = self.lin1(x)
        x = F.leaky_relu(x)

        # Attention layers
        c = self.gat(x, edge_index, edge_attr) # Context vector
        c = F.elu(c)
        c = F.dropout(c, p=self.dropout, training=self.training)

        # Recursive update
        x = self.gru(c, x)
        x = F.relu(x)

        # GAT / PNA Layers
        for gat, pna, lin, gru in zip(self.atom_gats, self.atom_pnas, self.atom_lins, self.atom_grus):
            gatc = gat(x, edge_index, edge_attr)
            gatc = F.elu(gatc)
            gatc = F.dropout(gatc, p=self.dropout, training=self.training)

            pnac= pna(x, edge_index, edge_attr)
            pnac = F.elu(pnac)
            pnac = F.dropout(pnac, p=self.dropout, training=self.training)

            c = torch.cat([gatc, pnac], dim=-1) # Combine GAT and PNA outputs
            c = lin(c) # Linear layer to mix GAT and PNA features
            c = F.elu(c)

            x = gru(c, x)
            x = F.relu(x)

        atom_emb = x

        ############### Molecule embedding ###############
        row = torch.arange(batch.size(0), device=batch.device) # Atom indices
        mol_edge_index = torch.stack([row, batch], dim=0) # New edge_index for "supernode" molecule
        
        mol = global_add_pool(atom_emb, batch).relu_() # Initial molecule state vector

        # Repeat for num_timesteps
        for _ in range(self.num_timesteps):
            c = self.mol_gat((atom_emb, mol), mol_edge_index) # Attention
            c = F.elu(c)
            c = F.dropout(c, p=self.dropout, training=self.training)

            mol = self.mol_gru(c, mol)
            mol = F.relu(mol)

        ############### Fragment embedding ###############
        atom_to_frag, frag_to_mol = make_global_fragment_ids(batch=batch, fragment_id=fragment_id)

        num_frags = int(atom_to_frag.max().item() + 1)
        num_mols = int(batch.max().item()) + 1

        # Atom to fragment pooling
        frag = global_mean_pool(atom_emb, atom_to_frag, size=num_frags).relu()

        frag_edge_index = build_fragment_complete_graph(frag_to_mol)

        # Fragment attention
        c = self.frag_gat(frag, frag_edge_index)
        c = F.elu(c)
        c = F.dropout(c, p=self.dropout, training=self.training)

        frag = c

        # Fragment pooling
        mol_frag = global_mean_pool(frag, frag_to_mol, size=num_mols).relu()

        # Final concat and output
        mol_final = torch.cat([mol, mol_frag], dim=-1)
        mol_final = self.lin2(mol_final)
        mol_final = F.relu(mol_final)

        out = F.dropout(mol_final, p=self.dropout, training=self.training)
        out = self.lin3(out)

        return out
