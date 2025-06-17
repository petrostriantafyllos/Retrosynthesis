#!/usr/bin/env python
"""reaction_center_fast.py — Parallel dataset builder + baseline GNN

This file supersedes the earlier notebook snippet.  It keeps the same
public API (command-line arguments, train/val/test split, model
architecture) while accelerating dataset construction via *joblib*.
Only speed-related edits were made; no other behaviour changes.
"""

from __future__ import annotations

import argparse, csv, os, re, sys
from pathlib import Path
from typing import List, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from joblib import Parallel, delayed
import tqdm.auto as tqdm

################################################################################
# Atom / bond features --------------------------------------------------------
################################################################################

_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br', 'Li', 'Na', 'K',
    'Mg', 'B', 'Sn', 'I', 'Se', 'unk'
]
_SYMBOL_TO_IDX = {s: i for i, s in enumerate(_SYMBOLS)}
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_TO_IDX = {b: i for i, b in enumerate(_BOND_TYPES)}


def _one_hot(idx: int, dim: int) -> torch.Tensor:
    v = torch.zeros(dim); v[idx] = 1.0; return v


def atom_features(a: Chem.Atom) -> torch.Tensor:
    return torch.cat([
        _one_hot(_SYMBOL_TO_IDX.get(a.GetSymbol(), _SYMBOL_TO_IDX['unk']), len(_SYMBOLS)),
        torch.tensor([
            a.GetFormalCharge(), a.GetTotalDegree(), a.GetTotalNumHs(),
            a.GetTotalValence(), float(a.GetIsAromatic()), float(a.IsInRing())
        ])
    ])


def bond_features(b: Chem.Bond) -> torch.Tensor:
    return torch.cat([
        _one_hot(_BOND_TO_IDX[b.GetBondType()], len(_BOND_TYPES)),
        torch.tensor([float(b.GetIsConjugated()), float(b.IsInRing())])
    ])

################################################################################
# Graph conversion ------------------------------------------------------------
################################################################################

def _bonds_to_break(rcts: Sequence[Chem.Mol], prod: Chem.Mol):
    r, p = set(), set()
    for m in rcts:
        for b in m.GetBonds():
            a1, a2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
            if a1 and a2: r.add(tuple(sorted((a1, a2))))
    for b in prod.GetBonds():
        a1, a2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        if a1 and a2: p.add(tuple(sorted((a1, a2))))
    return p - r


def _pair_to_graph(pair: Tuple[Sequence[Chem.Mol], Chem.Mol]):
    rcts, prod = pair
    x = torch.stack([atom_features(at) for at in prod.GetAtoms()])

    # mapping for fast lookup (skip mapNum == 0)
    map2idx = {a.GetAtomMapNum(): a.GetIdx() for a in prod.GetAtoms() if a.GetAtomMapNum()}
    break_idx = {tuple(sorted((map2idx[m1], map2idx[m2]))) for m1, m2 in _bonds_to_break(rcts, prod) if m1 in map2idx and m2 in map2idx}

    ei, ea, el = [], [], []
    for b in prod.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat = bond_features(b)
        lbl = 1 if tuple(sorted((i, j))) in break_idx else 0
        for u, v in ((i, j), (j, i)):
            ei.append([u, v]); ea.append(feat); el.append(lbl)

    return Data(
        x=x,
        edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(ea),
        y=torch.tensor(el, dtype=torch.float).view(-1, 1)
    )

################################################################################
# CSV parsing helper ----------------------------------------------------------
################################################################################

def _row_to_pair(row):
    """Convert a CSV row to (reactant mols, product mol) or return None."""
    try:
        left, right = row['rxnSmiles_Mapping_NameRxn'].split('>>')
        idx = [int(i) for i in re.findall(r'\d+', row['reactantSet_NameRxn'])]
        rcts = [left.split('.')[i] for i in idx]
        prods = right.split('.')
        if len(prods) != 1:
            return None
        r_mols = [Chem.MolFromSmiles(s) for s in rcts]
        p_mol = Chem.MolFromSmiles(prods[0])
        rxn = rdChemReactions.ChemicalReaction(); [rxn.AddReactantTemplate(m) for m in r_mols]; rxn.AddProductTemplate(p_mol)
        if rxn.Validate()[1]:
            return None
        return r_mols, p_mol
    except Exception:
        return None

################################################################################
# Dataset ---------------------------------------------------------------------
################################################################################

class CentreDataset(InMemoryDataset):
    def __init__(self, csv_path: str, jobs: int):
        self.csv_path, self.jobs = csv_path, jobs
        super().__init__(root=Path(csv_path).parent)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ["centre_data.pt"]

    # ----- heavy I/O and RDKit in parallel -----
    def _parse_pairs(self):
        rows = list(csv.DictReader(open(self.csv_path)))
        print(f"CSV rows: {len(rows)} — RDKit parsing on {self.jobs} processes …")
        pairs = Parallel(n_jobs=self.jobs)(delayed(_row_to_pair)(r) for r in rows)
        return [p for p in pairs if p]

    def process(self):
        pairs = self._parse_pairs()
        print(f"Mol-pairs: {len(pairs)} — graph build on {self.jobs} threads …")
        graphs = Parallel(n_jobs=self.jobs, backend="multiprocessing", batch_size=128)(
            delayed(_pair_to_graph)(p) for p in tqdm.tqdm(pairs, desc="graphs")
        )
        pos = sum(int(g.y.sum()) for g in graphs)
        tot = sum(int(g.y.numel()) for g in graphs)
        print(f"Positive edge ratio = {pos/tot:.2%}")
        data, slices = self.collate(graphs)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

################################################################################
# Model -----------------------------------------------------------------------
################################################################################

class Encoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128):
        super().__init__()
        self.t1 = TransformerConv(node_dim, hidden // 8, heads=8, edge_dim=edge_dim)
        self.t2 = TransformerConv(hidden, hidden, concat=False, edge_dim=edge_dim)
    def forward(self, x, ei, ea):
        return self.t2(F.leaky_relu(self.t1(x, ei, ea)), ei, ea)

class LinkClassifier(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(node_dim * 3 + edge_dim, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, h, ei, ea, batch):
        s, t = ei
        g = global_mean_pool(h, batch)[batch[s]]
        return self.mlp(torch.cat([h[s], h[t], ea, g], dim=-1))

class GNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128):
        super().__init__()
        self.enc = Encoder(node_dim, edge_dim, hidden)
        self.cls = LinkClassifier(hidden, edge_dim)
    def forward(self, data: Data):
        h = self.enc(data.x, data.edge_index, data.edge_attr)
        return self.cls(h, data.edge_index, data.edge_attr, data.batch)

def _collect_preds(model, loader, device="cpu"):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logit = model(batch)
            # mask = batch.edge_index[0] < batch.edge_index[1]  # undirected collapse
            probs.append(torch.sigmoid(logit).cpu())
            labels.append(batch.y.cpu())
    return torch.cat(probs), torch.cat(labels)

################################################################################
# Training utilities ----------------------------------------------------------
################################################################################

def _roc_auc(y_true: torch.Tensor, y_prob: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score
    y_numpy, p_numpy = y_true.numpy(), y_prob.numpy()
    if y_numpy.sum() in (0, len(y_numpy)):
        return 0.0
    return float(roc_auc_score(y_numpy, p_numpy))


def _run_epoch(model, loader, criterion, opt=None, device="cpu"):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            logit = model(batch)
            loss = criterion(logit, batch.y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                # collapse undirected duplicates for metrics
                mask = batch.edge_index[0] < batch.edge_index[1]
                preds.append(torch.sigmoid(logit[mask]).cpu())
                labels.append(batch.y[mask].cpu())
            tot_loss += loss.item() * batch.num_graphs
    if train:
        return tot_loss / len(loader.dataset)
    else:
        if labels:
            p = torch.cat(preds); l = torch.cat(labels)
            return tot_loss / len(loader.dataset), _roc_auc(l, p)
        return tot_loss / len(loader.dataset), 0.0


def run_beam_search_pipeline(model, loader, beam_size, device="cpu"):
    """
    Runs beam search inference on a dataset and returns structured results.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(loader.dataset, desc="Beam Search")):
            # The loader's dataset gives individual data objects.
            # We need to manually create a batch for the model.
            data = data.to(device)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

            bond_logits = model(data)
            top_hypotheses = beam_search_bond_sets(bond_logits, data.edge_index, beam_size=beam_size)
            
            # Get true bond breaks for comparison
            true_bonds_mask = data.y.view(-1) == 1
            true_bonds_indices = data.edge_index.t()[true_bonds_mask]
            # Filter for unique bonds (i < j)
            unique_true_bonds_mask = true_bonds_indices[:, 0] < true_bonds_indices[:, 1]
            true_bonds = {tuple(bond.tolist()) for bond in true_bonds_indices[unique_true_bonds_mask]}

            results.append({
                "example_index": i,
                "num_atoms": data.num_nodes,
                "num_bonds": data.num_edges // 2,
                "true_bonds": true_bonds,
                "hypotheses": top_hypotheses
            })
    return results
    

import itertools
import torch.nn.functional as F

def beam_search_bond_sets(logits: torch.Tensor, edge_index: torch.Tensor, beam_size: int = 5):
    """
    Performs a hypothesis search to find the most likely sets of bond breaks.

    This is not a traditional sequence-decoding beam search, but a search
    for the best set of bond cleavages.

    Args:
        logits (torch.Tensor): The raw output logits from the model for each edge.
        edge_index (torch.Tensor): The edge_index of the graph.
        beam_size (int): The number of top hypotheses to return.

    Returns:
        A list of tuples, where each tuple contains (score, bond_indices).
        The list is sorted by score in descending order.
    """
    # Use logsigmoid to get log probabilities, which are numerically stable
    log_probs = F.logsigmoid(logits.squeeze())

    # --- Step 1: Handle undirected edges ---
    # We only want to consider each bond once (e.g., i->j, not j->i)
    # The mask ensures we only look at edges where source index < target index
    mask = edge_index[0] < edge_index[1]
    unique_edge_indices = torch.arange(len(logits))[mask]
    unique_log_probs = log_probs[mask]

    # Get the original bond indices (i, j) for easier interpretation
    bonds = edge_index.t()[mask]

    # --- Step 2: Generate single-bond hypotheses ---
    # Sort all unique bonds by their log probability
    sorted_indices = torch.argsort(unique_log_probs, descending=True)

    hypotheses = []
    # Add the top `beam_size` single bonds as initial hypotheses
    for i in range(min(beam_size, len(sorted_indices))):
        idx = sorted_indices[i]
        score = unique_log_probs[idx].item()
        bond_tuple = tuple(bonds[idx].tolist())
        hypotheses.append((score, {bond_tuple})) # Store hypothesis as a set of bonds

    # --- Step 3: Generate double-bond hypotheses ---
    # Consider combinations of the top N bonds to create double-break hypotheses
    # Let's use a slightly larger pool (e.g., 2*beam_size) for combinations
    pool_size = min(2 * beam_size, len(sorted_indices))
    top_bond_pool_indices = sorted_indices[:pool_size]

    for combo in itertools.combinations(top_bond_pool_indices, 2):
        idx1, idx2 = combo
        # Score is the sum of log probabilities
        score = (unique_log_probs[idx1] + unique_log_probs[idx2]).item()
        bond1_tuple = tuple(bonds[idx1].tolist())
        bond2_tuple = tuple(bonds[idx2].tolist())
        hypotheses.append((score, {bond1_tuple, bond2_tuple}))

    # --- Step 4: Rank all hypotheses and return the best ones ---
    hypotheses.sort(key=lambda x: x[0], reverse=True)

    return hypotheses[:beam_size]

################################################################################
# Main training loop ----------------------------------------------------------
################################################################################

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ds = CentreDataset(args.csv, args.jobs or os.cpu_count())

    n_tr, n_val = int(0.8 * len(ds)), int(0.1 * len(ds))
    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_tr, n_val, len(ds) - n_tr - n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)
    test_loader  = DataLoader(test_ds, batch_size=64)

    hidden = 128
    model = GNN(ds.num_node_features, ds.num_edge_features, hidden=hidden).to(device)

    pos = sum(int(d.y.sum()) for d in ds)
    neg = sum(int(d.y.numel() - d.y.sum()) for d in ds)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_auc = 0.0
    if not args.no_train:
        print("Starting training...")
        for epoch in range(1, args.epochs + 1):
            tr_loss = _run_epoch(model, train_loader, criterion, opt=optimizer, device=device)
            val_loss, val_auc = _run_epoch(model, val_loader, criterion, device=device)
            print(f"Ep{epoch:02d}: train {tr_loss:.4f}  val {val_loss:.4f}  AUC {val_auc:.3f}")
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), "best_model.pt")
        print("Training finished. Best model saved to best_model.pt")


    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load("best_model.pt"))
    _, test_auc = _run_epoch(model, test_loader, criterion, device=device)
    print(f"Test AUC = {test_auc:.3f}")

    if args.run_beam_search:
        print("\n" + "="*50)
        print("RUNNING BEAM SEARCH PIPELINE")
        print("="*50)
        
        beam_results = run_beam_search_pipeline(model, test_loader, args.beam_size, device=device)

        # Print results for a few examples
        for i in range(min(3, len(beam_results))):
            res = beam_results[i]
            print(f"\n--- Example Molecule {res['example_index']+1} (has {res['num_atoms']} atoms and {res['num_bonds']} bonds) ---")
            print(f"True reaction center: {res['true_bonds']}")
            print(f"Top {args.beam_size} Reaction Center Hypotheses (Score, {{Bond(s)}}):")
            for score, bond_set in res['hypotheses']:
                formatted_bonds = ", ".join([str(bond) for bond in bond_set])
                print(f"  Score: {score:.4f}, Bonds: {{{formatted_bonds}}}")


    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, roc_curve
    probs, y_true = _collect_preds(model, test_loader, device)
    y_pred = (probs > 0.50).int()  # threshold can be tuned

    print("\nPrecision / recall / F1 on test bonds:")
    print(classification_report(y_true.numpy(), y_pred.numpy(), digits=3))

    fpr, tpr, _ = roc_curve(y_true.numpy(), probs.numpy())
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate");  plt.ylabel("True Positive Rate")
    plt.title("ROC – Reaction-Centre Classifier");  plt.legend();  plt.grid(alpha=.3)
    plt.show()

################################################################################
# CLI -------------------------------------------------------------------------
################################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reaction-centre baseline with fast dataset build")
    ap.add_argument("--csv", required=True, help="CSV path with USPTO reactions")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=None, help="Parallel workers (default: all cores)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    ap.add_argument("--no-train", action="store_true", help="Skip training and load best_model.pt")
    ap.add_argument("--run-beam-search", action="store_true", help="Run beam search pipeline on the test set")
    ap.add_argument("--beam-size", type=int, default=5, help="Beam size for hypothesis search")
    main(ap.parse_args())