#!/usr/bin/env python
"""reaction_center_fast.py — Parallel dataset builder + baseline GNN

This file supersedes the earlier notebook snippet.  It keeps the same
public API (command-line arguments, train/val/test split, model
architecture) while accelerating dataset construction via *joblib*.
Only speed-related edits were made; no other behaviour changes.
"""

from __future__ import annotations

import argparse, csv, os, re, sys, random
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
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
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score

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
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.t1 = TransformerConv(node_dim, hidden // 8, heads=8, edge_dim=edge_dim)
        self.t2 = TransformerConv(hidden, hidden, concat=False, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, ei, ea):
        x = F.leaky_relu(self.t1(x, ei, ea))
        x = self.dropout(x)
        return self.t2(x, ei, ea)

class LinkClassifier(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 3 + edge_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, h, ei, ea, batch):
        s, t = ei
        g = global_mean_pool(h, batch)[batch[s]]
        return self.mlp(torch.cat([h[s], h[t], ea, g], dim=-1))

class GNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.enc = Encoder(node_dim, edge_dim, hidden, dropout=dropout)
        self.cls = LinkClassifier(hidden, edge_dim, dropout=dropout)
    def forward(self, data: Data):
        h = self.enc(data.x, data.edge_index, data.edge_attr)
        logits = self.cls(h, data.edge_index, data.edge_attr, data.batch)
        return logits.squeeze(-1)

################################################################################
# Training utilities ----------------------------------------------------------
################################################################################

def get_loss_fn(pos_weight: float, use_focal=False, gamma=2.0):
    if use_focal:
        class FocalLoss(nn.Module):
            def __init__(self, alpha, gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            def forward(self, logits, target):
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=self.alpha)
                p_t = torch.exp(-bce)
                loss = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()
                return loss
        return FocalLoss(pos_weight, gamma)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

def train_one(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device, mc_dropout=0):
    model.eval()
    if mc_dropout > 0:
        for module in model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    y_true, y_scores = [], []
    iters = mc_dropout if mc_dropout > 0 else 1
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits_all = []
            for _ in range(iters):
                logits_all.append(model(batch))
            logits_all = torch.stack(logits_all)  # [T, N]
            logits_mean = logits_all.mean(0)
            y_scores.append(torch.sigmoid(logits_mean).cpu())
            y_true.append(batch.y.view(-1).cpu())
    y_true = torch.cat(y_true)
    y_scores = torch.cat(y_scores)
    return y_true.numpy(), y_scores.numpy()

def tune_threshold(y_true, y_score, beta=0.5):
    precision, recall, thresh = precision_recall_curve(y_true, y_score)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)
    idx = np.nanargmax(f_beta)
    return thresh[idx]


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
    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_tr, n_val, len(ds) - n_tr - n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch)

    pos = sum(int(d.y.sum()) for d in train_ds)
    neg = sum(int(d.y.numel() - d.y.sum()) for d in train_ds)
    pos_weight = neg / pos if pos > 0 else 1.0

    ensemble_scores, ensemble_true = [], None
    thresholds = []

    for seed in range(args.n_ensemble):
        print(f"\n===== ENSEMBLE MEMBER {seed+1}/{args.n_ensemble} (seed {seed}) =====")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = GNN(ds.num_node_features, ds.num_edge_features, hidden=args.hidden, dropout=args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = get_loss_fn(pos_weight, use_focal=args.focal).to(device)

        best_val_auc = 0.0
        patience = 0
        best_state = None

        if not args.no_train:
            print("Starting training...")
            for epoch in range(1, args.epochs + 1):
                tr_loss = train_one(model, train_loader, optimizer, loss_fn, device=device)
                y_v, s_v = eval_model(model, val_loader, device)
                val_auc = roc_auc_score(y_v, s_v)
                print(f"Ep{epoch:02d}: train_loss {tr_loss:.4f}  val_auc {val_auc:.3f}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = model.state_dict()
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if best_state:
                model.load_state_dict(best_state)
            torch.save(model.state_dict(), f"best_model_seed{seed}.pt")
            print(f"Best model for seed {seed} saved (val AUC: {best_val_auc:.3f})")
        else:
            model.load_state_dict(torch.load(f"best_model_seed{seed}.pt", map_location=device))

        y_v, s_v = eval_model(model, val_loader, device, mc_dropout=args.mc)
        thresh = tune_threshold(y_v, s_v, beta=args.beta)
        thresholds.append(thresh)
        print(f"Threshold for seed {seed} (F-beta={args.beta}): {thresh:.4f}")

        y_t, s_t = eval_model(model, test_loader, device, mc_dropout=args.mc)
        if ensemble_true is None:
            ensemble_true = y_t
        ensemble_scores.append(s_t)

    final_thresh = np.mean(thresholds)
    scores_mean = np.mean(ensemble_scores, axis=0)
    scores_var = np.var(ensemble_scores, axis=0)
    y_pred = (scores_mean >= final_thresh).astype(int)

    print("\n===== ENSEMBLE RESULTS =====")
    print(f"Using mean threshold: {final_thresh:.4f}")
    test_auc = roc_auc_score(ensemble_true, scores_mean)
    print(f"Test AUC: {test_auc:.3f}")
    print(classification_report(ensemble_true, y_pred, digits=3))
    np.savez("probabilities.npz", y=ensemble_true, p=scores_mean, var=scores_var)
    print("Saved probabilities and variance to probabilities.npz")

    if args.run_beam_search:
        print("\n" + "="*50)
        print("RUNNING BEAM SEARCH PIPELINE (on first ensemble model)")
        print("="*50)
        model.load_state_dict(torch.load("best_model_seed0.pt", map_location=device))
        beam_results = run_beam_search_pipeline(model, test_loader, args.beam_size, device=device)

        # Print results for a few examples
        for i in range(min(3, len(beam_results))):
            res = beam_results[i]
            print(f"\n--- Example Molecule {res['example_index']} (has {res['num_atoms']} atoms and {res['num_bonds']} bonds) ---")
            print(f"True reaction center: {res['true_bonds']}")
            print(f"Top {args.beam_size} Reaction Center Hypotheses (Score, {{Bond(s)}}):")
            for score, bond_set in res['hypotheses']:
                formatted_bonds = ", ".join([str(bond) for bond in bond_set])
                print(f"  Score: {score:.4f}, Bonds: {{{formatted_bonds}}}")


    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(ensemble_true, scores_mean)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate");  plt.ylabel("True Positive Rate")
    plt.title("ROC – Reaction-Centre Classifier (Ensemble)");  plt.legend();  plt.grid(alpha=.3)
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
    ap.add_argument("--no-train", action="store_true", help="Skip training and load best_model_seedN.pt")
    ap.add_argument("--run-beam-search", action="store_true", help="Run beam search pipeline on the test set")
    ap.add_argument("--beam-size", type=int, default=5, help="Beam size for hypothesis search")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early_stop", type=int, default=15)
    ap.add_argument("--n_ensemble", type=int, default=1, help="Number of models to train in ensemble")
    ap.add_argument("--mc", type=int, default=0, help="MC-Dropout passes at test (0=off)")
    ap.add_argument("--focal", action="store_true", help="Use Focal Loss instead of weighted BCE")
    ap.add_argument("--beta", type=float, default=0.5, help="F-beta for threshold search")
    main(ap.parse_args())
