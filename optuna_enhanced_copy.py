from __future__ import annotations
import argparse, csv, os, random, re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from rdkit import Chem
# import rdkit.Chem.rdChemReactions # Not directly used for reaction validation in this version
from rdkit.Chem.Draw import MolToImage
from joblib import Parallel, delayed
import tqdm.auto as tqdm
import optuna

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)

import torch.optim.lr_scheduler as lr_scheduler
import sys
from pathlib import Path


def write_log(file,line):
    with open(f"{file}.log",'a',newline='') as f:
        f.write(f"{line}\n")

# ---------------------------------------------------------------------------
# 0.  Reproducibility helper
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# 1.  Atom / bond featurisation
# ---------------------------------------------------------------------------
_SYMBOLS = [
    "C","N","O","S","F","H","Si","P","Cl","Br","Li","Na","K",
    "Mg","B","Sn","I","Se","unk"
]
_SYMBOL_TO_IDX = {s:i for i,s in enumerate(_SYMBOLS)}
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_TO_IDX = {b:i for i,b in enumerate(_BOND_TYPES)}

def _one_hot(idx:int, dim:int)->torch.Tensor:
    v=torch.zeros(dim); v[idx]=1; return v

def atom_features(a:Chem.Atom)->torch.Tensor:
    return torch.cat([
        _one_hot(_SYMBOL_TO_IDX.get(a.GetSymbol(),_SYMBOL_TO_IDX["unk"]),
                 len(_SYMBOLS)),
        torch.tensor([
            a.GetFormalCharge(), a.GetTotalDegree(), a.GetTotalNumHs(),
            a.GetTotalValence(), float(a.GetIsAromatic()), float(a.IsInRing())
        ])
    ])

def bond_features(b:Chem.Bond)->torch.Tensor:
    return torch.cat([
        _one_hot(_BOND_TO_IDX[b.GetBondType()], len(_BOND_TYPES)),
        torch.tensor([float(b.GetIsConjugated()), float(b.IsInRing())])
    ])

# ---------------------------------------------------------------------------
# 2.  CSV → graph (stores RDKit mol + true bonds)
# ---------------------------------------------------------------------------
def _bonds_to_break(rcts:Sequence[Chem.Mol], prod:Chem.Mol):
    r,p=set(),set()
    for m in rcts:
        for b in m.GetBonds():
            a1,a2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
            if a1 and a2: r.add(tuple(sorted((a1,a2))))
    for b in prod.GetBonds():
        a1,a2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        if a1 and a2: p.add(tuple(sorted((a1,a2))))
    return p-r

def _pair_to_graph(pair:Tuple[List[Chem.Mol],Chem.Mol]):
    rcts, prod = pair

    # Note: The mol_obj is needed for atom_features even if no bonds
    # x = torch.stack([atom_features(a) for a in prod.GetAtoms()]) # This needs prod.GetNumAtoms() > 0

    if prod is None or prod.GetNumAtoms() == 0:
        return None

    x = torch.stack([atom_features(a) for a in prod.GetAtoms()])

    map2idx = {a.GetAtomMapNum():a.GetIdx()
               for a in prod.GetAtoms() if a.GetAtomMapNum()}
    brk = {tuple(sorted((map2idx[m1],map2idx[m2])))
           for m1,m2 in _bonds_to_break(rcts, prod)
           if m1 in map2idx and m2 in map2idx}

    ei,ea,yl = [],[],[]
    for b in prod.GetBonds():
        i,j=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat=bond_features(b)
        lbl = 1 if tuple(sorted((i,j))) in brk else 0
        for u,v in ((i,j),(j,i)):
            ei.append([u,v]); ea.append(feat); yl.append(lbl)

    # Skip single-atom products (no bonds)
    if len(ea) == 0:
        return None

    g = Data(
        x=x,
        edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(ea),
        y=torch.tensor(yl,dtype=torch.float).view(-1,1)
    )
    g.mol_obj = prod # Store RDKit Mol object
    g.true_break_bonds_atom_indices = torch.tensor(list(brk), dtype=torch.long)
    g.n_breaks = torch.tensor([min(len(brk),2)], dtype=torch.long) # Cap at 2 or 3 breaks for count head
    return g

def _row_to_pair(row):
    try:
        left,right = row["rxnSmiles_Mapping_NameRxn"].split(">>")
        idx=[int(i) for i in re.findall(r"\d+",row["reactantSet_NameRxn"])]
        rcts=[left.split(".")[i] for i in idx]
        prod=right.split(".")[0]
        r_mols=[Chem.MolFromSmiles(s) for s in rcts]
        p_mol=Chem.MolFromSmiles(prod)
        if any(m is None for m in r_mols) or p_mol is None:
            return None
        return r_mols,p_mol
    except Exception:
        return None

class CentreDataset(InMemoryDataset):
    def __init__(self,csv_path:str,jobs:int):
        self.csv_path,self.jobs=csv_path,jobs
        super().__init__(root=Path(csv_path).parent)
        if not Path(self.processed_paths[0]).exists():
            print(f"Processing data from {csv_path}...")
            self.process()
        # Add weights_only=Falsefor newer pythons
        self.data,self.slices=torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self): return ["centre_data.pt"]
    def _parse_pairs(self):
        rows=list(csv.DictReader(open(self.csv_path)))
        pairs=Parallel(n_jobs=self.jobs)(delayed(_row_to_pair)(r) for r in rows)
        return [p for p in pairs if p]
    def process(self):
        pairs=self._parse_pairs()
        graphs=Parallel(n_jobs=self.jobs,backend="multiprocessing",batch_size=128)(
            delayed(_pair_to_graph)(p) for p in tqdm.tqdm(pairs,desc="graphs"))
        graphs=[g for g in graphs if g is not None] # Filter out None graphs (e.g., from invalid SMILES or no-bond molecules)
        # Calculate positive edge ratio for display
        pos_edges = sum(int(g.y.sum()) for g in graphs)
        total_edges = sum(int(g.y.numel()) for g in graphs)
        print(f"Positive edge ratio in processed data = {pos_edges/total_edges:.2%}")

        data,slices=self.collate(graphs)
        Path(self.processed_dir).mkdir(parents=True,exist_ok=True)
        torch.save((data,slices), self.processed_paths[0])

# ---------------------------------------------------------------------------
# 3.  Model
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self,node_dim:int,edge_dim:int,hidden:int,drop:float,layers:int):
        super().__init__()
        self.layers=nn.ModuleList()
        self.layers.append(TransformerConv(node_dim, hidden//8, heads=8,
                                           edge_dim=edge_dim))
        for _ in range(layers-1):
            # (heads*hidden_per_head)
            self.layers.append(TransformerConv(hidden, hidden, concat=False,
                                               edge_dim=edge_dim))
        self.drop=nn.Dropout(drop)
    def forward(self,x,ei,ea):
        for conv in self.layers:
            x=F.leaky_relu(conv(x,ei,ea)); x=self.drop(x)
        return x

class BondHead(nn.Module):
    def __init__(self,hidden:int,edge_dim:int,drop:float):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(hidden*2+edge_dim,128), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(128,1)
        )
    def forward(self,h,ei,ea,batch):
        s,t = ei
        # g_mean = global_mean_pool(h, batch)[batch[s]]
        # g_max  = global_max_pool(h, batch)[batch[s]]
        # g      = torch.cat([g_mean, g_max], dim=-1)
        return self.mlp(torch.cat([h[s],h[t],ea],-1)).squeeze(-1)

class GNN(nn.Module):
    def __init__(self,node_dim:int,edge_dim:int,
                 hidden:int=128,layers:int=3,drop:float=0.1):
        super().__init__()
        self.enc = Encoder(node_dim,edge_dim,hidden,drop,layers)
        self.bond= BondHead(hidden,edge_dim,drop)
        self.count=nn.Sequential(
            nn.Linear(hidden*2,hidden//2), nn.ReLU(), nn.Linear(hidden//2,3) # Output for 0, 1, 2 breaks
        )
    def forward(self,data:Data):
        h=self.enc(data.x,data.edge_index,data.edge_attr)
        bond_logits=self.bond(h,data.edge_index,data.edge_attr,data.batch)
        g_mean = global_mean_pool(h,data.batch)
        g_max = global_max_pool(h, data.batch)
        g_emb = torch.cat([g_mean, g_max], dim=-1)
        return bond_logits, self.count(g_emb)

# ---------------------------------------------------------------------------
# 4.  Loss
# ---------------------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self,alpha:float,gamma:float):
        super().__init__()
        self.register_buffer("alpha_val", torch.tensor(alpha)) # Alpha is weight for positive class
        self.gamma=gamma
    def forward(self,logits,target):
        # p = sigmoid(logits)
        p   = torch.sigmoid(logits)
        # pt = p if target == 1 else 1-p (probability of the true class)
        pt  = torch.where(target==1, p, 1-p)
        
        # alpha_t is the balancing factor for the loss
        # It's alpha_val for positive class, (1 - alpha_val) for negative class
        alpha_t = torch.where(target==1, self.alpha_val, 1.0 - self.alpha_val)
        
        # BCE_loss = -log(pt)
        bce_loss = -torch.log(pt + 1e-9) # Added epsilon for numerical stability in log

        # Focal Loss = -alpha_t * (1 - pt)^gamma * log(pt)
        loss = alpha_t * ((1 - pt)**self.gamma) * bce_loss
        return loss.mean()

def make_bond_loss(pos_weight,focal:bool,gamma:float):
    # pos_weight here is `neg_count / pos_count` from the dataset
    if focal:
        # Calculate alpha for BinaryFocalLoss. This alpha is the weight for the positive class.
        # If pos_weight (neg_count/pos_count) is large, it means positive class is rare,
        # so we want alpha for the positive class to be large (closer to 1) to weight it more.
        # Example: if neg=90, pos=10 -> pos_weight=9. Then alpha_for_focal = 9/(1+9) = 0.9.
        # This correctly assigns 0.9 weight to positive errors and 0.1 to negative errors.
        alpha_for_focal = pos_weight / (1.0 + pos_weight)
        return BinaryFocalLoss(alpha=alpha_for_focal, gamma=gamma)
    else:
        # pos_weight for BCEWithLogitsLoss is a direct scalar multiplier for positive examples
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

# ---------------------------------------------------------------------------
# 5.  Train / evaluate
# ---------------------------------------------------------------------------
def train_one(model,loader,opt,bond_loss,λ,device):
    model.train(); tot=0
    for batch in loader:
        batch=batch.to(device); opt.zero_grad()
        bl,cl = model(batch)
        loss  = bond_loss(bl,batch.y.view(-1)) + λ*F.cross_entropy(cl,batch.n_breaks.view(-1))
        loss.backward(); opt.step()
        tot += loss.item()*batch.num_graphs
    return tot/len(loader.dataset)

@torch.no_grad()
def eval_model(model,loader,device)->Dict:
    model.eval()
    ys,ps,ct,cp=[],[],[],[]
    for b in loader:
        b=b.to(device)
        bl,cl=model(b)
        ys.append(b.y.view(-1).cpu().numpy())
        ps.append(torch.sigmoid(bl).cpu().numpy())
        ct.append(b.n_breaks.cpu().numpy());  
        cp.append(cl.softmax(-1).argmax(-1).cpu().numpy()) 
    
    y=np.concatenate(ys); p=np.concatenate(ps)
    ct=np.concatenate(ct); cp=np.concatenate(cp)

    return dict(
        roc=roc_auc_score(y,p),
        auprc=average_precision_score(y,p),
        count_acc=(ct==cp).mean(),
        y_true=y, y_score=p, count_true=ct, count_pred=cp
    )

# molecule-level
def topk_accuracy(ds,model,device,k:int)->float:
    model.eval(); correct=0
    with torch.no_grad():
        for g in ds:
            # Skip if mol_obj is None or has no bonds (should be handled by _pair_to_graph, but defensive)
            if getattr(g, 'mol_obj', None) is None or g.mol_obj.GetNumBonds() == 0:
                continue

            g_batch_data=g.to(device); g_batch_data.batch=torch.zeros(g_batch_data.num_nodes,device=device,dtype=torch.long)
            bl,cl=model(g_batch_data)
            probs=torch.sigmoid(bl).cpu().numpy()

            # Ensure unique edges
            uniq_mask=g_batch_data.edge_index[0] < g_batch_data.edge_index[1]
            edges_unique=g_batch_data.edge_index.t()[uniq_mask].cpu().numpy()
            probs_unique=probs[uniq_mask.cpu()]

            # If there are no unique edges after filtering, no breaks can be predicted.
            # Compare with true breaks.
            if len(probs_unique) == 0:
                true_set={tuple(sorted(t)) for t in g.true_break_bonds_atom_indices.cpu().numpy()}
                if not true_set: # If 0 true breaks and 0 predicted breaks
                    correct += 1
                continue

            pred_n = int(cl.softmax(-1).argmax(-1).item())
            pred_n = max(1,min(pred_n,2)) # safety clamp (>=1 και <=2)

            top_prob_indices=probs_unique.argsort()[::-1] # Indices into unique_edges

            cand_sets=[]

            if not args.no_multi_task:
                seen_frozensets=set() # prevent duplicate candidate sets

                if pred_n==1:
                    for idx_in_unique in top_prob_indices:
                        bond = tuple(sorted(edges_unique[idx_in_unique]))
                        current_set_frozen = frozenset({bond}) # Represents a set with one bond
                        if current_set_frozen not in seen_frozensets:
                            seen_frozensets.add(current_set_frozen)
                            cand_sets.append(set(current_set_frozen)) # mutable set
                            if len(cand_sets) >= k:
                                break
                else: # pred_n == 2
                    # Take top `pool_size` individual bonds to form pairs
                    pool_size = min(len(top_prob_indices), max(pred_n, 10)) # Consider at most 10 best individual bonds for pairs
                    
                    for i in range(pool_size):
                        for j in range(i + 1, pool_size): # Ensure i < j for unique pairs from distinct individual bonds
                            if len(cand_sets) >= k: break # Stop if we have k unique candidate sets
                            bond1 = tuple(sorted(edges_unique[top_prob_indices[i]]))
                            bond2 = tuple(sorted(edges_unique[top_prob_indices[j]]))
                            s = frozenset({bond1, bond2})
                            if s not in seen_frozensets:
                                seen_frozensets.add(s)
                                cand_sets.append(set(s)) # mutable set
                        if len(cand_sets) >= k: break
            else:
                for i in range(min(k, len(top_prob_indices))):
                    bond_idx = top_prob_indices[i]
                    bond = tuple(sorted(edges_unique[bond_idx]))
                    cand_sets.append({bond}) # A set containing one bond
            
            true_set={tuple(sorted(t)) for t in g.true_break_bonds_atom_indices.cpu().numpy()}
            if any(cs==true_set for cs in cand_sets): correct+=1
    return correct/len(ds)

# ---------------------------------------------------------------------------
# 6.  Optuna objective
# ---------------------------------------------------------------------------
def objective(trial,args,splits)->float:
    tr,val,_=splits
    dev="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Core GNN Hyperparameters
    lr=trial.suggest_float("lr",1e-4,5e-3,log=True)
    hidden=trial.suggest_categorical("hidden",[64,128,256])
    drop=trial.suggest_float("drop",0.0,0.5)
    layers=trial.suggest_int("layers",2,4)
    
    # Loss Function Hyperparameters
    focal=trial.suggest_categorical("focal",[True,False])
    gamma=trial.suggest_float("gamma",1.0,4.0) if focal else 2.0
    λ=trial.suggest_float("lambda",0.5,2.0)

    # Learning Rate Scheduler Hyperparameters
    scheduler_type = trial.suggest_categorical("scheduler_type", ["None", "ReduceLROnPlateau", "CosineAnnealingLR"])
    scheduler_patience = None
    scheduler_factor = None
    scheduler_t_max = None
    if scheduler_type == "ReduceLROnPlateau":
        scheduler_patience = trial.suggest_int("scheduler_patience", 3, 10)
        scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.5)
    elif scheduler_type == "CosineAnnealingLR":
        # T_max is typically related to total epochs
        scheduler_t_max = trial.suggest_int("scheduler_t_max", args.epochs // 2, args.epochs * 2)


    # Use pin_memory only if CUDA is available
    pin_memory_enabled=torch.cuda.is_available()
    ltr=DataLoader(tr,batch_size=args.batch,shuffle=True,
                   pin_memory=pin_memory_enabled)
    lva=DataLoader(val,batch_size=args.batch,
                   pin_memory=pin_memory_enabled)

    pos=sum(int(d.y.sum()) for d in tr)
    neg=sum(int(d.y.numel()-d.y.sum()) for d in tr)
    pos_weight = neg/pos if pos else 1.0

    model=GNN(tr.dataset.num_node_features,tr.dataset.num_edge_features,
              hidden,layers,drop).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    
    scheduler = None
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max', 
                                                   factor=scheduler_factor, 
                                                   patience=scheduler_patience,
                                                   verbose=False)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=scheduler_t_max)


    bond_loss=make_bond_loss(pos_weight,focal,gamma).to(dev)

    best=0; patience=0
    for ep in range(1,args.epochs+1):
        train_one(model,ltr,opt,bond_loss,λ,dev)
        m=eval_model(model,lva,dev)
        val_metric=m["auprc"]; trial.report(val_metric,ep)
        
        # Step the scheduler
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_metric) # Step with validation metric
        elif scheduler_type == "CosineAnnealingLR":
            scheduler.step() # Step every epoch

        if val_metric>best: best=val_metric; patience=0
        else: patience+=1
        if trial.should_prune(): raise optuna.TrialPruned()
        if patience>=args.early_stop: break
    return best

# ---------------------------------------------------------------------------
# 7.  Visual helpers
# ---------------------------------------------------------------------------
def plot_pr_roc(y,s,folder,title=""):
    p,r,_=precision_recall_curve(y,s)
    au=average_precision_score(y,s)
    fpr,tpr,_=roc_curve(y,s)
    ra=roc_auc_score(y,s)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(r,p); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title} PR-curve  (AUPRC={au:.3f})"); plt.grid()
    plt.subplot(1,2,2); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title} ROC  (AUC={ra:.3f})"); plt.grid(); plt.tight_layout()
    # plt.show()
    plt.savefig(f"{folder}/PR_ROC")
    plt.close()

def plot_count_cm(ct,cp,folder):
    labs=np.unique(np.concatenate([ct,cp])) # Get all unique labels present in true/pred
    cm=confusion_matrix(ct,cp,labels=labs)
    # Display labels might need to be sorted if not already
    cmd = ConfusionMatrixDisplay(cm,display_labels=sorted(labs)).plot(cmap="Blues")
    plt.title("Count-head confusion matrix")
    # plt.show()
    cmd.figure_.savefig(f"{folder}/CM")
    plt.close()

def plot_prob_hist(y,s, folder):
    plt.figure(figsize=(8,5))
    sns.histplot(s[y==0],color="steelblue",kde=False,stat="density",alpha=.6,label="non-break")
    sns.histplot(s[y==1],color="firebrick",kde=False,stat="density",alpha=.6,label="break")
    plt.xlabel("Predicted probability"); plt.ylabel("Density")
    plt.title("Bond-break probability distribution"); plt.legend()
    # plt.show()
    plt.savefig(f"{folder}/prob_hist")
    plt.close()

def show_examples(model,ds_subset,dev,n=5,require_pos=False, folder='', title=""):
    """
    Visualizes product molecules with predicted and true broken bonds highlighted.
    Args:
        model: Trained GNN model.
        ds_subset: A subset of the PyG dataset (e.g., test_ds or a random sample from it).
        dev: 'cpu' or 'cuda'.
        n: Number of examples to visualize.
        require_pos: If True, only show examples with at least one true breaking bond.
    """
    model.eval(); shown=0
    with torch.no_grad():
        for g_idx, g in enumerate(ds_subset): # Iterate through subset
            if shown>=n: break
            
            mol = getattr(g, 'mol_obj', None)
            # Skip if no RDKit mol object or no bonds
            if not mol or mol.GetNumBonds() == 0: 
                continue 

            if require_pos and g.true_break_bonds_atom_indices.numel()==0: continue
            
            # Prepare graph for model inference (single graph as a batch)
            g_batch_data = g.to(dev)
            g_batch_data.batch = torch.zeros(g_batch_data.num_nodes, device=dev, dtype=torch.long)
            
            bl,cl=model(g_batch_data)
            probs=torch.sigmoid(bl).cpu().numpy()
            
            # Extract unique bonds from edge_index
            uniq_mask=g_batch_data.edge_index[0]<g_batch_data.edge_index[1]
            edges_unique=g_batch_data.edge_index.t()[uniq_mask].cpu().numpy()
            probs_unique=probs[uniq_mask.cpu()]
            
            # another check
            if len(probs_unique) == 0: continue

            # Predicted number of breaks from count head
            pred_n=max(1,min(int(cl.softmax(-1).argmax(-1)),2))
            
            # Get top N predicted bonds (individual score)
            top_prob_indices=probs_unique.argsort()[::-1]
            predicted_bonds_atom_indices=[]
            for idx_in_unique in top_prob_indices[:pred_n]:
                predicted_bonds_atom_indices.append(tuple(edges_unique[idx_in_unique]))

            # True breaking bonds
            true_bonds_atom_indices=g.true_break_bonds_atom_indices.cpu().numpy()

            # Convert for highlighting
            true_rdkit_bond_idxs = []
            for a1,a2 in true_bonds_atom_indices:
                bond = mol.GetBondBetweenAtoms(int(a1),int(a2))
                if bond: true_rdkit_bond_idxs.append(bond.GetIdx())
            
            pred_rdkit_bond_idxs = []
            for a1,a2 in predicted_bonds_atom_indices:
                bond = mol.GetBondBetweenAtoms(int(a1),int(a2))
                if bond: pred_rdkit_bond_idxs.append(bond.GetIdx())
            
            colors_dict={}
            # Define colors: Green for true, Red for predicted, Orange for overlap
            COLOR_TRUE = (0.2, 0.8, 0.2) # Green
            COLOR_PRED = (1.0, 0.2, 0.2) # Red
            COLOR_OVERLAP = (1.0, 0.65, 0.0) # Orange

            for idx in true_rdkit_bond_idxs: colors_dict[idx]=COLOR_TRUE
            for idx in pred_rdkit_bond_idxs:
                if idx in colors_dict: colors_dict[idx]=COLOR_OVERLAP
                else: colors_dict[idx]=COLOR_PRED
            
            try:
                img=MolToImage(mol,highlightBonds=list(colors_dict.keys()),
                               highlightBondColors=colors_dict,kekulize=False)
            except Exception as e:
                print(f"Warning: Could not draw molecule (original index {g_idx}). Error: {e}")
                continue

            true_bonds_display = sorted([tuple(sorted(b)) for b in true_bonds_atom_indices])
            pred_bonds_display = sorted([tuple(sorted(b)) for b in predicted_bonds_atom_indices])

            plt.figure(figsize=(6,6)); plt.imshow(img); plt.axis("off")
            plt.title(f"True: {true_bonds_display} | Pred: {pred_bonds_display}",fontsize=8)
            plt.savefig("figs")
            plt.tight_layout(); plt.show()
            plt.savefig(f"{folder}/{title}-{shown}")
            shown+=1

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

            bl,cl = model(data)
            top_hypotheses = beam_search_bond_sets(bl, data.edge_index, beam_size=beam_size)
            
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
    # unique_edge_indices = torch.arange(len(logits))[mask]
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

def calculate_accuracies(beam_results):
    """
    Calculates and prints top-1, top-3, and top-5 accuracies.
    """
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_examples = len(beam_results)

    if total_examples == 0:
        print("Cannot calculate accuracy. The 'beam_results' list is empty.")
        return

    for res in beam_results:
        # Ensure true_bonds is a set for order-agnostic comparison
        true_bonds = (res['true_bonds'])
        
        # Extract the bond sets from the top 5 hypotheses
        # The `set(h[1])` ensures the predicted bonds are also treated as a set
        top10_hypotheses = [set(h[1]) for h in res['hypotheses'][:10]]

        # Check if the true bond set is in the top-k predictions
        # This implementation is cumulative. A top-1 match is also a top-3 and top-5 match.
        if true_bonds in top10_hypotheses:
            top10_correct+=1
            if true_bonds in top10_hypotheses[:5]:
                top5_correct += 1
                if true_bonds in top10_hypotheses[:3]:
                    top3_correct += 1
                    if true_bonds == top10_hypotheses[0]:
                        top1_correct += 1
    
    # Calculate and print the final accuracies
    top1_acc = (top1_correct / total_examples) * 100
    top3_acc = (top3_correct / total_examples) * 100
    top5_acc = (top5_correct / total_examples) * 100
    top10_acc = (top10_correct / total_examples) * 100

    log = f"\n--- Beamsearch Accuracy Results ---"
    log += f"\nTop-1 {top1_acc:.1f}%    Top-3 {top3_acc:.1f}%    Top-5 {top5_acc:.1f}%    Top-10 {top10_acc:.1f}%"
    print(log)
    write_log(log_file,log)

    # print(f"\n--- Beamsearch Accuracy Results ---")
    # print(f"Evaluated on {total_examples} examples.")
    # print(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{total_examples})")
    # print(f"Top-3 Accuracy: {top3_acc:.2f}% ({top3_correct}/{total_examples})")
    # print(f"Top-5 Accuracy: {top5_acc:.2f}% ({top5_correct}/{total_examples})")
    # print(f"Top-10 Accuracy: {top10_acc:.2f}% ({top10_correct}/{total_examples})")

# ---------------------------------------------------------------------------
# 8.  Run modes
# ---------------------------------------------------------------------------
def run_train(args):

    global log_file
    log_file = f"logs/new_logs/{args.name}"

    with open(f"{log_file}.log", 'a', newline='') as f:
        f.write(' '.join(sys.argv))
        f.write("\n")

    set_seed(args.seed)
    ds=CentreDataset(args.csv,args.jobs or os.cpu_count())
    n_tr,n_val=int(.8*len(ds)),int(.1*len(ds))
    te_size = len(ds) - n_tr - n_val
    tr,val,te=torch.utils.data.random_split(
        ds,[n_tr,n_val,te_size],
        generator=torch.Generator().manual_seed(args.seed))
    dev="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model=GNN(ds.num_node_features,ds.num_edge_features,
              args.hidden,args.layers,args.dropout).to(dev)
    
    # Use pin_memory only if CUDA is available
    pin_memory_enabled=torch.cuda.is_available()
    ltr=DataLoader(tr,batch_size=args.batch,shuffle=True,pin_memory=pin_memory_enabled)
    lva=DataLoader(val,batch_size=args.batch,pin_memory=pin_memory_enabled)
    lte=DataLoader(te,batch_size=args.batch,pin_memory=pin_memory_enabled)

    pos=sum(int(d.y.sum()) for d in tr); neg=sum(int(d.y.numel()-d.y.sum()) for d in tr)
    pos_weight = neg/pos if pos else 1.0 # Calculate pos_weight for the training set

    bond_loss=make_bond_loss(pos_weight,args.focal,args.gamma).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    
    scheduler = None
    if args.scheduler_type == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max', 
                                                   factor=args.scheduler_factor, 
                                                   patience=args.scheduler_patience,
                                                   verbose=True)
    elif args.scheduler_type == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if(not args.no_train):
        best=0; patience=0; best_state=None
        for ep in range(1,args.epochs+1):
            tl=train_one(model,ltr,opt,bond_loss,args.lambda_count,dev)
            m=eval_model(model,lva,dev)

            log = f"ep{ep:02d}  loss{tl:.3f}  valAUPRC{m['auprc']:.3f}"
            print(log)
            write_log(log_file,log)
            
            # Step the scheduler
            if scheduler:
                if args.scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(m['auprc']) # Step with validation metric
                elif args.scheduler_type == "CosineAnnealingLR":
                    scheduler.step() # Step every epoch

            if m["auprc"]>best: best=m["auprc"]; best_state=model.state_dict(); patience=0
            else: patience+=1
            if patience>=args.early_stop: break
    
        if best_state: # Only load if a better state was found
            model.load_state_dict(best_state)
            torch.save(model.state_dict(),f"best_model-{args.name}.pt")
        else:
            print("Warning: No improvement on validation AUPRC. Saving model from last epoch as 'last_model.pt'.")
            torch.save(model.state_dict(),"last_model.pt")
    else:
        model.load_state_dict(torch.load(f"best_model-{args.name}.pt", map_location=dev))

    fig_folder = f"figs/{args.name}"
    Path(fig_folder).mkdir(parents=True, exist_ok=True)
    res=eval_model(model,lte,dev)
    t1=topk_accuracy(te,model,dev,1); t3=topk_accuracy(te,model,dev,3); t5=topk_accuracy(te,model,dev,5); t10=topk_accuracy(te,model,dev,10)

    if args.run_beam_search:
        print("\n" + "="*50)
        print("RUNNING BEAM SEARCH PIPELINE (on first ensemble model)")
        print("="*50)

        global beam_results
        beam_results = run_beam_search_pipeline(model, lte, args.beam_size, device=dev)

        calculate_accuracies(beam_results)



    print("\n=== TEST ===")


    log = f"\n--- Multi-Task Accuracy Results ---"
    log += f"\nTop-1 {t1*100:.1f}%    Top-3 {t3*100:.1f}%    Top-5 {t5*100:.1f}%    Top-10 {t10*100:.1f}%"
    print(log)
    write_log(log_file,log)

    log= f"\n--- Overall Results ---"
    log += f"\nROC AUC {res['roc']:.3f}  PR AUC {res['auprc']:.3f}  CountAcc {res['count_acc']:.3f}"
    print(log)
    write_log(log_file,log)

    print("\n— Diagnostics —")
    plot_pr_roc(res['y_true'],res['y_score'],fig_folder,title="Test")
    plot_count_cm(res['count_true'],res['count_pred'],fig_folder)
    # plot_prob_hist(res['y_true'],res['y_score'],fig_folder)
    
    # Randomly sample from test set for visualization
    # Ensure num_examples doesn't exceed dataset size for random.sample
    print("\n--- Visualizing Examples with True Breaks (Green/Orange) ---")
    # Filter test_ds for graphs with at least one true break for these examples
    positive_break_examples = [g for g in te if g.true_break_bonds_atom_indices.numel() > 0]
    # Sample from these positive examples, up to 5
    sample_pos_breaks = random.sample(positive_break_examples, min(5, len(positive_break_examples)))
    show_examples(model,sample_pos_breaks,dev,5,True,folder=fig_folder,title=f"Pos_examples-{args.name}") 

    print("\n--- Visualizing Random Examples (might include no true breaks) ---")
    random_sample_ds = random.sample(list(te), min(3, len(te))) # Ensure not to sample more than available
    show_examples(model,random_sample_ds,dev,3,False,folder=fig_folder,title=f"Rand_examples-{args.name}")


def run_hpo(args):
    set_seed(args.seed)
    ds=CentreDataset(args.csv,args.jobs or os.cpu_count())
    n_tr,n_val=int(.8*len(ds)),int(.1*len(ds))
    te_size = len(ds) - n_tr - n_val # Calculate test size dynamically
    splits=torch.utils.data.random_split(
        ds,[n_tr,n_val,te_size], # Use calculated test size
        generator=torch.Generator().manual_seed(args.seed))
    
    study=optuna.create_study(direction="maximize",
                             pruner=optuna.pruners.MedianPruner(
                                 n_warmup_steps=args.early_stop//2))
    study.optimize(lambda t:objective(t,args,splits),n_trials=args.n_trials)
    print("Best:",study.best_params,study.best_value)

# ---------------------------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Multi-task GNN for Retrosynthesis Reaction Center Prediction.")
    ap.add_argument("--name", default="new_run", help="A name to differentiate logs, figures and the resulting best model.")
    ap.add_argument("--csv",default="data/dataSetB.csv", help="Path to the USPTO-50K CSV file.")
    ap.add_argument("--jobs",type=int,default=None, help="Number of parallel jobs for data processing. Defaults to CPU count.")
    ap.add_argument("--cpu",action="store_true", help="Force CPU usage even if CUDA is available.")
    ap.add_argument("--batch",type=int,default=32, help="Batch size for training and evaluation.")
    ap.add_argument("--epochs",type=int,default=50, help="Number of training epochs.")
    ap.add_argument("--early_stop",type=int,default=10, help="Patience for early stopping based on validation AUPRC.")
    ap.add_argument("--seed",type=int,default=42, help="Random seed for reproducibility.")
    ap.add_argument("--no-train", action="store_true", help="Skip training.")
    ap.add_argument("--no-multi-task", action="store_true", help="Skip multi-task evaluation.")

    # hyper-params for direct training
    ap.add_argument("--hidden",type=int,default=128, help="Hidden dimension of the GNN layers.")
    ap.add_argument("--layers",type=int,default=3, help="Number of GNN layers.")
    ap.add_argument("--dropout",type=float,default=0.1, help="Dropout rate.")
    ap.add_argument("--lr",type=float,default=1e-3, help="Learning rate for AdamW optimizer.")
    ap.add_argument("--focal",action="store_true", help="Use Focal Loss for bond prediction (alpha automatically calculated based on imbalance).")
    ap.add_argument("--gamma",type=float,default=2.0, help="Gamma parameter for Focal Loss (if focal is true).")
    ap.add_argument("--lambda_count",type=float,default=1.0, help="Weight for the count head loss in the total loss.")

    # NEW: Scheduler arguments for direct training
    ap.add_argument("--scheduler_type", type=str, default="None", 
                    choices=["None", "ReduceLROnPlateau", "CosineAnnealingLR"],
                    help="Type of LR scheduler to use.")
    ap.add_argument("--scheduler_patience", type=int, default=5, 
                    help="Patience for ReduceLROnPlateau scheduler.")
    ap.add_argument("--scheduler_factor", type=float, default=0.5, 
                    help="Factor by which to reduce LR for ReduceLROnPlateau scheduler.")
    # T_max for CosineAnnealingLR will be set to --epochs in run_train for simplicity.


    ap.add_argument("--run-beam-search", action="store_true", help="Run Beam Search on the first ensemble model.")
    ap.add_argument("--beam-size", type=int, default=10, help="Beam size for hypothesis search")

    # HPO
    ap.add_argument("--hpo",action="store_true", help="Run Optuna Hyperparameter Optimization.")
    ap.add_argument("--n_trials",type=int,default=40, help="Number of trials for Optuna HPO.")
    args=ap.parse_args()

    # Apply global seed setting (important for reproducibility)
    set_seed(args.seed)

    if args.hpo: run_hpo(args)
    else:         run_train(args)

    