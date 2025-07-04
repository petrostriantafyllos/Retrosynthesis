from __future__ import annotations
import argparse, csv, os, random, re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from rdkit import Chem
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
# 2.  CSV → graph (now stores RDKit mol + true bonds)
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

    if len(ea) == 0:
        return None

    g = Data(
        x=x,
        edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(ea),
        y=torch.tensor(yl,dtype=torch.float).view(-1,1)
    )
    g.mol_obj = prod
    g.true_break_bonds_atom_indices = torch.tensor(list(brk), dtype=torch.long)
    g.n_breaks = torch.tensor([min(len(brk),2)], dtype=torch.long)
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
        graphs=[g for g in graphs if g is not None]
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
            nn.Linear(hidden*3+edge_dim,128), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(128,1)
        )
    def forward(self,h,ei,ea,batch):
        s,t = ei
        g   = global_mean_pool(h,batch)[batch[s]]
        return self.mlp(torch.cat([h[s],h[t],ea,g],-1)).squeeze(-1)

class GNN(nn.Module):
    def __init__(self,node_dim:int,edge_dim:int,
                 hidden:int=128,layers:int=3,drop:float=0.1):
        super().__init__()
        self.enc = Encoder(node_dim,edge_dim,hidden,drop,layers)
        self.bond= BondHead(hidden,edge_dim,drop)
        self.count=nn.Sequential(
            nn.Linear(hidden,hidden//2), nn.ReLU(), nn.Linear(hidden//2,3)
        )
    def forward(self,data:Data):
        h=self.enc(data.x,data.edge_index,data.edge_attr)
        bond_logits=self.bond(h,data.edge_index,data.edge_attr,data.batch)
        g_emb=global_mean_pool(h,data.batch)
        return bond_logits, self.count(g_emb)

# ---------------------------------------------------------------------------
# 4.  Losses
# ---------------------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self,alpha:float,gamma:float):
        super().__init__()
        self.register_buffer("alpha_val", torch.tensor(alpha))
        self.gamma=gamma
    def forward(self,logits,target):
        p   = torch.sigmoid(logits)
        pt  = torch.where(target==1, p, 1-p)
        alpha_t = torch.where(target==1, self.alpha_val, 1.0 - self.alpha_val)
        bce_loss = -torch.log(pt + 1e-9)
        loss = alpha_t * ((1 - pt)**self.gamma) * bce_loss
        return loss.mean()

def make_bond_loss(pos_weight,focal:bool,gamma:float):
    if focal:
        alpha_for_focal = pos_weight / (1.0 + pos_weight)
        return BinaryFocalLoss(alpha=alpha_for_focal, gamma=gamma)
    else:
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

def topk_accuracy(ds,model,device,k:int)->float:
    model.eval(); correct=0
    with torch.no_grad():
        for g in ds:
            if getattr(g, 'mol_obj', None) is None or g.mol_obj.GetNumBonds() == 0:
                continue

            g_batch_data=g.to(device); g_batch_data.batch=torch.zeros(g_batch_data.num_nodes,device=device,dtype=torch.long)
            bl,cl=model(g_batch_data)
            probs=torch.sigmoid(bl).cpu().numpy()

            uniq_mask=g_batch_data.edge_index[0] < g_batch_data.edge_index[1]
            edges_unique=g_batch_data.edge_index.t()[uniq_mask].cpu().numpy()
            
            # FIX: Move mask to CPU before indexing NumPy array
            probs_unique=probs[uniq_mask.cpu()]

            if len(probs_unique) == 0:
                true_set={tuple(sorted(t)) for t in g.true_break_bonds_atom_indices.cpu().numpy()}
                if not true_set:
                    correct += 1
                continue

            pred_n = int(cl.softmax(-1).argmax(-1).item())
            pred_n = max(1,min(pred_n,2))

            top_prob_indices=probs_unique.argsort()[::-1]

            cand_sets=[]
            seen_frozensets=set()

            if pred_n==1:
                for idx_in_unique in top_prob_indices:
                    bond = tuple(sorted(edges_unique[idx_in_unique]))
                    current_set_frozen = frozenset({bond})
                    if current_set_frozen not in seen_frozensets:
                        seen_frozensets.add(current_set_frozen)
                        cand_sets.append(set(current_set_frozen))
                        if len(cand_sets) >= k:
                            break
            else: # pred_n == 2
                pool_size = min(len(top_prob_indices), max(pred_n, 10))
                
                for i in range(pool_size):
                    for j in range(i + 1, pool_size):
                        if len(cand_sets) >= k: break
                        bond1 = tuple(sorted(edges_unique[top_prob_indices[i]]))
                        bond2 = tuple(sorted(edges_unique[top_prob_indices[j]]))
                        s = frozenset({bond1, bond2})
                        if s not in seen_frozensets:
                            seen_frozensets.add(s)
                            cand_sets.append(set(s))
                    if len(cand_sets) >= k: break
            
            true_set={tuple(sorted(t)) for t in g.true_break_bonds_atom_indices.cpu().numpy()}
            if any(cs==true_set for cs in cand_sets): correct+=1
    return correct/len(ds)

# ---------------------------------------------------------------------------
# 6.  Optuna objective (Comprehensive Search)
# ---------------------------------------------------------------------------
def objective(trial,args,splits)->float:
    tr,val,_=splits
    dev="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # --- Hyperparameters for Comprehensive Search ---
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    drop = trial.suggest_float("drop", 0.0, 0.5)
    layers = trial.suggest_int("layers", 2, 4)
    focal = trial.suggest_categorical("focal", [True, False])
    gamma = trial.suggest_float("gamma", 1.0, 4.0) if focal else 2.0
    λ = trial.suggest_float("lambda_count", 0.5, 2.0)
    scheduler_type = trial.suggest_categorical("scheduler_type", ["None", "ReduceLROnPlateau", "CosineAnnealingLR"])

    scheduler_patience = None
    scheduler_factor = None
    scheduler_t_max = None
    if scheduler_type == "ReduceLROnPlateau":
        scheduler_patience = trial.suggest_int("scheduler_patience", 3, 10)
        scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.5)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler_t_max = trial.suggest_int("scheduler_t_max", args.epochs // 2, args.epochs * 2)

    pin_memory_enabled=torch.cuda.is_available()
    ltr=DataLoader(tr,batch_size=args.batch,shuffle=True, pin_memory=pin_memory_enabled)
    lva=DataLoader(val,batch_size=args.batch, pin_memory=pin_memory_enabled)

    pos=sum(int(d.y.sum()) for d in tr); neg=sum(int(d.y.numel()-d.y.sum()) for d in tr)
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
        
        if scheduler:
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(val_metric)
            elif scheduler_type == "CosineAnnealingLR":
                scheduler.step()

        if val_metric>best: best=val_metric; patience=0
        else: patience+=1
        if trial.should_prune(): raise optuna.TrialPruned()
        if patience>=args.early_stop: break
    return best

# ---------------------------------------------------------------------------
# 7.  Visual helpers
# ---------------------------------------------------------------------------
def plot_pr_roc(y,s,title=""):
    p,r,_=precision_recall_curve(y,s)
    au=average_precision_score(y,s)
    fpr,tpr,_=roc_curve(y,s)
    ra=roc_auc_score(y,s)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(r,p); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title} PR-curve  (AUPRC={au:.3f})"); plt.grid()
    plt.subplot(1,2,2); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title} ROC  (AUC={ra:.3f})"); plt.grid(); plt.tight_layout(); plt.show()

def plot_count_cm(ct,cp):
    labs=np.unique(np.concatenate([ct,cp]))
    cm=confusion_matrix(ct,cp,labels=labs)
    ConfusionMatrixDisplay(cm,display_labels=sorted(labs)).plot(cmap="Blues")
    plt.title("Count-head confusion matrix"); plt.show()

def plot_prob_hist(y,s):
    plt.figure(figsize=(8,5))
    sns.histplot(s[y==0],color="steelblue",kde=True,stat="density",alpha=.6,label="non-break")
    sns.histplot(s[y==1],color="firebrick",kde=True,stat="density",alpha=.6,label="break")
    plt.xlabel("Predicted probability"); plt.ylabel("Density")
    plt.title("Bond-break probability distribution"); plt.legend(); plt.show()

def show_examples(model,ds_subset,dev,n=5,require_pos=False):
    model.eval(); shown=0
    with torch.no_grad():
        for g_idx, g in enumerate(ds_subset):
            if shown>=n: break
            
            mol = getattr(g, 'mol_obj', None)
            if not mol or mol.GetNumBonds() == 0: 
                continue 

            if require_pos and g.true_break_bonds_atom_indices.numel()==0: continue
            
            g_batch_data = g.to(dev)
            g_batch_data.batch = torch.zeros(g_batch_data.num_nodes, device=dev, dtype=torch.long)
            
            bl,cl=model(g_batch_data)
            probs=torch.sigmoid(bl).cpu().numpy()
            
            uniq_mask=g_batch_data.edge_index[0]<g_batch_data.edge_index[1]
            edges_unique=g_batch_data.edge_index.t()[uniq_mask].cpu().numpy()
            
            # FIX: Move mask to CPU before indexing NumPy array
            probs_unique=probs[uniq_mask.cpu()]
            
            if len(probs_unique) == 0: continue

            pred_n=max(1,min(int(cl.softmax(-1).argmax(-1)),2))
            
            top_prob_indices=probs_unique.argsort()[::-1]
            predicted_bonds_atom_indices=[]
            for idx_in_unique in top_prob_indices[:pred_n]:
                predicted_bonds_atom_indices.append(tuple(edges_unique[idx_in_unique]))

            true_bonds_atom_indices=g.true_break_bonds_atom_indices.cpu().numpy()

            true_rdkit_bond_idxs = []
            for a1,a2 in true_bonds_atom_indices:
                bond = mol.GetBondBetweenAtoms(int(a1),int(a2))
                if bond: true_rdkit_bond_idxs.append(bond.GetIdx())
            
            pred_rdkit_bond_idxs = []
            for a1,a2 in predicted_bonds_atom_indices:
                bond = mol.GetBondBetweenAtoms(int(a1),int(a2))
                if bond: pred_rdkit_bond_idxs.append(bond.GetIdx())
            
            colors_dict={}
            COLOR_TRUE = (0.2, 0.8, 0.2)
            COLOR_PRED = (1.0, 0.2, 0.2)
            COLOR_OVERLAP = (1.0, 0.65, 0.0)

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
            plt.tight_layout(); plt.show()
            shown+=1

# ---------------------------------------------------------------------------
# 8.  Run modes
# ---------------------------------------------------------------------------
def run_train(args):
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
    
    pin_memory_enabled=torch.cuda.is_available()
    ltr=DataLoader(tr,batch_size=args.batch,shuffle=True,pin_memory=pin_memory_enabled)
    lva=DataLoader(val,batch_size=args.batch,pin_memory=pin_memory_enabled)
    lte=DataLoader(te,batch_size=args.batch,pin_memory=pin_memory_enabled)

    pos=sum(int(d.y.sum()) for d in tr); neg=sum(int(d.y.numel()-d.y.sum()) for d in tr)
    pos_weight = neg/pos if pos else 1.0

    bond_loss=make_bond_loss(pos_weight,args.focal,args.gamma).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    
    scheduler = None
    if args.scheduler_type == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max', 
                                                   factor=args.scheduler_factor, 
                                                   patience=args.scheduler_patience,
                                                   verbose=True)
    elif args.scheduler_type == "CosineAnnealingLR":
        final_t_max = args.scheduler_t_max if args.scheduler_t_max is not None else args.epochs
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=final_t_max)

    best=0; patience=0; best_state=None
    for ep in range(1,args.epochs+1):
        tl=train_one(model,ltr,opt,bond_loss,args.lambda_count,dev)
        m=eval_model(model,lva,dev)
        print(f"ep{ep:02d}  loss{tl:.3f}  valAUPRC{m['auprc']:.3f}")
        
        if scheduler:
            if args.scheduler_type == "ReduceLROnPlateau":
                scheduler.step(m['auprc'])
            elif args.scheduler_type == "CosineAnnealingLR":
                scheduler.step()

        if m["auprc"]>best: best=m["auprc"]; best_state=model.state_dict(); patience=0
        else: patience+=1
        if patience>=args.early_stop: break
    
    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(),"best_model.pt")
    else:
        print("Warning: No improvement on validation AUPRC. Saving model from last epoch as 'last_model.pt'.")
        torch.save(model.state_dict(),"last_model.pt")

    res=eval_model(model,lte,dev)
    t1=topk_accuracy(te,model,dev,1); t5=topk_accuracy(te,model,dev,5)
    print("\n=== TEST ===")
    print(f"ROC AUC {res['roc']:.3f}  PR AUC {res['auprc']:.3f}  CountAcc {res['count_acc']:.3f}")
    print(f"Top-1 {t1*100:.1f}%    Top-5 {t5*100:.1f}%")

    print("\n— Diagnostics —")
    plot_pr_roc(res['y_true'],res['y_score'],"Test")
    plot_count_cm(res['count_true'],res['count_pred'])
    plot_prob_hist(res['y_true'],res['y_score'])
    
    print("\n--- Visualizing Examples with True Breaks (Green/Orange) ---")
    positive_break_examples = [g for g in te if g.true_break_bonds_atom_indices.numel() > 0]
    sample_pos_breaks = random.sample(positive_break_examples, min(5, len(positive_break_examples)))
    show_examples(model,sample_pos_breaks,dev,5,True) 

    print("\n--- Visualizing Random Examples (might include no true breaks) ---")
    random_sample_ds = random.sample(list(te), min(3, len(te)))
    show_examples(model,random_sample_ds,dev,3,False)


def run_hpo(args):
    set_seed(args.seed)
    ds=CentreDataset(args.csv,args.jobs or os.cpu_count())
    n_tr,n_val=int(.8*len(ds)),int(.1*len(ds))
    te_size = len(ds) - n_tr - n_val
    splits=torch.utils.data.random_split(
        ds,[n_tr,n_val,te_size],
        generator=torch.Generator().manual_seed(args.seed))
    
    study=optuna.create_study(direction="maximize",
                             pruner=optuna.pruners.MedianPruner(
                                 n_warmup_steps=args.early_stop//2))
    study.optimize(lambda t:objective(t,args,splits),n_trials=args.n_trials)
    print("Best:",study.best_params,study.best_value)

# ---------------------------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Multi-task GNN for Retrosynthesis Reaction Center Prediction.")
    ap.add_argument("--csv",required=True, help="Path to the USPTO-50K CSV file.")
    ap.add_argument("--jobs",type=int,default=None, help="Number of parallel jobs for data processing. Defaults to CPU count.")
    ap.add_argument("--cpu",action="store_true", help="Force CPU usage even if CUDA is available.")
    ap.add_argument("--batch",type=int,default=32, help="Batch size for training and evaluation.")
    ap.add_argument("--epochs",type=int,default=50, help="Number of training epochs.")
    ap.add_argument("--early_stop",type=int,default=10, help="Patience for early stopping based on validation AUPRC.")
    ap.add_argument("--seed",type=int,default=42, help="Random seed for reproducibility.")

    # hyper-params for direct training
    ap.add_argument("--hidden",type=int,default=128, help="Hidden dimension of the GNN layers.")
    ap.add_argument("--layers",type=int,default=3, help="Number of GNN layers.")
    ap.add_argument("--dropout",type=float,default=0.1, help="Dropout rate.")
    ap.add_argument("--lr",type=float,default=1e-3, help="Learning rate for AdamW optimizer.")
    ap.add_argument("--focal",action="store_true", help="Use Focal Loss for bond prediction.")
    ap.add_argument("--gamma",type=float,default=2.0, help="Gamma parameter for Focal Loss.")
    ap.add_argument("--lambda_count",type=float,default=1.0, help="Weight for the count head loss.")

    # Scheduler arguments
    ap.add_argument("--scheduler_type", type=str, default="None", 
                    choices=["None", "ReduceLROnPlateau", "CosineAnnealingLR"],
                    help="Type of LR scheduler to use.")
    ap.add_argument("--scheduler_patience", type=int, default=5, 
                    help="Patience for ReduceLROnPlateau scheduler.")
    ap.add_argument("--scheduler_factor", type=float, default=0.5, 
                    help="Factor for ReduceLROnPlateau scheduler.")
    ap.add_argument("--scheduler_t_max", type=int, default=None,
                    help="T_max for CosineAnnealingLR scheduler. Defaults to --epochs.")

    # HPO
    ap.add_argument("--hpo",action="store_true", help="Run Optuna Hyperparameter Optimization.")
    ap.add_argument("--n_trials",type=int,default=40, help="Number of trials for Optuna HPO.")
    args=ap.parse_args()

    set_seed(args.seed)

    if args.hpo:
        run_hpo(args)
    else:
        run_train(args)