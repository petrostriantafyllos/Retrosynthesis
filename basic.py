import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Embedding
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, global_add_pool
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# Suppress RDKit warnings for clarity
warnings.filterwarnings("ignore", category=UserWarning)


# Each entry contains:
#   - 'rxn_smiles': The reaction SMILES string with atom mapping.
#   - 'class_label': An integer representing the reaction class (our "semi-template").
DUMMY_DATASET = [
    {
        # A clearer example of esterification: Acetic Acid + Ethanol -> Ethyl Acetate
        "rxn_smiles": "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]",
        "class_label": 0, # "Esterification"
    },
    {
        # Diels-Alder with corrected product connectivity
        "rxn_smiles": "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH:6][C:7]#[N:8]>>[N:8]#[C:7][CH:6]1[CH2:5][CH2:4][CH:3]=[CH:2][CH2:1]1",
        "class_label": 1, # "Diels-Alder"
    },
    {
        "rxn_smiles": "[CH3:1][C:2](=[O:3])[CH2:4][Br:5].[O:6]=[C:7]([O-:8])[C:9]1=[CH:10][CH:11]=[CH:12][CH:13]=[CH:14]1.[K+]>>[CH3:1][C:2](=[O:3])[CH2:4][O:6][C:7](=O)[C:9]1=[CH:10][CH:11]=[CH:12][CH:13]=[CH:14]1",
        "class_label": 2, # "Williamson Ether Synthesis analogue"
    },
    {
        "rxn_smiles": "[c:1]1([CH3:7])[cH:2][c:3]([C:8](=[O:9])[OH:10])[cH:4][cH:5][c:6]1[N+:11](=[O:12])[O-:13].[CH3:14][OH:15]>>[c:1]1([CH3:7])[cH:2][c:3]([C:8](=[O:9])[O:15][CH3:14])[cH:4][cH:5][c:6]1[N+:11](=[O:12])[O-:13]",
        "class_label": 0, # "Esterification"
    }
]
NUM_REACTION_CLASSES = 3

def get_atom_features(atom):
    """ Extracts features for a single atom. """
    features = []
    features.append(atom.GetAtomicNum())
    features.append(atom.GetFormalCharge())
    features.append(atom.GetDegree())
    features.append(int(atom.GetHybridization()))
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetTotalNumHs())
    # Add a feature for atom map number to track atoms
    features.append(atom.GetAtomMapNum())
    return features

def get_reaction_center_label(rxn_smiles):
    """
    Identifies reaction centers by comparing reactants and products.
    Returns a dictionary mapping atom_map_num to a 0/1 label.
    1 = in reaction center, 0 = not in reaction center.
    """
    reactants_smi, products_smi = rxn_smiles.split(">>")
    
    # Create Mol objects
    reactants_mol = Chem.MolFromSmiles(reactants_smi, sanitize=False)
    products_mol = Chem.MolFromSmiles(products_smi, sanitize=False)
    
    # It's crucial to clean up and sanitize the molecules
    Chem.SanitizeMol(reactants_mol)
    Chem.SanitizeMol(products_mol)

    # Find reactant atoms that are part of the reaction center
    reaction_center_maps = set()
    
    # An atom is in the center if its connectivity or bonding changes
    for atom in reactants_mol.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap == 0: continue

        # Find corresponding atom in product
        product_atom = None
        for p_atom in products_mol.GetAtoms():
            if p_atom.GetAtomMapNum() == amap:
                product_atom = p_atom
                break
        
        # If atom is not in product (leaving group), it's in the center
        if product_atom is None:
            reaction_center_maps.add(amap)
            continue
            
        # Check for changes in degree (number of bonds)
        if atom.GetDegree() != product_atom.GetDegree():
            reaction_center_maps.add(amap)
            continue

        # Check for changes in bond orders with neighbors
        is_center = False
        for neighbor in atom.GetNeighbors():
            product_neighbor = None
            for p_neighbor in product_atom.GetNeighbors():
                if p_neighbor.GetAtomMapNum() == neighbor.GetAtomMapNum():
                    product_neighbor = p_neighbor
                    break
            
            if product_neighbor is None: # Bond broken
                is_center = True
                break
            
            bond = reactants_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            p_bond = products_mol.GetBondBetweenAtoms(product_atom.GetIdx(), product_neighbor.GetIdx())
            if bond.GetBondType() != p_bond.GetBondType(): # Bond order changed
                is_center = True
                break
        if is_center:
            reaction_center_maps.add(amap)

    return {atom.GetAtomMapNum(): 1 if atom.GetAtomMapNum() in reaction_center_maps else 0 
            for atom in reactants_mol.GetAtoms() if atom.GetAtomMapNum() > 0}


def smi_to_graph(rxn_smiles, class_label):
    reactants_smi, _ = rxn_smiles.split(">>")
    mol = Chem.MolFromSmiles(reactants_smi)
    
    if mol is None:
        return None

    atom_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    
    # Get ground-truth labels for each atom
    rc_labels_dict = get_reaction_center_label(rxn_smiles)
    labels = [rc_labels_dict.get(int(af[-1]), 0) for af in atom_features] # Use map number from features
    
    # Exclude atom map number from final features
    atom_features = torch.tensor(atom_features[:, :-1], dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Get bond information (edge_index)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i))
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create the graph data object
    data = Data(
        x=atom_features,
        edge_index=edge_index,
        y=labels.unsqueeze(1), # Target labels for atoms
        class_label=torch.tensor([class_label], dtype=torch.long) # "Semi-template" hint
    )
    return data


class ReactionDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class SemiTemplateGNN(Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_gat_layers, heads=4):
        super(SemiTemplateGNN, self).__init__()
        
        # Embedding layer for the reaction class (the "semi-template")
        self.class_embedding = Embedding(num_classes, hidden_dim)

        self.convs = torch.nn.ModuleList()
        # The input to the first GAT layer includes the node features + class embedding
        self.convs.append(GATConv(input_dim + hidden_dim, hidden_dim, heads=heads))
        for _ in range(num_gat_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))

        # Output classifier for each node (atom)
        self.atom_classifier = Linear(hidden_dim * heads, 1)

    def forward(self, data):
        x, edge_index, batch, class_label = data.x, data.edge_index, data.batch, data.class_label
        
        # 1. Get the semi-template embedding
        # class_label has shape [batch_size], embedding has shape [batch_size, hidden_dim]
        template_embed = self.class_embedding(class_label)
        
        # 2. Align embedding with nodes
        # We need to expand the template embedding to match the number of nodes in each graph
        # `template_embed[batch]` maps each node to its corresponding graph's template embedding
        aligned_template_embed = template_embed[batch]
        
        # 3. Concatenate atom features with the template hint
        x = torch.cat([x, aligned_template_embed], dim=-1)
        
        # 4. Pass through GAT layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.leaky_relu(x)
        
        # 5. Predict for each atom
        atom_predictions = self.atom_classifier(x)
        
        return atom_predictions


# ==============================================================================

def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict(model, rxn_smiles, class_label):
    model.eval()
    
    # Preprocess the input SMILES into a graph
    data = smi_to_graph(rxn_smiles, class_label)
    if data is None:
        print("Error: Could not parse reaction SMILES.")
        return None, None
        
    # The model expects a batch, so we create a mini-batch of size 1
    loader = DataLoader([data], batch_size=1)
    batch_data = next(iter(loader))
    
    # Get model predictions
    output = model(batch_data)
    
    # Apply sigmoid to get probabilities between 0 and 1
    probabilities = torch.sigmoid(output).squeeze()
    
    # Get original molecule to map results back
    mol = Chem.MolFromSmiles(rxn_smiles.split(">>")[0])
    
    print("\n--- Prediction Results ---")
    print(f"Reaction: {rxn_smiles}")
    print(f"Assumed Class: {class_label}")
    print("--------------------------")
    print("Atom Index | Atom Type | Map Num | P(is_center)")
    print("-------------------------------------------------")
    
    results = []
    for i, atom in enumerate(mol.GetAtoms()):
        prob = probabilities[i].item()
        results.append((i, atom.GetSymbol(), atom.GetAtomMapNum(), prob))
        print(f"{i:^10} | {atom.GetSymbol():^9} | {atom.GetAtomMapNum():^7} | {prob:.4f}")
        
    return results, data.y # Return predictions and true labels for comparison


# ==============================================================================
# PART 5: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    print("Processing dataset...")
    processed_data = [smi_to_graph(d['rxn_smiles'], d['class_label']) for d in DUMMY_DATASET]
    processed_data = [d for d in processed_data if d is not None]

    train_data, test_data = train_test_split(processed_data, test_size=0.25, random_state=42)
    
    train_dataset = ReactionDataset(train_data)
    test_dataset = ReactionDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Initialize Model, Optimizer, and Loss
    INPUT_FEATURES = 6 
    HIDDEN_DIM = 64
    
    model = SemiTemplateGNN(
        input_dim=INPUT_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_REACTION_CLASSES,
        num_gat_layers=3
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # Sigmoid + BCELoss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    print("\nStarting training on dummy data...")
    for epoch in range(25): # More epochs for better convergence
        loss = train(model, train_loader, optimizer, loss_fn)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}, Loss: {loss:.4f}")
            
    # Inference on test sample
    test_sample = DUMMY_DATASET[1] # The Diels-Alder reaction
    
    predictions, true_labels = predict(model, test_sample['rxn_smiles'], test_sample['class_label'])
    
    print("\n--- Ground Truth ---")
    print([int(l.item()) for l in true_labels])