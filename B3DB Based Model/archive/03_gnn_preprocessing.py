import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

DATA_PATH = "data/B3DB_classification.tsv"
OUTPUT_PATH = "data/b3db_graphs.pt"

# Helper function to one-hot encode categorical features
def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    if index != -1:
        encoding[index] = 1
    return encoding

def get_node_features(atom):
    # Basic atom features
    # 1. Atomic number (C, N, O, S, F, Cl, Br, I, P, etc.)
    permitted_list_of_atoms = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'H']
    atom_type = one_hot_encoding(atom.GetSymbol(), permitted_list_of_atoms)
    
    # 2. Atom degree (number of attached atoms)
    degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # 3. Formal charge and radical electrons
    formal_charge = [atom.GetFormalCharge()]
    num_radical_electrons = [atom.GetNumRadicalElectrons()]
    
    # 4. Hybridization
    hybridization = one_hot_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'])
    
    # 5. Aromaticity
    is_aromatic = [1 if atom.GetIsAromatic() else 0]
    
    # 6. Mass
    mass = [atom.GetMass() * 0.01] # scale down slightly
    
    # Combine all into a single vector (approx 50 dimensions per atom)
    features = atom_type + degree + formal_charge + num_radical_electrons + hybridization + is_aromatic + mass
    return features

def smiles_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    # --- 1. NODE FEATURES (Atom level) ---
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_node_features(atom))
    
    # x represents our Node Features matrix (Num_Atoms x Num_Features)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # --- 2. EDGE INDEX & FEATURES (Bond level) ---
    edges_list = []
    edge_features_list = []
    
    # Define possible bond types for one-hot encoding
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
                  
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Extract edge features
        bond_type_feat = one_hot_encoding(bond.GetBondType(), bond_types)
        is_conjugated = [1 if bond.GetIsConjugated() else 0]
        is_in_ring = [1 if bond.IsInRing() else 0]
        
        # 6-dimensional edge feature vector
        edge_feat = bond_type_feat + is_conjugated + is_in_ring
        
        # PyTorch Geometric requires undirected graphs to have edges in both directions
        edges_list.append((i, j))
        edge_features_list.append(edge_feat)
        
        edges_list.append((j, i))
        edge_features_list.append(edge_feat)
        
    if len(edges_list) == 0:
        # Edge case: single atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
        
    # Target label
    y = torch.tensor([label], dtype=torch.float)
    
    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

def process_data():
    print(f"Loading dataset from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, sep='\t')
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}.")
        return

    df = df[['SMILES', 'BBB+/BBB-']].dropna()
    df['Label'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
    
    print("Converting molecules into 2D Graph representations...")
    graph_data_list = []
    
    # Loop through each molecule in the dataset
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(df)} molecules...")
            
        data = smiles_to_graph(row['SMILES'], row['Label'])
        if data is not None:
            graph_data_list.append(data)
            
    print(f"Successfully converted {len(graph_data_list)} molecules to graphs.")
    
    print(f"Saving list of Graph objects to {OUTPUT_PATH}...")
    torch.save(graph_data_list, OUTPUT_PATH)
    print("Done! You are ready for Phase 3 (GNN Training).")

if __name__ == "__main__":
    process_data()
