import os
import pandas as pd
import torch
from rdkit import Chem


# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = "Dataset/cleaned_bbb_dataset.csv"
GRAPH_FOLDER = "graphs"

os.makedirs(GRAPH_FOLDER, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset = pd.read_csv(DATASET_FILE)


# ============================================================
# 2. ATOM FEATURES
# ============================================================

def get_atom_features(atom):

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetExplicitValence(),
        atom.GetImplicitValence(),
        int(atom.GetHybridization()),
    ]


# ============================================================
# 3. BOND FEATURES
# ============================================================

def get_bond_features(bond):

    bond_type = bond.GetBondType()

    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        int(bond.GetIsAromatic()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


# ============================================================
# 4. MOLECULE → GRAPH
# ============================================================

def molecule_to_graph(mol):

    # -------------------------
    # Nodes
    # -------------------------

    node_features = []

    for atom in mol.GetAtoms():

        node_features.append(
            get_atom_features(atom)
        )


    # -------------------------
    # Edges
    # -------------------------

    edge_index = []
    edge_features = []

    for bond in mol.GetBonds():

        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()

        features = get_bond_features(bond)

        # Forward
        edge_index.append(
            [source, target]
        )

        edge_features.append(
            features
        )

        # Reverse
        edge_index.append(
            [target, source]
        )

        edge_features.append(
            features
        )


    # -------------------------
    # Convert to tensors
    # -------------------------

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_features,
        dtype=torch.float
    )

    return x, edge_index, edge_attr


# ============================================================
# 5. GENERATE GRAPHS
# ============================================================

print("=" * 60)
print("MOLECULAR GRAPH GENERATION")
print("=" * 60)


graph_summary = []


for i, row in dataset.iterrows():

    smiles = row["SMILES"]

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue


    x, edge_index, edge_attr = molecule_to_graph(mol)


    # Save graph
    graph_file = os.path.join(
        GRAPH_FOLDER,
        f"graph_{i + 1:04d}.pt"
    )


    torch.save(
        {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": torch.tensor(
                [row["BBB"]],
                dtype=torch.long
            ),
            "smiles": smiles,
            "name": row["Name"],
            "source": row["Source"]
        },
        graph_file
    )


    graph_summary.append(
        {
            "ID": i + 1,
            "SMILES": smiles,
            "BBB": row["BBB"],
            "Nodes": x.shape[0],
            "Edges": edge_index.shape[1]
        }
    )


# ============================================================
# 6. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(graph_summary)

summary.to_csv(
    "Dataset/graph_summary.csv",
    index=False
)


# ============================================================
# 7. DISPLAY FIRST GRAPH
# ============================================================

first = graph_summary[0]

print("\nFirst Graph")
print("-" * 40)

print("SMILES :", first["SMILES"])
print("BBB    :", first["BBB"])
print("Nodes  :", first["Nodes"])
print("Edges  :", first["Edges"])


print("\n" + "=" * 60)
print("GRAPH GENERATION COMPLETE")
print("=" * 60)

print("Total graphs:", len(graph_summary))
print("Saved in   :", GRAPH_FOLDER)
