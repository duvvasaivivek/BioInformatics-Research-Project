import os
import torch

from torch_geometric.data import Data, Dataset


# ============================================================
# CONFIG
# ============================================================

GRAPH_FOLDER = "graphs"


# ============================================================
# DATASET
# ============================================================

class BBBGraphDataset(Dataset):

    def __init__(self, graph_folder):

        self.graph_folder = graph_folder

        self.files = sorted(
            [
                f for f in os.listdir(graph_folder)
                if f.endswith(".pt")
            ]
        )

        super().__init__(None)


    def len(self):

        return len(self.files)


    def get(self, idx):

        file = self.files[idx]

        path = os.path.join(
            self.graph_folder,
            file
        )

        graph = torch.load(
            path,
            weights_only=False
        )

        data = Data(
            x=graph["x"],
            edge_index=graph["edge_index"],
            edge_attr=graph["edge_attr"],
            y=graph["y"]
        )

        data.smiles = graph["smiles"]
        data.name = graph["name"]
        data.source = graph["source"]

        return data


# ============================================================
# LOAD DATASET
# ============================================================

dataset = BBBGraphDataset(
    GRAPH_FOLDER
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("PYTORCH GEOMETRIC DATASET")
print("=" * 60)

print("Total graphs:", len(dataset))


# ============================================================
# CHECK FIRST GRAPH
# ============================================================

data = dataset[0]

print("\nFirst graph:")
print(data)


print("\nNode features:")
print(data.x.shape)


print("\nEdges:")
print(data.edge_index.shape)


print("\nEdge features:")
print(data.edge_attr.shape)


print("\nBBB label:")
print(data.y)


print("\nSMILES:")
print(data.smiles)


print("\nName:")
print(data.name)


print("\nSource:")
print(data.source)