import torch

graph = torch.load(
    "graphs/graph_0001.pt",
    weights_only=False
)

print("=" * 50)
print("GRAPH CHECK")
print("=" * 50)

print("SMILES:")
print(graph["smiles"])

print("\nBBB:")
print(graph["y"])

print("\nNode feature matrix:")
print(graph["x"])

print("\nNode feature shape:")
print(graph["x"].shape)

print("\nEdge index:")
print(graph["edge_index"])

print("\nEdge index shape:")
print(graph["edge_index"].shape)

print("\nEdge attributes:")
print(graph["edge_attr"])

print("\nEdge attribute shape:")
print(graph["edge_attr"].shape)