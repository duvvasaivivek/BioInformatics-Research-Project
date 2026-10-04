import torch
import torch.nn.functional as F

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "best_bbb_gnn_model.pt"

HIDDEN_CHANNELS = 128
NODE_FEATURES = 9
EDGE_FEATURES = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("BBB GNN PREDICTION")
print("=" * 60)

print("PyTorch :", torch.__version__)
print("Device  :", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU     :",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA    :",
        torch.version.cuda
    )


# ============================================================
# MODEL
# MUST MATCH TRAINING MODEL
# ============================================================

class BBB_GNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # GINE 1
        # ----------------------------------------------------

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(
                NODE_FEATURES,
                HIDDEN_CHANNELS
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            )
        )

        self.conv1 = GINEConv(
            nn1,
            edge_dim=EDGE_FEATURES
        )

        self.bn1 = torch.nn.BatchNorm1d(
            HIDDEN_CHANNELS
        )


        # ----------------------------------------------------
        # GINE 2
        # ----------------------------------------------------

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            )
        )

        self.conv2 = GINEConv(
            nn2,
            edge_dim=EDGE_FEATURES
        )

        self.bn2 = torch.nn.BatchNorm1d(
            HIDDEN_CHANNELS
        )


        # ----------------------------------------------------
        # GINE 3
        # ----------------------------------------------------

        nn3 = torch.nn.Sequential(
            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS
            )
        )

        self.conv3 = GINEConv(
            nn3,
            edge_dim=EDGE_FEATURES
        )

        self.bn3 = torch.nn.BatchNorm1d(
            HIDDEN_CHANNELS
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.fc1 = torch.nn.Linear(
            HIDDEN_CHANNELS,
            32
        )

        self.fc2 = torch.nn.Linear(
            32,
            2
        )


    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch
    ):

        # GINE 1
        x = self.conv1(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn1(x)

        x = F.relu(x)


        # GINE 2
        x = self.conv2(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn2(x)

        x = F.relu(x)


        # GINE 3
        x = self.conv3(
            x,
            edge_index,
            edge_attr
        )

        x = self.bn3(x)

        x = F.relu(x)


        # ----------------------------------------------------
        # GRAPH POOLING
        # ----------------------------------------------------

        # Only one molecule is being predicted,
        # so mean over all its atoms.

        x = torch.mean(
            x,
            dim=0,
            keepdim=True
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

model = BBB_GNN().to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("\nModel loaded:")
print(MODEL_FILE)


# ============================================================
# ATOM FEATURES
# EXACTLY MATCHES GRAPH GENERATION
# ============================================================

def get_atom_features(atom):

    return [

        # 1. Atomic number
        atom.GetAtomicNum(),

        # 2. Degree
        atom.GetDegree(),

        # 3. Formal charge
        atom.GetFormalCharge(),

        # 4. Number of hydrogens
        atom.GetTotalNumHs(),

        # 5. Aromatic
        int(
            atom.GetIsAromatic()
        ),

        # 6. Ring
        int(
            atom.IsInRing()
        ),

        # 7. Explicit valence
        atom.GetValence(
            Chem.ValenceType.EXPLICIT
        ),

        # 8. Implicit valence
        atom.GetValence(
            Chem.ValenceType.IMPLICIT
        ),

        # 9. Hybridization
        int(
            atom.GetHybridization()
        )
    ]


# ============================================================
# BOND FEATURES
# EXACTLY MATCHES GRAPH GENERATION
# ============================================================

def get_bond_features(bond):

    bond_type = bond.GetBondType()

    return [

        # 1. Single
        float(
            bond_type == Chem.BondType.SINGLE
        ),

        # 2. Double
        float(
            bond_type == Chem.BondType.DOUBLE
        ),

        # 3. Triple
        float(
            bond_type == Chem.BondType.TRIPLE
        ),

        # 4. Aromatic bond
        float(
            bond_type == Chem.BondType.AROMATIC
        ),

        # 5. Aromatic
        int(
            bond.GetIsAromatic()
        ),

        # 6. Conjugated
        int(
            bond.GetIsConjugated()
        ),

        # 7. Ring
        int(
            bond.IsInRing()
        )
    ]


# ============================================================
# SMILES → GRAPH
# ============================================================

def molecule_to_graph(smiles):

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        raise ValueError(
            "Invalid SMILES."
        )


    # --------------------------------------------------------
    # NODE FEATURES
    # --------------------------------------------------------

    node_features = []

    for atom in mol.GetAtoms():

        node_features.append(
            get_atom_features(atom)
        )


    # --------------------------------------------------------
    # EDGE FEATURES
    # --------------------------------------------------------

    edge_index = []

    edge_features = []


    for bond in mol.GetBonds():

        source = bond.GetBeginAtomIdx()

        target = bond.GetEndAtomIdx()

        features = get_bond_features(
            bond
        )


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


    # --------------------------------------------------------
    # TENSORS
    # --------------------------------------------------------

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )


    if len(edge_index) > 0:

        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()


        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_attr = torch.empty(
            (0, EDGE_FEATURES),
            dtype=torch.float
        )


    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    batch = torch.zeros(
        x.size(0),
        dtype=torch.long
    )


    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_bbb(smiles):

    graph = molecule_to_graph(
        smiles
    )


    graph = graph.to(
        DEVICE
    )


    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch
        )


        probabilities = F.softmax(
            output,
            dim=1
        )


        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()


        prob_0 = probabilities[
            0,
            0
        ].item()


        prob_1 = probabilities[
            0,
            1
        ].item()


    confidence = max(
        prob_0,
        prob_1
    )


    return (
        prediction,
        confidence,
        prob_0,
        prob_1,
        graph
    )


# ============================================================
# CONTINUOUS TESTING
# ============================================================

print("\n" + "=" * 60)
print("BBB COMPOUND TESTING")
print("=" * 60)

print("\nEnter one SMILES at a time.")
print("The program will keep running.")
print("Type 'exit' or 'quit' to stop.")


while True:

    print("\n" + "-" * 60)

    smiles = input(
        "Enter SMILES: "
    ).strip()


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if smiles.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print("\nExiting BBB prediction.")

        break


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not smiles:

        print(
            "Please enter a SMILES."
        )

        continue


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        (
            prediction,
            confidence,
            prob_0,
            prob_1,
            graph
        ) = predict_bbb(
            smiles
        )


        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)


        print("\nSMILES:")
        print(smiles)


        print("\nPrediction:")

        if prediction == 1:

            print(
                "1 → BBB CROSSING"
            )

        else:

            print(
                "0 → NON-BBB CROSSING"
            )


        print("\nClass Probabilities:")

        print(
            f"BBB = 0 : "
            f"{prob_0 * 100:.2f}%"
        )

        print(
            f"BBB = 1 : "
            f"{prob_1 * 100:.2f}%"
        )


        print("\nConfidence:")

        print(
            f"{confidence * 100:.2f}%"
        )


        print("\nGraph Information:")

        print(
            "Nodes :",
            graph.x.shape[0]
        )

        print(
            "Edges :",
            graph.edge_index.shape[1]
        )

        print(
            "Node features :",
            tuple(graph.x.shape)
        )

        print(
            "Edge features :",
            tuple(graph.edge_attr.shape)
        )


        print("\nModel:")
        print(MODEL_FILE)


        print("\nDevice:")
        print(DEVICE)


        print("=" * 60)


    except Exception as e:

        print("\nERROR:")
        print(e)

        print(
            "\nPlease check that the SMILES is valid."
        )

        continue