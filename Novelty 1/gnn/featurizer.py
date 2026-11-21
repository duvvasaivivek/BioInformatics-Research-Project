"""
Canonical SMILES -> Molecular Graph Featurization for the Multi-Task GNN.

This is the single place a SMILES string becomes a torch_geometric Data object.
Training, inference and attribution all import from here.

Two feature modes are provided:

  'legacy'  - the 9 atom / 7 bond raw-ordinal encoding used by the base-paper
              reference implementation. Retained ONLY so the encoded mode can be
              validated against it and so published baselines stay reproducible.
              It is not recommended: atomic number enters as a raw magnitude, so
              chlorine (17) is numerically "17x" carbon (6), and hybridization
              enters as a raw enum ordinal. Nothing is normalised.

  'encoded' - one-hot categorical encoding plus bounded continuous features. This
              is the mode the multi-task model uses. Comparable in spirit to the
              33-feature encoding used by graphB3 (Brief. Bioinform. 2025), which
              reported that one-hot formal charge and chirality helped.

The legacy bond encoding also contains a near-collinear pair: feature 4 is
(bond_type == AROMATIC) and feature 5 is bond.GetIsAromatic(). The encoded mode
drops the redundant one.
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

FEATURE_MODES = ('legacy', 'encoded')

# ----------------------------------------------------------------------------
# Legacy encoding (base-paper parity)
# ----------------------------------------------------------------------------

LEGACY_ATOM_FEATURE_NAMES = [
    'AtomicNum', 'Degree', 'FormalCharge', 'TotalNumHs', 'IsAromatic',
    'IsInRing', 'ExplicitValence', 'ImplicitValence', 'Hybridization',
]

LEGACY_BOND_FEATURE_NAMES = [
    'IsSingle', 'IsDouble', 'IsTriple', 'IsAromaticBondType',
    'GetIsAromatic', 'IsConjugated', 'IsInRing',
]

# ----------------------------------------------------------------------------
# Encoded-mode vocabularies
# ----------------------------------------------------------------------------

# Elements covering essentially all of B3DB and Broccatelli; anything else falls
# into the trailing "other" slot rather than silently colliding with a real element.
ELEMENT_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se',
    'Na', 'K', 'Li', 'Ca', 'Mg', 'Zn', 'Fe', 'As', 'Al', 'H',
]
DEGREE_VOCAB = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_VOCAB = [-2, -1, 0, 1, 2]
NUM_HS_VOCAB = [0, 1, 2, 3, 4]
HYBRIDIZATION_VOCAB = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRALITY_VOCAB = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
RING_SIZE_VOCAB = [3, 4, 5, 6, 7]  # 8+ falls into "other"
BOND_TYPE_VOCAB = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_VOCAB = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def _one_hot(value: Any, vocab: List[Any]) -> List[float]:
    """One-hot with a trailing catch-all slot, so length is always len(vocab) + 1."""
    vec = [0.0] * (len(vocab) + 1)
    try:
        vec[vocab.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def _explicit_valence(atom: Chem.Atom) -> int:
    """
    Explicit valence, tolerant of the RDKit valence API change.

    The reference implementation calls atom.GetExplicitValence(); newer RDKit
    prefers atom.GetValence(Chem.ValenceType.EXPLICIT). Both return the same value,
    so this picks whichever the installed build exposes.
    """
    try:
        return int(atom.GetValence(Chem.ValenceType.EXPLICIT))
    except (AttributeError, TypeError):
        return int(atom.GetExplicitValence())


def _implicit_valence(atom: Chem.Atom) -> int:
    try:
        return int(atom.GetValence(Chem.ValenceType.IMPLICIT))
    except (AttributeError, TypeError):
        return int(atom.GetImplicitValence())


def _legacy_atom_features(atom: Chem.Atom) -> List[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(_explicit_valence(atom)),
        float(_implicit_valence(atom)),
        float(int(atom.GetHybridization())),
    ]


def _legacy_bond_features(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def _encoded_atom_features(atom: Chem.Atom, ring_info: Chem.RingInfo) -> List[float]:
    feats: List[float] = []
    feats += _one_hot(atom.GetSymbol(), ELEMENT_VOCAB)
    feats += _one_hot(atom.GetDegree(), DEGREE_VOCAB)
    feats += _one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_VOCAB)
    feats += _one_hot(atom.GetTotalNumHs(), NUM_HS_VOCAB)
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATION_VOCAB)
    feats += _one_hot(atom.GetChiralTag(), CHIRALITY_VOCAB)

    # Smallest ring this atom belongs to; all-zero plus catch-all when acyclic.
    idx = atom.GetIdx()
    ring_sizes = [len(r) for r in ring_info.AtomRings() if idx in r]
    feats += _one_hot(min(ring_sizes) if ring_sizes else None, RING_SIZE_VOCAB)

    feats += [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetNumRadicalElectrons() > 0),
        # Bounded continuous features. Divisors are fixed constants, not data
        # statistics, so no train/test leakage is possible through this path.
        _explicit_valence(atom) / 8.0,
        _implicit_valence(atom) / 8.0,
        atom.GetMass() / 100.0,
    ]
    return feats


def _encoded_bond_features(bond: Chem.Bond) -> List[float]:
    feats: List[float] = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPE_VOCAB)
    feats += _one_hot(bond.GetStereo(), BOND_STEREO_VOCAB)
    # bond.GetIsAromatic() is deliberately omitted: it is near-perfectly collinear
    # with the AROMATIC slot of the bond-type one-hot above.
    feats += [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    return feats


def feature_dims(mode: str = 'encoded') -> Tuple[int, int]:
    """
    Returns (num_atom_features, num_bond_features) for a mode.

    Computed from a probe molecule rather than hand-counted, so the dimensions can
    never drift out of step with the encoders above. The base-paper reference
    hardcoded these as literals in both the model and the checkpoint metadata,
    which is how metadata starts silently lying after a refactor.
    """
    _validate_mode(mode)
    probe = Chem.MolFromSmiles('C[C@H](N)C(=O)O')  # has chirality and a polar group
    ring_info = probe.GetRingInfo()
    if mode == 'legacy':
        return len(_legacy_atom_features(probe.GetAtomWithIdx(0))), len(LEGACY_BOND_FEATURE_NAMES)
    atom_dim = len(_encoded_atom_features(probe.GetAtomWithIdx(0), ring_info))
    bond_dim = len(_encoded_bond_features(probe.GetBondWithIdx(0)))
    return atom_dim, bond_dim


def _validate_mode(mode: str) -> None:
    if mode not in FEATURE_MODES:
        raise ValueError(f'Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}')


def mol_to_data(
    mol: Chem.Mol,
    mode: str = 'encoded',
    **attrs: Any
) -> Optional[Data]:
    """
    Converts an RDKit Mol into a torch_geometric Data object.

    Edges are duplicated in both directions (GINEConv expects an explicit directed
    edge list). Hydrogens stay implicit, matching the reference implementation.
    Any extra keyword arguments are attached as graph-level attributes, which is
    how the per-task labels and masks ride along.

    Returns None if the molecule has no atoms.
    """
    _validate_mode(mode)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    ring_info = mol.GetRingInfo()
    if mode == 'legacy':
        atom_feats = [_legacy_atom_features(a) for a in mol.GetAtoms()]
    else:
        atom_feats = [_encoded_atom_features(a, ring_info) for a in mol.GetAtoms()]

    x = torch.tensor(atom_feats, dtype=torch.float32)

    _, bond_dim = feature_dims(mode)
    edge_indices: List[List[int]] = []
    edge_feats: List[List[float]] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _legacy_bond_features(bond) if mode == 'legacy' else _encoded_bond_features(bond)
        # Forward then reverse, same feature vector for both directions.
        edge_indices.append([i, j])
        edge_feats.append(bf)
        edge_indices.append([j, i])
        edge_feats.append(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        # Single-atom molecules (e.g. elemental species in B3DB) have no bonds.
        # Empty tensors with the right shape keep GINEConv happy instead of
        # crashing on .t() of an empty list, which the reference implementation does.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_dim), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    for key, value in attrs.items():
        setattr(data, key, value)
    return data


def smiles_to_data(
    smiles: str,
    mode: str = 'encoded',
    **attrs: Any
) -> Optional[Data]:
    """
    Parses a SMILES string and returns its graph, or None if RDKit rejects it.

    This replaces the two separate extractors in the reference implementation,
    which had already drifted apart on the valence API.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return mol_to_data(mol, mode=mode, **attrs)


def describe_featurization(mode: str = 'encoded') -> Dict[str, Any]:
    """Feature-space description for the run log and checkpoint metadata."""
    atom_dim, bond_dim = feature_dims(mode)
    desc: Dict[str, Any] = {
        'mode': mode,
        'num_atom_features': atom_dim,
        'num_bond_features': bond_dim,
    }
    if mode == 'legacy':
        desc['atom_feature_names'] = list(LEGACY_ATOM_FEATURE_NAMES)
        desc['bond_feature_names'] = list(LEGACY_BOND_FEATURE_NAMES)
        desc['encoding'] = 'raw ordinals, unnormalised (base-paper parity only)'
    else:
        desc['atom_blocks'] = {
            'element_onehot': len(ELEMENT_VOCAB) + 1,
            'degree_onehot': len(DEGREE_VOCAB) + 1,
            'formal_charge_onehot': len(FORMAL_CHARGE_VOCAB) + 1,
            'num_hs_onehot': len(NUM_HS_VOCAB) + 1,
            'hybridization_onehot': len(HYBRIDIZATION_VOCAB) + 1,
            'chirality_onehot': len(CHIRALITY_VOCAB) + 1,
            'min_ring_size_onehot': len(RING_SIZE_VOCAB) + 1,
            'binary_flags': 3,
            'bounded_continuous': 3,
        }
        desc['bond_blocks'] = {
            'bond_type_onehot': len(BOND_TYPE_VOCAB) + 1,
            'stereo_onehot': len(BOND_STEREO_VOCAB) + 1,
            'binary_flags': 2,
        }
        desc['encoding'] = 'one-hot categorical + bounded continuous'
    return desc
