"""
Morgan Fingerprint Substructure Mapper for RDKit Molecules.
Maps active 2048-bit Morgan Fingerprint features back to their exact atomic environments
and fragment SMILES, enabling structural explainability for TreeSHAP contributions.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple, Optional


class MorganSubstructureMapper:
    """
    Extracts atomic subgraphs and chemical environments corresponding to specific
    Morgan fingerprint bits (radius 2, 2048 bits).
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def get_bit_info(self, mol: Chem.Mol) -> Dict[int, List[Tuple[int, int]]]:
        """
        Computes Morgan fingerprint and returns the bitInfo dictionary.
        Returns dict mapping bit_id -> list of (atom_idx, radius).
        """
        bit_info: Dict[int, List[Tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.n_bits, bitInfo=bit_info
        )
        return bit_info

    def extract_substructure_smiles(
        self, mol: Chem.Mol, bit_id: int, bit_info: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Extracts structural information for a given bit_id present in the molecule.
        Returns a list of dictionaries with atom environment details and fragment SMILES.
        """
        if bit_info is None:
            bit_info = self.get_bit_info(mol)

        if bit_id not in bit_info:
            return []

        occurrences = bit_info[bit_id]
        results = []

        for occurrence_idx, (atom_idx, radius) in enumerate(occurrences):
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_symbol = atom.GetSymbol()
            is_aromatic = atom.GetIsAromatic()

            if radius == 0:
                # Single atom environment
                frag_smiles = atom_symbol.lower() if is_aromatic else atom_symbol
                description = f"Isolated/terminal atom: {atom_symbol} (Idx: {atom_idx})"
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                try:
                    frag_smiles = Chem.MolToSmiles(submol)
                except Exception:
                    frag_smiles = f"[{atom_symbol}]*r{radius}"

                description = (
                    f"Substructure around {atom_symbol} (Idx: {atom_idx}, Radius: {radius})"
                )

            # Heuristic functional group tags
            fg_tag = self._annotate_functional_group(frag_smiles, atom_symbol)

            results.append({
                "bit_id": bit_id,
                "feature_name": f"FP_{bit_id}",
                "occurrence": occurrence_idx + 1,
                "center_atom_idx": atom_idx,
                "center_atom_symbol": atom_symbol,
                "radius": radius,
                "fragment_smiles": frag_smiles,
                "description": description,
                "functional_group": fg_tag,
            })

        return results

    def _annotate_functional_group(self, frag_smiles: str, center_symbol: str) -> str:
        """Assigns an intuitive medicinal-chemistry tag to common fragment motifs."""
        s = frag_smiles.lower()
        if "c(=o)n" in s or "nc(=o)" in s:
            return "Amide Linkage / Peptidic Motif"
        if "c(=o)o" in s:
            return "Ester / Carboxylate Group"
        if "c=o" in s:
            return "Carbonyl / Ketone Functionality"
        if "n(" in s or center_symbol == "N":
            if "c" in s and any(c.islower() for c in s):
                return "Basic / Aromatic Nitrogen Heterocycle"
            return "Aliphatic Amine / Nitrogen Center"
        if "c1ccccc1" in s or "c1ccncc1" in s or (len([c for c in s if c == 'c']) >= 5):
            return "Aromatic Hydrophobic Core"
        if "s" in s:
            return "Sulfur-Containing Moiety"
        if "cl" in s or "f" in s or "br" in s:
            return "Halogenated Substituent"
        if "o" in s:
            return "Ether / Hydroxyl Oxygen"
        return "Carbon Backbone / Hydrocarbon Fragment"
