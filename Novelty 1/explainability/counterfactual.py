"""
Chemically Valid Molecular Counterfactual Generation Engine.
Applies realistic medicinal chemistry reaction transformations to generate valid analogs,
re-extracts full 2D/3D descriptors, and identifies minimal structural modifications
that flip BBB or P-gp classification while preserving chemical sanity.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import rdChemReactions, DataStructs, rdFingerprintGenerator

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class MolecularCounterfactualEngine:
    """
    Searches for realistic chemical modifications (e.g. bioisosteric substitutions,
    N-methylation, hydroxyl capping, fluorination) to flip BBB or P-gp classifications.
    """

    # Library of realistic medicinal chemistry transformations (SMARTS reaction rules)
    TRANSFORMATIONS = [
        # 1. HBD capping: Alcohol methylation (-OH -> -OMe)
        ("[O;H1:1]>>[O:1]C", "Alcohol O-Methylation (-OH -> -OMe: decreases TPSA & HBD, increases LogP)"),
        # 2. HBD capping: Primary amine mono-methylation (-NH2 -> -NHMe)
        ("[N;H2;!$(N*=O):1]>>[N:1]C", "Primary Amine N-Methylation (-NH2 -> -NHMe: decreases HBD, lowers polarity)"),
        # 3. Esterification / Bioisostere of Carboxylic Acid (-C(=O)OH -> -C(=O)OMe)
        ("[C:1](=O)[O;H1:2]>>[C:1](=O)OC", "Carboxylate Methyl Esterification (mitigates negative charge / high desolvation)"),
        # 4. Bioisosteric replacement: Acid to tetrazole or primary amide
        ("[C:1](=O)[O;H1:2]>>[C:1](=O)N", "Acid to Primary Amide Conversion (-COOH -> -CONH2)"),
        # 5. Aromatic Fluorination (adds lipophilicity with minimal steric penalty)
        ("[c;H1:1]>>[c:1]F", "Aromatic Fluorination (C-H -> C-F: modulates LogP and metabolic stability)"),
        # 6. Demethylation (capping removal for overly lipophilic substrates)
        ("[O:1]C>>[O:1][H]", "Ether O-Demethylation (-OMe -> -OH: reduces lipophilicity for excessive LogP)"),
        # 7. Amine Demethylation
        ("[N;H0;!$(N*=O):1]C>>[N:1][H]", "Tertiary to Secondary Amine Demethylation"),
        # 8. Carbonyl reduction / removal
        ("[C:1](=O)[C:2]>>[C:1][C:2]", "Ketone Deoxygenation (reduces HBA count)"),
    ]

    def __init__(self, bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds):
        self.bbb_model = bbb_model
        self.pgp_model = pgp_model
        self.bbb_means = bbb_means
        self.bbb_stds = bbb_stds
        self.pgp_means = pgp_means
        self.pgp_stds = pgp_stds

    def find_counterfactuals(
        self,
        smiles: str,
        current_bbb_pred: int,
        current_pgp_pred: int,
        target_endpoint: str = "both",  # "BBB", "Pgp", or "both"
        max_counterfactuals: int = 3,
        min_similarity: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Generates chemically valid analogs and evaluates if any flip the prediction
        towards the desired outcome (e.g. BBB+ and/or P-gp-) while maintaining
        high structural similarity (Tanimoto >= min_similarity).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        if target_endpoint not in ('BBB', 'Pgp', 'both'):
            raise ValueError('Unknown target endpoint')
        from core.features import extract_features_pgp
        _, parent_descriptors, _ = extract_features_pgp(smiles)
        if parent_descriptors is None or max_counterfactuals < 1:
            return []
        orig_fp = _MORGAN_GEN.GetFingerprint(mol)
        counterfactuals = []
        seen_smiles = {Chem.MolToSmiles(mol)}

        # Evaluate transformations
        for rxn_smarts, description in self.TRANSFORMATIONS:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
                products_list = rxn.RunReactants((mol,))
            except Exception:
                continue

            for prod_tuple in products_list:
                if not prod_tuple:
                    continue
                cand_mol = prod_tuple[0]
                try:
                    Chem.SanitizeMol(cand_mol)
                    cand_smiles = Chem.MolToSmiles(cand_mol)
                except Exception:
                    continue

                if cand_smiles in seen_smiles or len(cand_smiles) == 0:
                    continue
                seen_smiles.add(cand_smiles)

                # Check structural similarity first
                cand_fp = _MORGAN_GEN.GetFingerprint(cand_mol)
                tanimoto = float(DataStructs.TanimotoSimilarity(orig_fp, cand_fp))
                if tanimoto < min_similarity:
                    continue

                # Predict on the new candidate
                eval_res = self._evaluate_candidate(cand_mol, cand_smiles)
                if eval_res is None:
                    continue

                cand_bbb = eval_res["bbb_pred"]
                cand_pgp = eval_res["pgp_pred"]

                # Check if outcome flipped appropriately
                flipped = False
                objective_note = ""

                if target_endpoint in ["BBB", "both"] and current_bbb_pred == 0 and cand_bbb == 1:
                    flipped = True
                    objective_note = "Successfully achieved BBB permeability (BBB- -> BBB+)."

                if target_endpoint in ["Pgp", "both"] and current_pgp_pred == 1 and cand_pgp == 0:
                    flipped = True
                    objective_note = (
                        f"{objective_note} Successfully evaded P-gp efflux (P-gp+ -> P-gp-)."
                        if objective_note else "Successfully evaded P-gp efflux (P-gp+ -> P-gp-)."
                    )

                if target_endpoint == 'both':
                    flipped = flipped and cand_bbb >= current_bbb_pred and cand_pgp <= current_pgp_pred
                if flipped:
                    counterfactuals.append({
                        "analog_smiles": cand_smiles,
                        "transformation": description,
                        "objective_note": objective_note.strip(),
                        "tanimoto_similarity": round(tanimoto, 4),
                        "bbb_pred": cand_bbb,
                        "bbb_prob": eval_res["bbb_prob"],
                        "pgp_pred": cand_pgp,
                        "pgp_prob": eval_res["pgp_prob"],
                        "delta_descriptors": {k: v - parent_descriptors[k] for k, v in eval_res["descriptors"].items()},
                    })

                if len(counterfactuals) >= max_counterfactuals * 2:
                    break

        # Sort counterfactuals by chemical similarity to parent
        counterfactuals.sort(key=lambda x: x["tanimoto_similarity"], reverse=True)
        return counterfactuals[:max_counterfactuals]

    def _evaluate_candidate(self, mol: Chem.Mol, smiles: str) -> Optional[Dict[str, Any]]:
        """Extracts features and predicts candidate analog outcomes."""
        from core.features import extract_features_bbb, extract_features_pgp, prepare_dataframe
        bbb, _, _ = extract_features_bbb(smiles)
        pgp, descriptors, _ = extract_features_pgp(smiles)
        if bbb is None or pgp is None:
            return None
        bbb_df = prepare_dataframe(*bbb, self.bbb_means, self.bbb_stds)
        pgp_df = prepare_dataframe(*pgp, self.pgp_means, self.pgp_stds, is_pgp=True)
        bbb_prob = float(self.bbb_model.predict_proba(bbb_df)[0][1])
        pgp_prob = float(self.pgp_model.predict_proba(pgp_df)[0][1])
        return {'bbb_pred': int(bbb_prob > .5), 'pgp_pred': int(pgp_prob > .5),
                'bbb_prob': bbb_prob, 'pgp_prob': pgp_prob, 'descriptors': descriptors}
