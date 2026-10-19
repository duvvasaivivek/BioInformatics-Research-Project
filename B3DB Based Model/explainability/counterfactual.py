"""
Chemically Valid Molecular Counterfactual Generation Engine.
Applies realistic medicinal chemistry reaction transformations to generate valid analogs,
re-extracts full 2D/3D descriptors, and identifies minimal structural modifications
that flip BBB or P-gp classification while preserving chemical sanity.
"""

from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, DataStructs


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
        max_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generates chemically valid analogs and evaluates if any flip the prediction
        towards the desired outcome (e.g. BBB+ and/or P-gp-).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        orig_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
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

                if flipped:
                    cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, radius=2, nBits=2048)
                    tanimoto = float(DataStructs.TanimotoSimilarity(orig_fp, cand_fp))

                    counterfactuals.append({
                        "analog_smiles": cand_smiles,
                        "transformation": description,
                        "objective_note": objective_note.strip(),
                        "tanimoto_similarity": round(tanimoto, 4),
                        "bbb_pred": cand_bbb,
                        "bbb_prob": eval_res["bbb_prob"],
                        "pgp_pred": cand_pgp,
                        "pgp_prob": eval_res["pgp_prob"],
                        "delta_descriptors": eval_res["delta_descriptors"],
                    })

                if len(counterfactuals) >= max_counterfactuals * 2:
                    break

        # Sort counterfactuals by chemical similarity to parent
        counterfactuals.sort(key=lambda x: x["tanimoto_similarity"], reverse=True)
        return counterfactuals[:max_counterfactuals]

    def _evaluate_candidate(self, mol: Chem.Mol, smiles: str) -> Optional[Dict[str, Any]]:
        """Extracts features and predicts candidate analog outcomes."""
        import pandas as pd
        from rdkit.Chem import Descriptors, Descriptors3D

        # 1. BBB features
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = [int(b) for b in fp.ToBitString()]

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotb = Descriptors.NumRotatableBonds(mol)

        bbb_desc = [mw, logp, tpsa, hbd, hba, rotb]
        bbb_scaled = [(bbb_desc[i] - self.bbb_means[i]) / (self.bbb_stds[i] if self.bbb_stds[i] != 0 else 1.0) for i in range(6)]
        
        bbb_feat_dict = {f'FP_{i}': fp_array[i] for i in range(2048)}
        for i, name in enumerate(['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']):
            bbb_feat_dict[name] = bbb_scaled[i]
        bbb_df = pd.DataFrame([bbb_feat_dict])
        bbb_prob = float(self.bbb_model.predict_proba(bbb_df)[0][1])
        bbb_pred = int(bbb_prob > 0.5)

        # 2. P-gp 3D features
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        res = AllChem.EmbedMolecule(mol_h, params)
        if res != 0:
            res = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=50)
        if res != 0:
            return None

        rg = Descriptors3D.RadiusOfGyration(mol_h)
        asph = Descriptors3D.Asphericity(mol_h)
        ecc = Descriptors3D.Eccentricity(mol_h)
        sph = Descriptors3D.SpherocityIndex(mol_h)
        npr1 = Descriptors3D.NPR1(mol_h)
        npr2 = Descriptors3D.NPR2(mol_h)

        pgp_desc = [mw, logp, tpsa, hbd, hba, rotb, rg, asph, ecc, sph, npr1, npr2]
        pgp_scaled = [(pgp_desc[i] - self.pgp_means[i]) / (self.pgp_stds[i] if self.pgp_stds[i] != 0 else 1.0) for i in range(12)]
        
        pgp_feat_dict = {f'FP_{i}': fp_array[i] for i in range(2048)}
        desc_names_12 = [
            'MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds',
            'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2'
        ]
        for i, name in enumerate(desc_names_12):
            pgp_feat_dict[name] = pgp_scaled[i]
        pgp_df = pd.DataFrame([pgp_feat_dict])
        pgp_prob = float(self.pgp_model.predict_proba(pgp_df)[0][1])
        pgp_pred = int(pgp_prob > 0.5)

        return {
            "bbb_pred": bbb_pred,
            "bbb_prob": round(bbb_prob, 4),
            "pgp_pred": pgp_pred,
            "pgp_prob": round(pgp_prob, 4),
            "delta_descriptors": {
                "MolWt": round(mw, 2),
                "LogP": round(logp, 2),
                "TPSA": round(tpsa, 2),
                "HDonors": hbd,
                "HAcceptors": hba,
                "RotBonds": rotb,
                "RadiusOfGyration": round(rg, 2),
            }
        }
