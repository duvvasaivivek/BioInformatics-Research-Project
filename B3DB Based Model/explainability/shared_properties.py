"""
BBB - P-gp Shared Molecular Properties and Cross-Task Comparative Analysis Module.
Examines how continuous physicochemical properties and discrete substructural motifs
differentially govern passive blood-brain barrier permeation versus active P-gp transporter recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from rdkit import Chem
from scipy.stats import spearmanr

from .substructure_mapper import MorganSubstructureMapper


class SharedPropertiesAnalyzer:
    """
    Performs comparative analysis between BBB permeability and P-gp substrate models
    at both the individual compound level and across global feature distributions.
    """

    SHARED_CONTINUOUS_DESCRIPTORS = [
        'MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds'
    ]

    DESCRIPTOR_DISPLAY_NAMES = {
        'MolWt': 'Molecular Weight (Da)',
        'LogP': 'Lipophilicity (MolLogP)',
        'TPSA': 'Polar Surface Area (TPSA, A^2)',
        'HDonors': 'H-Bond Donors (HBD)',
        'HAcceptors': 'H-Bond Acceptors (HBA)',
        'RotBonds': 'Rotatable Bonds (Flexibility)',
    }

    def __init__(self, substructure_mapper: Optional[MorganSubstructureMapper] = None):
        self.substructure_mapper = substructure_mapper or MorganSubstructureMapper()

    def compare_compound_shared_properties(
        self,
        raw_descriptors: Dict[str, float],
        bbb_shap_result: Dict[str, Any],
        pgp_shap_result: Dict[str, Any],
        mol: Optional[Chem.Mol] = None
    ) -> Dict[str, Any]:
        """
        Builds the compound-level comparison table contrasting BBB vs P-gp feature contributions.
        
        Returns:
            Dictionary containing:
            - continuous_table: List of row dicts for the 6 shared continuous descriptors.
            - shared_substructures: List of active Morgan bits shared across both models.
            - task_divergence_summary: Text synthesis of convergent vs divergent forces.
        """
        bbb_cont = bbb_shap_result.get("continuous_shap", {})
        pgp_cont = pgp_shap_result.get("continuous_shap", {})

        table_rows = []
        for prop in self.SHARED_CONTINUOUS_DESCRIPTORS:
            raw_val = raw_descriptors.get(prop, 0.0)
            bbb_val = bbb_cont.get(prop, 0.0)
            pgp_val = pgp_cont.get(prop, 0.0)

            # Determine task relationship
            # Note: For BBB, positive SHAP favors BBB+ (permeation).
            # For P-gp, positive SHAP favors P-gp+ (efflux/substrate).
            relation, interp = self._interpret_shared_continuous_property(
                prop, raw_val, bbb_val, pgp_val
            )

            table_rows.append({
                "property_code": prop,
                "property_name": self.DESCRIPTOR_DISPLAY_NAMES.get(prop, prop),
                "molecular_value": raw_val,
                "bbb_contribution": round(bbb_val, 4),
                "pgp_contribution": round(pgp_val, 4),
                "task_relationship": relation,
                "scientific_interpretation": interp,
            })

        # Compare shared active substructures (Morgan bits)
        shared_bits = self._find_shared_active_substructures(
            bbb_shap_result, pgp_shap_result, mol
        )

        divergence_summary = self._synthesize_task_divergence(table_rows, shared_bits)

        return {
            "continuous_table": table_rows,
            "shared_substructures": shared_bits,
            "task_divergence_summary": divergence_summary,
        }

    def _interpret_shared_continuous_property(
        self, prop: str, val: float, bbb_shap: float, pgp_shap: float
    ) -> Tuple[str, str]:
        """
        Derives scientific interpretation grounded in the distinct biological nature of
        membrane passive diffusion (BBB) vs ABCB1 cavity binding (P-gp).
        """
        if prop == 'LogP':
            if bbb_shap > 0 and pgp_shap > 0:
                return (
                    "Antagonistic (Divergent)",
                    f"LogP ({val:.2f}) promotes passive membrane partition (BBB+), but simultaneously increases hydrophobic cavity affinity for P-gp efflux pump (P-gp+)."
                )
            elif bbb_shap > 0 and pgp_shap <= 0:
                return (
                    "Synergistic (Optimal)",
                    f"Moderate LogP ({val:.2f}) provides sufficient lipid partition for BBB penetration without triggering high-affinity P-gp efflux binding."
                )
            else:
                return (
                    "Concordant Impairment",
                    f"Low LogP ({val:.2f}) reduces P-gp cavity binding, but significantly hinders passive endothelial permeation."
                )

        elif prop == 'MolWt':
            if bbb_shap < 0 and pgp_shap > 0:
                return (
                    "Compound Barrier Burden",
                    f"Elevated MW ({val:.1f} Da) simultaneously retards passive transcellular diffusion (BBB-) and provides bulk contact points for P-gp binding (P-gp+)."
                )
            elif bbb_shap >= 0 and pgp_shap <= 0:
                return (
                    "Synergistic (Compact)",
                    f"Compact MW ({val:.1f} Da) diffuses readily across lipid membranes and easily evades the large P-gp translocation chamber."
                )
            else:
                return (
                    "Neutral / Moderate Impact",
                    f"Molecular weight ({val:.1f} Da) exerts balanced influence across both endpoints."
                )

        elif prop == 'TPSA':
            if bbb_shap < 0 and pgp_shap < 0:
                return (
                    "Polarity Trade-off",
                    f"TPSA ({val:.1f} A^2) incurs a desolvation penalty opposing BBB entry, but also lowers affinity for the hydrophobic P-gp cleft."
                )
            elif bbb_shap >= 0:
                return (
                    "Permeation Favorable",
                    f"Compact polar area ({val:.1f} A^2 <= 90) permits smooth desolvation upon membrane entry."
                )
            else:
                return (
                    "Substrate Hindrance",
                    f"Polarity ({val:.1f} A^2) reduces passive partition while weakly altering efflux recognition."
                )

        elif prop in ['HDonors', 'HAcceptors']:
            if pgp_shap > 0:
                return (
                    "Seelig Pharmacophore Trigger",
                    f"{prop} ({val}) fits Seelig's spatial hydrogen-bonding motifs, triggering P-gp ATPase activation and efflux."
                )
            else:
                return (
                    "Restricted H-Bonding",
                    f"Low {prop} ({val}) avoids forming electrostatic anchor points with P-gp transmembrane helices."
                )

        elif prop == 'RotBonds':
            if pgp_shap > 0:
                return (
                    "Induced-Fit Risk",
                    f"High rotatable bond count ({val}) provides conformational flexibility to adapt to P-gp's multisite binding chamber."
                )
            else:
                return (
                    "Rigid Framework",
                    f"Low rotatable bond count ({val}) provides rigid structural compactness, resisting induced-fit entrapment."
                )

        return (
            "Task-Specific Contribution",
            f"Property contributes {bbb_shap:+.4f} (BBB) and {pgp_shap:+.4f} (P-gp)."
        )

    def _find_shared_active_substructures(
        self,
        bbb_shap_result: Dict[str, Any],
        pgp_shap_result: Dict[str, Any],
        mol: Optional[Chem.Mol]
    ) -> List[Dict[str, Any]]:
        """
        Identifies active Morgan fingerprint bits that appear in the top attributions
        of BOTH the BBB and P-gp models for this compound.
        """
        bbb_top = bbb_shap_result.get("top_supporting", []) + bbb_shap_result.get("top_opposing", [])
        pgp_top = pgp_shap_result.get("top_supporting", []) + pgp_shap_result.get("top_opposing", [])

        bbb_map = {item["feature"]: item for item in bbb_top if item.get("is_active_bit")}
        pgp_map = {item["feature"]: item for item in pgp_top if item.get("is_active_bit")}

        shared_feature_keys = set(bbb_map.keys()).intersection(set(pgp_map.keys()))
        results = []

        for key in shared_feature_keys:
            b_item = bbb_map[key]
            p_item = pgp_map[key]
            b_shap = b_item["shap_value"]
            p_shap = p_item["shap_value"]

            # Substructure fragment
            substructures = b_item.get("substructures", []) or p_item.get("substructures", [])
            frag_smiles = substructures[0]["fragment_smiles"] if substructures else "Active Bit Subgraph"
            fg_tag = substructures[0]["functional_group"] if substructures else "Pharmacophore Motif"

            # Check if directional impacts agree
            # For BBB: >0 is permeable, <0 is non-permeable
            # For P-gp: >0 is substrate, <0 is evader
            direction_note = (
                "Reinforcing efflux vulnerability (Increases BBB+ & P-gp+)"
                if (b_shap > 0 and p_shap > 0)
                else "Ideal CNS motif (Promotes BBB+ while suppressing P-gp efflux)"
                if (b_shap > 0 and p_shap < 0)
                else "Double negative (Reduces BBB permeation while promoting efflux)"
                if (b_shap < 0 and p_shap > 0)
                else "Concordant barrier resistance (Reduces both BBB and P-gp interaction)"
            )

            results.append({
                "feature": key,
                "fragment_smiles": frag_smiles,
                "functional_group": fg_tag,
                "bbb_shap": round(b_shap, 4),
                "pgp_shap": round(p_shap, 4),
                "direction_note": direction_note,
            })

        return results

    def _synthesize_task_divergence(
        self, continuous_rows: List[Dict], shared_substructures: List[Dict]
    ) -> str:
        """Creates a cohesive pharmacological summary explaining the cross-task forces."""
        divergent_props = [r["property_name"] for r in continuous_rows if "Divergent" in r["task_relationship"]]
        synergistic_props = [r["property_name"] for r in continuous_rows if "Synergistic" in r["task_relationship"]]

        summary_parts = []
        if divergent_props:
            summary_parts.append(
                f"Antagonistic forces observed in {', '.join(divergent_props)}: properties that facilitate passive lipid membrane partition simultaneously promote active ABCB1 recognition."
            )
        if synergistic_props:
            summary_parts.append(
                f"Synergistic CNS optimization in {', '.join(synergistic_props)}: molecular dimensions allow transcellular crossing while evading the active efflux pump."
            )
        if not summary_parts:
            summary_parts.append(
                "Continuous descriptors exhibit independent task-specific weighting without severe biophysical conflict."
            )

        return " ".join(summary_parts)

    @classmethod
    def compute_global_shared_feature_correlation(
        cls,
        bbb_feature_importances: pd.Series,
        pgp_feature_importances: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculates Spearman rank correlation and distribution overlap across shared features.
        """
        shared_cols = list(set(bbb_feature_importances.index).intersection(set(pgp_feature_importances.index)))
        if not shared_cols:
            return {"error": "No overlapping features found between models."}

        b_vals = bbb_feature_importances[shared_cols]
        p_vals = pgp_feature_importances[shared_cols]

        rho, pval = spearmanr(b_vals, p_vals)

        # Separate continuous vs fingerprint correlation
        shared_continuous = [c for c in cls.SHARED_CONTINUOUS_DESCRIPTORS if c in shared_cols]
        if len(shared_continuous) >= 3:
            rho_cont, pval_cont = spearmanr(
                bbb_feature_importances[shared_continuous],
                pgp_feature_importances[shared_continuous]
            )
        else:
            rho_cont, pval_cont = np.nan, np.nan

        return {
            "num_shared_features": len(shared_cols),
            "num_shared_continuous": len(shared_continuous),
            "spearman_rank_correlation_all": round(float(rho), 4) if not np.isnan(rho) else 0.0,
            "spearman_pvalue_all": float(pval) if not np.isnan(pval) else 1.0,
            "spearman_rank_correlation_continuous": round(float(rho_cont), 4) if not np.isnan(rho_cont) else None,
            "spearman_pvalue_continuous": float(pval_cont) if not np.isnan(pval_cont) else None,
        }
