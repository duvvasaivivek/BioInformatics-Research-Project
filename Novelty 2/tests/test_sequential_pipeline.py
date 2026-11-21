"""
Unit and Regression Test Suite for Novelty 2 (Sequential Sieve & Transfer).
Executable with pytest or unittest.
"""

import os
import sys
import unittest
import torch
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TEST_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.sequential_engine import SequentialCascadedEngine, BENCHMARK_DRUGS
from gnn.sequential_transfer_gnn import (
    GINEEncoderTrunk,
    SequentialTransferGNN,
    load_pretrained_bbb_encoder,
)


class TestSequentialPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SequentialCascadedEngine()
        cls.device = torch.device("cpu")
        cls.models_dir = os.path.join(BASE_DIR, "models")
        cls.bbb_ckpt = os.path.join(cls.models_dir, "gnn_single_bbb_random_encoded.pt")

    def test_keppra_cns_penetrant(self):
        """Levetiracetam must pass Stage 1 and Stage 2, resulting in CNS_PENETRANT."""
        smi = BENCHMARK_DRUGS["Levetiracetam (Keppra)"]["smiles"]
        res = self.engine.evaluate_molecule(smi)
        self.assertTrue(res.stage1_pass)
        self.assertTrue(res.stage2_evaluated)
        self.assertTrue(res.stage2_pass)
        self.assertEqual(res.final_disposition, "CNS_PENETRANT")
        self.assertFalse(res.compute_saved)
        self.assertGreater(res.composite_cns_score, 0.5)

    def test_loperamide_paradoxical_efflux(self):
        """Loperamide must pass Stage 1 (permeant) but fail Stage 2 (caught by P-gp)."""
        smi = BENCHMARK_DRUGS["Loperamide (Imodium)"]["smiles"]
        res = self.engine.evaluate_molecule(smi)
        self.assertTrue(res.stage1_pass)
        self.assertTrue(res.stage2_evaluated)
        self.assertFalse(res.stage2_pass)
        self.assertEqual(res.final_disposition, "PARADOXICAL_EFFLUX")
        self.assertEqual(res.composite_cns_score, 0.0)

    def test_penicillin_g_early_termination(self):
        """Penicillin G must fail Stage 1, completely bypassing Stage 2 (compute saved)."""
        smi = BENCHMARK_DRUGS["Penicillin G"]["smiles"]
        res = self.engine.evaluate_molecule(smi)
        self.assertFalse(res.stage1_pass)
        self.assertFalse(res.stage2_evaluated)
        self.assertIsNone(res.stage2_pass)
        self.assertIsNone(res.stage2_prob_pgp)
        self.assertTrue(res.compute_saved)
        self.assertEqual(res.final_disposition, "PERIPHERAL_LOW_PERMEATION")

    def test_caffeine_and_diazepam(self):
        """Caffeine and Diazepam must both be classified as CNS_PENETRANT."""
        for name in ["Caffeine", "Diazepam (Valium)"]:
            smi = BENCHMARK_DRUGS[name]["smiles"]
            res = self.engine.evaluate_molecule(smi)
            self.assertEqual(res.final_disposition, "CNS_PENETRANT", f"Failed for {name}")

    def test_invalid_smiles_raises(self):
        """Engine must raise ValueError on unparseable SMILES."""
        with self.assertRaises(ValueError):
            self.engine.evaluate_molecule("NOT_A_SMILES_STRING_12345")

    def test_library_attrition_metrics(self):
        """Batch evaluation must accurately track cascade attrition rates."""
        smiles_subset = [
            BENCHMARK_DRUGS["Levetiracetam (Keppra)"]["smiles"],
            BENCHMARK_DRUGS["Loperamide (Imodium)"]["smiles"],
            BENCHMARK_DRUGS["Penicillin G"]["smiles"],
        ]
        report = self.engine.evaluate_library(smiles_subset)
        self.assertEqual(report["n_total"], 3)
        self.assertEqual(report["n_stage1_pass"], 2)
        self.assertEqual(report["n_stage1_rejected"], 1)
        self.assertEqual(report["n_cns_leads"], 1)
        self.assertEqual(report["n_paradoxical_effluxed"], 1)
        self.assertAlmostEqual(report["early_compute_saving_ratio"], 1.0 / 3.0)

    def test_transfer_gnn_trunk_and_freeze(self):
        """Sequential Transfer GNN must extract trunk and respect freeze/unfreeze contracts."""
        trunk = load_pretrained_bbb_encoder(self.bbb_ckpt, device=self.device)
        self.assertEqual(len(trunk.convs), 3)
        self.assertEqual(trunk.hidden_channels, 256)

        model = SequentialTransferGNN(trunk=trunk, readout_dim=32, num_classes=2, freeze_trunk=True)
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        self.assertTrue(not any(p.requires_grad for p in model.trunk.parameters()))

        model.unfreeze_trunk()
        self.assertTrue(all(p.requires_grad for p in model.trunk.parameters()))


if __name__ == "__main__":
    unittest.main()
