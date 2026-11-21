"""Run with unittest or pytest; exercises silent-failure cases from the audit."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from core.artifacts import clean_structures, read_metadata, save_model_bundle, training_smiles
from core.features import load_scaler_metadata, BBB_DESCRIPTOR_NAMES
from explainability.shap_explainer import DualBarrierShapExplainer
from explainability.counterfactual import MolecularCounterfactualEngine
from gnn.dataset import build_graph_dataset
from gnn.model import MultiTaskGNN, build_model_from_checkpoint
from gnn.splits import assert_split_integrity


class AuditRegressions(unittest.TestCase):
    def test_paired_auc_identical_predictions(self):
        from scripts.compare_gnn_experiments import paired_auc_contrast
        frame = pd.DataFrame(dict(task=['bbb']*4, inchikey=list('abcd'),
            y_true=[0,0,1,1], y_prob=[.1,.3,.7,.9]))
        with tempfile.TemporaryDirectory() as temp:
            for tag in ['single_bbb','multitask']:
                frame.to_csv(Path(temp)/f'gnn_{tag}_random_encoded_test_predictions.csv',index=False)
            with patch('scripts.compare_gnn_experiments.RESULTS_DIR',temp):
                result = paired_auc_contrast('bbb','random','encoded',n_bootstraps=20)
            self.assertEqual(result['paired_delta_ROC_AUC'],0)
            self.assertEqual(result['paired_delta_CI95_low'],0)
            self.assertEqual(result['paired_delta_CI95_high'],0)

    def test_report_escapes_input(self):
        from explainability.report_generator import ScientificReportGenerator
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'report.html'
            ScientificReportGenerator.export_html({'smiles':'<script>alert(1)</script>'},str(path))
            html=path.read_text(encoding='utf-8')
            self.assertNotIn('<script>alert(1)</script>',html)
            self.assertIn('&lt;script&gt;',html)

    def test_screen_rejects_failed_conformer(self):
        from scripts.run_repurposing_screen import process_single_molecule
        with patch('scripts.run_repurposing_screen.calculate_3d_descriptors', return_value=(None,None)):
            result = process_single_molecule((0,'ethanol','CCO','','','','',''))
        self.assertIsNone(result)

    def test_shared_training_loop_smoke(self):
        from gnn.training import train_one_model, TrainConfig, predict_graphs
        corpus = pd.DataFrame(dict(inchikey=['a','b','c','d'],
            canonical_smiles=['CC','CCC','CO','CN'], y_bbb=[0,1,0,1],
            m_bbb=[1]*4, y_pgp=[1,0,1,0], m_pgp=[1]*4))
        graphs, stats = build_graph_dataset(corpus, use_cache=False, verbose=False)
        device = torch.device('cpu')
        model, info = train_one_model(graphs, graphs, corpus, np.arange(4),
            TrainConfig(epochs=1, hidden=8, layers=1, batch_size=4), device,
            stats['num_atom_features'], stats['num_bond_features'])
        self.assertEqual(info['epochs_run'], 1)
        pack = predict_graphs(model, graphs, info['criterion'], device, ['bbb','pgp'])
        self.assertEqual(len(pack['per_task']['bbb']['y_true']), 4)

    def test_cache_invalidates_changed_labels_and_order(self):
        corpus = pd.DataFrame(dict(inchikey=['a','b'], canonical_smiles=['CC','CCC'],
            y_bbb=[0,1], m_bbb=[1,1], y_pgp=[0,1], m_pgp=[1,1]), index=[10,20])
        with tempfile.TemporaryDirectory() as temp, patch('gnn.dataset.CACHE_DIR', temp):
            first, _ = build_graph_dataset(corpus, verbose=False)
            self.assertEqual([g.corpus_row.item() for g in first], [0,1])
            corpus.iloc[0, corpus.columns.get_loc('y_bbb')] = 1
            changed, _ = build_graph_dataset(corpus, verbose=False)
            self.assertEqual(changed[0].y_bbb.item(), 1)
            reordered, _ = build_graph_dataset(corpus.iloc[::-1], verbose=False)
            self.assertEqual(reordered[0].smiles, 'CCC')

    def test_checkpoint_nondefault_readout(self):
        model = MultiTaskGNN(66,14,['bbb'], hidden_channels=16, readout_dim=7)
        restored = build_model_from_checkpoint(dict(architecture=model.architecture_metadata(),
                                                    model_state_dict=model.state_dict()))
        self.assertEqual(restored.readout_dim, 7)
        for a,b in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(a,b))

    def test_duplicate_split_index_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_split_integrity(pd.DataFrame({'inchikey':['a','b','c']}),
                {'train':np.array([0,0]),'val':np.array([1]),'test':np.array([], dtype=int)}, 'random')

    def test_structure_conflicts_and_equivalent_smiles(self):
        frame = pd.DataFrame({'SMILES':['CCO','OCC','CC','CC','bad'], 'Label':[1,1,0,1,0]})
        cleaned = clean_structures(frame)
        self.assertEqual(cleaned.SMILES.tolist(), ['CCO'])

    def test_missing_scaler_does_not_use_defaults(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(FileNotFoundError):
            load_scaler_metadata(str(Path(temp)/'bbb_scaler_metadata.json'), [0]*6, [1]*6)

    def test_model_bundle_and_interrupted_save(self):
        model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
        model.fit(np.array([[0],[1],[2],[3]]), [0,0,1,1])
        metadata = dict(features=BBB_DESCRIPTOR_NAMES, means=[0]*6, stds=[1]*6)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bbb_xgboost.json'
            save_model_bundle(model, path, metadata)
            before = path.read_bytes()
            self.assertEqual(read_metadata(Path(temp)/'bbb_scaler_metadata.json'), metadata)
            with patch.object(model, 'save_model', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    save_model_bundle(model, path, dict(metadata, means=[3]*6))
            self.assertEqual(path.read_bytes(), before)

    def test_exact_shap_fallback(self):
        model = xgb.XGBClassifier()
        model.load_model(BASE/'models'/'bbb_xgboost.json')
        from core.features import extract_features_bbb, prepare_dataframe
        features, descriptors, mol = extract_features_bbb('CCO')
        means, stds = load_scaler_metadata(str(BASE/'models'/'bbb_scaler_metadata.json'))
        frame = prepare_dataframe(*features, means, stds)
        with patch('explainability.shap_explainer.SHAP_AVAILABLE', False):
            explainer = DualBarrierShapExplainer(model, model)
        result = explainer.explain_prediction('BBB', frame, mol=mol, raw_descriptors=descriptors)
        self.assertTrue(result['is_tree_shap'])
        self.assertAlmostEqual(result['model_margin'], float(model.predict(frame, output_margin=True)[0]), places=4)

    def test_counterfactual_rejects_other_barrier_regression(self):
        engine = MolecularCounterfactualEngine(None,None,None,None,None,None)
        fake = dict(bbb_pred=0,pgp_pred=0,bbb_prob=.1,pgp_prob=.1,descriptors={})
        with patch.object(engine, '_evaluate_candidate', return_value=fake):
            self.assertEqual(engine.find_counterfactuals('CCO',1,1,min_similarity=0), [])

    def test_legacy_training_references_verified(self):
        for task, filename in [('bbb','processed_b3db_data_advanced.csv'),('pgp','processed_pgp_data.csv')]:
            references = training_smiles(BASE/'data',task)
            self.assertGreater(len(references),0)
            self.assertLess(len(references),len(pd.read_csv(BASE/'data'/filename)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
