import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
from extremilefed.config import ERNNConfig, FederatedConfig, QuadratureConfig
from extremilefed.training import fit_standardized, parameter_vector, train_ernn
from extremilefed.preprocessing import fit_standardizer
from extremilefed.simulation import generate_data
from extremilefed.federated import train_extremilefed, partition_indices
from extremilefed.baselines import train_host, train_scaffold
from extremilefed.reproducibility import freeze_config, load_frozen_config


class SolverStageTests(unittest.TestCase):
    def setUp(self):
        self.X,self.y=generate_data(31,1,'normal',141)
        self.cfg=ERNNConfig(hidden_units=3,max_epochs=5,min_epochs=5,patience=6,
            batch_size=8,quadrature=QuadratureConfig(6,12),lambda_=.01)

    def test_chunk_accumulation_matches_single_full_batch(self):
        scaler=fit_standardizer(self.X,self.y)
        X,y=scaler.transform_X(self.X),scaler.transform_y(self.y)
        a,ha=fit_standardized(X,y,self.cfg,full_batch=True)
        b,hb=fit_standardized(X,y,replace(self.cfg,batch_size=len(y)),full_batch=True)
        np.testing.assert_allclose(parameter_vector(a).detach().numpy(),parameter_vector(b).detach().numpy(),atol=1e-10,rtol=1e-10)
        self.assertAlmostEqual(ha['terminal_loss'],hb['terminal_loss'],places=12)
        self.assertAlmostEqual(ha['terminal_gradient_norm'],hb['terminal_gradient_norm'],places=11)

    def test_refinement_continues_exact_returned_model(self):
        cfg=replace(self.cfg,refinement_epochs=6)
        fit=train_ernn(self.X,self.y,cfg)
        coarse,fine=fit.history['phases']['coarse'],fit.history['phases']['refinement']
        self.assertAlmostEqual(coarse['terminal_loss'],fine['initial_loss'],places=12)
        self.assertLessEqual(fine['terminal_loss'],fine['initial_loss']+1e-12)
        self.assertEqual(fit.history['epochs'],11)
        self.assertEqual(fit.history['loss'],fine['loss'])
        self.assertTrue(fine['full_batch_updates'])
        self.assertEqual(fit.history['config_fingerprint'],cfg.fingerprint())

    def test_host_baseline_matches_initializer_and_budgets_stay_separate(self):
        cfg=FederatedConfig(ernn=replace(self.cfg,refinement_epochs=6),rounds=1,
            host_epochs=3,host_patience=10,host_learning_rate=.002,
            surrogate_epochs=4,surrogate_patience=10,surrogate_learning_rate=.001)
        parts=[(self.X[i],self.y[i]) for i in partition_indices(len(self.y),3,99)]
        host=train_host(parts,cfg.host_config())
        fed=train_extremilefed(parts,cfg)
        self.assertEqual(fed.history['host']['epochs'],3)
        self.assertEqual(fed.history['rounds'][0]['solver']['epochs'],4)
        self.assertNotIn('phases',fed.history['host'])
        self.assertAlmostEqual(host.history['terminal_loss'],fed.history['host']['terminal_loss'],places=12)
        scaffold=train_scaffold(parts,cfg.ernn,1,1,.001)
        self.assertTrue(all(c['epochs']==1 and not c['full_batch_updates'] for c in scaffold.history['rounds'][0]['clients']))

    def test_solver_choices_survive_freezing(self):
        cfg=FederatedConfig(ernn=replace(self.cfg,refinement_epochs=6),host_epochs=3,
            host_learning_rate=.002,host_patience=10,surrogate_patience=11)
        with tempfile.TemporaryDirectory() as tmp:
            p=freeze_config(cfg,Path(tmp)/'selected.json')
            self.assertEqual(load_frozen_config(p),cfg)
