import unittest, tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
from extremilefed.config import ERNNConfig,FederatedConfig,QuadratureConfig
from extremilefed.development import DevelopmentProtocol,convergence,TaskStore,run_case,implementation_hash
from extremilefed.federated import train_extremilefed,partition_indices
from extremilefed.baselines import train_host
from extremilefed.simulation import generate_data
from extremilefed.reproducibility import freeze_config,load_frozen_config

class FiniteDevelopmentTests(unittest.TestCase):
 def test_policy_is_explicit_and_does_not_certify_stationarity(self):
  h={'loss':[2.,1.5,1.2,1.1,1.],'initial_loss':3.,'terminal_loss':1.,
     'terminal_gradient_norm':.5,'epochs':5,'best_epoch':5}
  p=DevelopmentProtocol()
  self.assertFalse(convergence(h,p)['passed'])
  r=convergence(h,replace(p,numerical_acceptance='finite_step'))
  self.assertTrue(r['passed'])
  self.assertFalse(r['stationarity_passed'])
  self.assertFalse(r['theory_certified'])
  self.assertFalse(convergence(dict(h,terminal_loss=4.),replace(p,numerical_acceptance='finite_step'))['passed'])
 def test_schedule_roundtrip_and_host_baseline(self):
  c=FederatedConfig(ernn=ERNNConfig(hidden_units=3,max_epochs=3,min_epochs=2,patience=4,
    quadrature=QuadratureConfig(6,12)),rounds=3,host_epochs=3,host_patience=4,
    host_refinement_epochs=2,surrogate_epochs=2,surrogate_learning_rates=(.001,.001,.0003))
  X,y=generate_data(40,1,'normal',392)
  parts=[(X[i],y[i]) for i in partition_indices(40,4,22)]
  f=train_extremilefed(parts,c);h=train_host(parts,c.host_config())
  self.assertEqual(f.history['host']['epochs'],5)
  self.assertAlmostEqual(f.history['host']['terminal_loss'],h.history['terminal_loss'],places=12)
  self.assertEqual(len(f.history['rounds']),3)
  with tempfile.TemporaryDirectory() as tmp:
   self.assertEqual(load_frozen_config(freeze_config(c,Path(tmp)/'config.json')),c)
 def test_finite_smoke_resume(self):
  p=replace(DevelopmentProtocol.small(),numerical_acceptance='finite_step',host_refinement_epochs=2,
    surrogate_learning_rates=(.001,.001,.0003),j_grid=(3,),lambda_grid=(.001,),eta_grid=(0.,),
    local_epochs_grid=(1,),local_lr_grid=(.001,),prox_grid=(.001,))
  with tempfile.TemporaryDirectory() as tmp:
   manifest={'implementation_hash':implementation_hash()}
   a=run_case(TaskStore(tmp,manifest),p,1,'normal')
   b=run_case(TaskStore(tmp,manifest),p,1,'normal')
   self.assertEqual(a,b)
   self.assertFalse(a['eligible_for_formal_simulation'])
