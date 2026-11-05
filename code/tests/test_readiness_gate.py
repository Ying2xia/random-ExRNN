import unittest,tempfile,json,hashlib
from pathlib import Path
from dataclasses import asdict
from extremilefed.config import FederatedConfig
from extremilefed.development import implementation_hash,write_json
from extremilefed.reproducibility import verify_simulation_selection
from extremilefed.experiments import ALL_METHODS

class ReadinessGateTests(unittest.TestCase):
 def test_pending_is_not_formally_launchable_and_evidence_is_checked(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);c=FederatedConfig();evidence={}
   for name in ('cv_selection.json','eta_selection.json','round_selection.json','fedavg_selection.json','fedprox_selection.json','scaffold_selection.json','convergence_report.json'):
    write_json(root/name,{'unit_test_fixture':True});evidence[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
   selection={'eligible_for_formal_simulation':False,'smoke':False,'scope':{'scenario':1,'error':'normal'},
    'implementation_hash':implementation_hash(),'evidence_root':str(root),'evidence_files_sha256':evidence,
    'selection_complete':True,'numerical_acceptance':'finite_step','formal_size_verified':False}
   payload={'config':asdict(c),'fingerprint':c.fingerprint(),'selection':selection}
   path=root/'config.json';write_json(path,payload)
   self.assertEqual(verify_simulation_selection(path,1,'normal',allow_size_pending=True),c)
   with self.assertRaises(ValueError):verify_simulation_selection(path,1,'normal')
   tasks={}
   for name in ALL_METHODS:
    f=root/'tasks'/(name+'.json');write_json(f,{'unit_test_fixture':True});tasks[name]=hashlib.sha256(f.read_bytes()).hexdigest()
   proof={'passed':True,'scope':selection['scope'],'config_fingerprint':c.fingerprint(),
    'implementation_hash':implementation_hash(),'n_train':100000,'n_eval':5000,'clients':20,
    'methods':list(ALL_METHODS),'method_evidence_sha256':tasks}
   write_json(root/'proof.json',proof)
   selection.update(eligible_for_formal_simulation=True,formal_size_verified=True,readiness_evidence=str(root/'proof.json'),
    readiness_sha256=hashlib.sha256((root/'proof.json').read_bytes()).hexdigest())
   write_json(path,payload)
   self.assertEqual(verify_simulation_selection(path,1,'normal'),c)
   write_json(root/'tasks/centralized.json',{'tampered':True})
   with self.assertRaises(ValueError):verify_simulation_selection(path,1,'normal')
