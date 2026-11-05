"""Finite-step diagnostic pilot; no formal freeze or theoretical certification."""
import sys, argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from dataclasses import replace
import numpy as np
import torch
from extremilefed.config import ERNNConfig
from extremilefed.development import DevelopmentProtocol,isolated_seed,implementation_hash,TaskStore,exclusive_run,write_json,convergence,score_fit,Paused
from extremilefed.federated import partition_indices,sample_size_weighted_average
from extremilefed.preprocessing import fit_federated_standardizer
from extremilefed.simulation import generate_data
from extremilefed.model import ERNNResult, data_loss
from extremilefed.training import fit_standardized,parameter_vector,data_gradient,surrogate_gradient,_tensor_quadrature

from extremilefed.solver_audit import audit_step

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--scenario',type=int,choices=(1,2,3),default=1)
 parser.add_argument('--error',choices=('normal','t3','lognormal','contaminated'),default='normal')
 parser.add_argument('--smoke',action='store_true',help='tiny integration check; not scientific validation')
 parser.add_argument('--seed',type=int,default=913055)
 parser.add_argument('--max-new-tasks',type=int,default=1,help='new initialization chains per invocation; each includes host and 3 rounds')
 args=parser.parse_args()
 if args.seed<0:parser.error('seed must be nonnegative')
 if args.max_new_tasks<1:parser.error('--max-new-tasks must be positive')
 torch.set_num_threads(1)
 roles=['model','curvature_init_1','curvature_init_2']
 folder=ROOT/'工作记录与备份'/'扩大验证_v1'/('smoke' if args.smoke else 'development')/f'scenario{args.scenario}_{args.error}'/f'seed{args.seed}'
 manifest={'source':implementation_hash(),'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'rules_hash':hashlib.sha256((Path(__file__).parent/'solver_audit.py').read_bytes()).hexdigest(),
  'seed':args.seed,'roles':roles,'host':[[800,.01,801],[200,.001,201]],
  'round_lrs':[.001,.001,.0003],'round_epochs':120,'patience':20,'radius':2.,
  'scenario':args.scenario,'error':args.error,'n':100 if args.smoke else 10000,'clients':20,'validation_n':50 if args.smoke else 1000,'smoke':args.smoke,'formal_config_created':False}
 if args.smoke:
  manifest['host']=[[2,.01,3],[2,.001,3]]
  manifest['round_epochs']=2
 p=replace(DevelopmentProtocol(),seed=args.seed)
 seed=lambda role:isolated_seed(p,args.scenario,args.error,role)
 X,y=generate_data(manifest['n'],args.scenario,args.error,seed('pilot_train'))
 V,z=generate_data(manifest['validation_n'],args.scenario,args.error,seed('initialization_audit_validation'))
 parts=[(X[i],y[i]) for i in partition_indices(len(y),20,seed('pilot_partition'))]
 scaler=fit_federated_standardizer(parts)
 std=[(scaler.transform_X(a),scaler.transform_y(b)) for a,b in parts]
 def vector(m):return parameter_vector(m).detach().numpy().copy()
 def pooled_objective(m,cfg):
  nodes,weights=_tensor_quadrature(m,32)
  with torch.no_grad():
   return float(sum(data_loss(m,torch.tensor(a),torch.tensor(b),nodes,weights,cfg.h,cfg.eta)[0]*len(b)/len(y) for a,b in std)+cfg.lambda_*m.l2_first_layer())
 with exclusive_run(folder):
  store=TaskStore(folder,manifest,args.max_new_tasks)
  results={}
  try:
   for role in roles:
    cfg=ERNNConfig(seed=seed(role))
    def fit():
     m,ch=fit_standardized(*std[0],replace(cfg,max_epochs=manifest['host'][0][0],patience=manifest['host'][0][2]))
     m,fh=fit_standardized(*std[0],replace(cfg,learning_rate=.001,max_epochs=manifest['host'][1][0],min_epochs=manifest['host'][1][0],patience=manifest['host'][1][2]),init_vector=vector(m))
     assert abs(ch['terminal_loss']-fh['initial_loss'])<1e-12
     host={'check':convergence(fh,p),'validation':score_fit(ERNNResult(m,scaler,cfg,fh),V,z)}
     rounds=[]
     for t,lr in enumerate(manifest['round_lrs'],1):
      center=vector(m)
      pooled_before=pooled_objective(m,cfg)
      gradients=[data_gradient(m,a,b,cfg) for a,b in std]
      correction=gradients[0]-sample_size_weighted_average(gradients,[len(b) for a,b in std])
      m,h=fit_standardized(*std[0],replace(cfg,learning_rate=lr,max_epochs=manifest['round_epochs']),init_vector=center,
       linear_correction=correction,trust_center=center,trust_radius=2.)
      raw_residual=surrogate_gradient(m,*std[0],cfg,correction)
      step_audit=audit_step(h['initial_loss'],h['terminal_loss'],center,vector(m),raw_residual)
      rounds.append({'finite_step_audit':step_audit,'pooled_before':pooled_before,'pooled_after':pooled_objective(m,cfg),'round':t,'check':convergence(h,p),'validation':score_fit(ERNNResult(m,scaler,cfg,h),V,z),
       'distance':float(np.linalg.norm(vector(m)-center)),'vector':vector(m).tolist(),'history':h})
     return {'model_seed':cfg.seed,'host':host,'rounds':rounds,'coarse_history':ch,'fine_history':fh,
      'passed':host['check']['passed'] and all(r['check']['passed'] for r in rounds)}
    r=store.task(role,fit)
    results[role]={'passed':r['passed'],'host':r['host'],
     'rounds':[{k:v for k,v in row.items() if k not in ('history','vector')} for row in r['rounds']]}
    print(role,json.dumps(results[role],ensure_ascii=False),flush=True)
   status={'status':'diagnostic_complete',
    'legacy_all_passed':all(r['passed'] for r in results.values()),
    'all_surrogate_steps_numerically_valid':all(row['finite_step_audit']['numerically_valid'] for r in results.values() for row in r['rounds']),
    'theory_certified':False,'completed_initializations':len(results),
    'eligible_for_formal_simulation':False}
  except Paused:
   status={'status':'paused','completed_initializations':len(results),
    'message':'repeat the same command to continue','eligible_for_formal_simulation':False}
  write_json(folder/'summary.json',results)
  write_json(folder/'status.json',status)
  print(json.dumps(status,indent=2))
  print('Results:',folder)

if __name__=='__main__':main()
