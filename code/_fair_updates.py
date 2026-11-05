"""Numerical update functions used by the shared-initialization comparison."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from extremilefed import training
from extremilefed.config import TAU_LEVELS
from extremilefed.model import ERNNResult, predict_quantiles, predict_extremiles
from extremilefed.metrics import coverage_error, crossing_rate, extremile_errors
from extremilefed.quadrature import gauss_legendre_unit
from extremilefed.federated import partition_indices, sample_size_weighted_average
from extremilefed.preprocessing import fit_federated_standardizer
from extremilefed.simulation import generate_data, true_extremiles
from extremilefed.reproducibility import load_frozen_config

def configuration(setting, seed):
    path = Path(__file__).resolve().parents[1] / 'configs' / 'main' / (setting + '.json')
    config = load_frozen_config(path)
    return replace(config, ernn=replace(config.ernn, seed=seed)), hashlib.sha256(path.read_bytes()).hexdigest()

def vector(model):return training.parameter_vector(model).detach().cpu().numpy().copy()

def init(p,c,v=None):
    model=training.initialize_model(p,c)
    if v is not None:training.load_parameter_vector(model,v)
    return model

@contextmanager
def count_work():
    original=training.data_loss
    work={'gradient_observations':0,'forward_observations':0}
    def wrapped(model,X,*args,**kwargs):
        work['gradient_observations' if torch.is_grad_enabled() else 'forward_observations']+=len(X)
        return original(model,X,*args,**kwargs)
    training.data_loss=wrapped
    try:yield work
    finally:training.data_loss=original

def round_update(method,model,std,c,params,state,t,fc):
    sizes=[len(b) for a,b in std];broadcast=vector(model)
    if method in ('extremilefed','zero'):
        if method=='zero':correction=np.zeros_like(broadcast)
        else:
            gs=[training.data_gradient(model,a,b,c) for a,b in std]
            correction=gs[0]-sample_size_weighted_average(gs,sizes)
        model,h=training.fit_standardized(*std[0],replace(c,patience=fc.surrogate_patience or c.patience,refinement_epochs=0),
                init_vector=broadcast,linear_correction=correction,max_epochs=fc.surrogate_epochs,
                learning_rate=.001 if t<2 else .0003,trust_center=broadcast,trust_radius=fc.trust_radius)
        return model,{'epochs':h['epochs'],'initial_loss':h['initial_loss'],'terminal_loss':h['terminal_loss'],
                      'residual':h['terminal_gradient_norm'],'correction_norm':float(np.linalg.norm(correction))}
    vectors=[];controls=[];hist=[]
    for m,(a,b) in enumerate(std):
        kwargs={'fixed_sgd':True,'linear_correction':state['controls'][m]-state['server']} if method=='scaffold' else {'prox_center':broadcast if params['mu'] else None,'prox_mu':params['mu']}
        local,h=training.fit_standardized(a,b,c,init_vector=broadcast,max_epochs=params['epochs'],learning_rate=params['lr'],**kwargs)
        v=vector(local);vectors.append(v);hist.append(h['epochs'])
        if method=='scaffold':
            steps=max(1,h['epochs']*int(np.ceil(len(b)/c.batch_size)))
            controls.append(state['controls'][m]-state['server']+(broadcast-v)/(steps*params['lr']))
    training.load_parameter_vector(model,sample_size_weighted_average(vectors,sizes))
    if method=='scaffold':state.update(controls=controls,server=sample_size_weighted_average(controls,sizes))
    return model,{'client_epochs':hist}
