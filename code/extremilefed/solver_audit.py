"""Finite-step reporting rules, not a formal selection or convergence certificate."""
import numpy as np

def audit_step(initial_loss, final_loss, initial_vector, final_vector, gradient,
               radius=2., mapping_step=1., objective_rtol=1e-10):
    x0,x,g=[np.asarray(v,dtype=float) for v in (initial_vector,final_vector,gradient)]
    if x0.ndim!=1 or x.shape!=x0.shape or g.shape!=x0.shape:
        raise ValueError('matching parameter/gradient vectors required')
    if mapping_step<=0 or objective_rtol<0 or (radius is not None and radius<=0):
        raise ValueError('invalid diagnostic scales')
    finite=bool(np.isfinite([initial_loss,final_loss]).all() and all(np.isfinite(v).all() for v in (x0,x,g)))
    if not finite:
        return {'classification':'numerical_failure','numerically_valid':False,
                'formal_eligible':False,'theory_certified':False}
    distance=float(np.linalg.norm(x-x0))
    feasible=radius is None or distance<=radius+1e-8*max(1.,radius)
    trial=x-mapping_step*g
    if radius is not None:
        d=trial-x0
        trial=x0+d*min(1.,radius/max(np.linalg.norm(d),1e-30))
    tolerance=objective_rtol*max(1.,abs(initial_loss),abs(final_loss))
    descent=final_loss<initial_loss-tolerance
    nonincrease=final_loss<=initial_loss+tolerance
    classification=('constraint_violation' if not feasible else
                    'objective_increase' if not nonincrease else
                    'finite_step_descent' if descent else 'no_verified_progress')
    return {'classification':classification,'numerically_valid':bool(feasible and nonincrease),
            'objective_decrease':float(initial_loss-final_loss),'distance':distance,
            'raw_gradient_norm':float(np.linalg.norm(g)),
            'projected_gradient_mapping_norm':float(np.linalg.norm(x-trial)/mapping_step),
            'mapping_step':mapping_step,'radius':radius,
            'boundary_active':bool(radius is not None and distance>=radius-1e-8*max(1.,radius)),
            'formal_eligible':False,'theory_certified':False,
            'note':'descent of this surrogate only; neither stationarity nor pooled descent certified'}
