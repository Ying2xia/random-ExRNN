"""Independent full-size, all-method preflight; never generates formal MC cells."""
import argparse, json, hashlib, math
from pathlib import Path
import torch
from .development import TaskStore, exclusive_run, write_json, implementation_hash, Paused
from .experiments import SimulationJob, ALL_METHODS, run_job
from .reproducibility import verify_simulation_selection

def finite_tree(value):
    if isinstance(value,float):
        return math.isfinite(value)
    if isinstance(value,dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return all(finite_tree(v) for v in value)
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--max-new-tasks",type=int,default=1)
    args=parser.parse_args()
    if args.max_new_tasks<1:parser.error("positive task budget required")
    torch.set_num_threads(1)
    payload=json.loads(args.config.read_text())
    scope=payload["selection"]["scope"]
    c=verify_simulation_selection(args.config,**scope,allow_size_pending=True)
    root=Path(__file__).resolve().parents[2]
    out=root/"工作记录与备份"/"正式规模预检_v1"/f"scenario{scope['scenario']}_{scope['error']}"
    manifest={"implementation_hash":implementation_hash(),"config_fingerprint":c.fingerprint(),
        "source_config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scope":scope,"n_train":100000,"n_eval":5000,"clients":20,"methods":ALL_METHODS,
        "data_seed_base":1818000000,"formal_mc":False}
    job=SimulationJob(scope["scenario"],scope["error"],0,100000,5000,20,1818000000)
    records={}
    with exclusive_run(out):
        store=TaskStore(out,manifest,args.max_new_tasks)
        try:
            for method in ALL_METHODS:
                record=store.task(method,lambda:run_job(job,c,(method,)))
                if not finite_tree(record):
                    raise FloatingPointError(f"nonfinite full-size result: {method}")
                if method=="extremilefed":
                    rounds=record["methods"]["ExtremileFed"]["gradient_diagnostics"]
                    if not all(r["finite_step_audit"]["numerically_valid"] for r in rounds):
                        raise FloatingPointError("full-size surrogate numerical validity failure")
                records[method]=record
            proof={"passed":True,"scope":scope,"config_fingerprint":c.fingerprint(),
                "implementation_hash":implementation_hash(),"n_train":100000,"n_eval":5000,
                "clients":20,"methods":list(ALL_METHODS),"data_seed_base":1818000000,
                "method_evidence_sha256":{m:hashlib.sha256((out/"tasks"/(m+".json")).read_bytes()).hexdigest() for m in ALL_METHODS},
                "note":"finite execution and protocol checks, not optimality, dominance, or theoretical equivalence"}
            write_json(out/"readiness.json",proof)
            payload["selection"].update(eligible_for_formal_simulation=True,formal_size_verified=True,
                readiness_evidence=str(out/"readiness.json"),
                readiness_sha256=hashlib.sha256((out/"readiness.json").read_bytes()).hexdigest())
            destination=root/"code"/"configs"/"ready"/f"scenario{scope['scenario']}_{scope['error']}.json"
            if destination.exists() and json.loads(destination.read_text())!=payload:
                raise RuntimeError("refusing to overwrite a different ready configuration")
            write_json(destination,payload)
            verify_simulation_selection(destination,**scope)
            status={"status":"ready_for_formal_simulation","config":str(destination)}
        except Paused:
            status={"status":"paused","completed_methods":list(records),"eligible_for_formal_simulation":False}
        write_json(out/"status.json",status)
        print(json.dumps(status,indent=2))

if __name__=="__main__":main()
