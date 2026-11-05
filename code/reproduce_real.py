"""Evaluate fixed application settings using the original data split and fitting code."""
from __future__ import annotations
import release_utils as util
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import run_real_data_experiments as original
from extremilefed.experiments import ALL_METHODS
from extremilefed.real_cli import evaluate_dataset


def evaluate(raw, config, output, methods, identity):
    data=original.full_prepared(raw)
    output=Path(output)
    signature=util.digest(identity)
    with util.output_lock(output):
        protocol=output/'protocol.json'
        if protocol.exists() and util.read(protocol)!=identity:
            raise ValueError('Output identity changed; choose a fresh directory.')
        util.write(protocol,identity)
        combined={}
        for method in methods:
            metrics_path=output/method/'metrics.json'
            prediction_path=output/method/'predictions.npz'
            marker=output/method/'complete.json'
            if marker.exists():
                expected={'identity':signature,'metrics':util.sha(metrics_path),'predictions':util.sha(prediction_path)}
                if util.read(marker)!=expected:raise ValueError('Cached application result changed: '+method)
                metrics=util.read(metrics_path)
            else:
                metrics,predictions=evaluate_dataset(data,config,(method,),split='test')
                original.validate_evaluation(method,metrics,predictions,config,len(data.y_test))
                metrics_path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(prediction_path,**predictions)
                util.write(metrics_path,metrics)
                util.write(marker,{'identity':signature,'metrics':util.sha(metrics_path),'predictions':util.sha(prediction_path)})
            combined[method]=metrics['methods'][method]
            print(method+' complete',flush=True)
        if 'centralized' in combined:
            with np.load(output/'centralized/predictions.npz',allow_pickle=False) as z:central=z['centralized_extremiles']
            for method in methods:
                with np.load(output/method/'predictions.npz',allow_pickle=False) as z:
                    combined[method]['mean_absolute_centralized_extremile_gap']=np.mean(np.abs(z[method+'_extremiles']-central),axis=0).tolist()
        util.write(output/'metrics.json',{'methods':combined,'metadata':data.metadata,'config':asdict(config)})
    return combined


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=('fremtpl2','beijing'))
    parser.add_argument('--data-root',type=Path,default=util.ROOT/'data')
    parser.add_argument('--methods',nargs='+',choices=ALL_METHODS,default=list(ALL_METHODS))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    args.data_root=args.data_root.resolve()
    config,snapshot=util.configuration('real',args.dataset)
    for relative,checksum in snapshot['raw_sha256'].items():
        path=args.data_root/relative
        if not path.exists():
            parser.error('Missing data file: '+str(path)+'; see data/README.md')
        if util.sha(path)!=checksum:parser.error('Raw data checksum differs: '+relative)
    if not args.run:
        print({'dataset':args.dataset,'data_verified':True,'methods':args.methods,'run':False})
        return
    util.configure()
    raw=original.load_raw(args.dataset,args.data_root)
    normalized={str(Path(name).relative_to(args.data_root.resolve())):value for name,value in raw.raw_hashes.items()}
    if normalized!=snapshot['raw_sha256']:raise ValueError('Application source files differ from the published snapshot.')
    identity={'dataset':args.dataset,'raw_sha256':normalized,'configuration':util.digest(snapshot),
              'methods':args.methods,'sources':util.sources()}
    evaluate(raw,config,args.output or util.ROOT/'outputs/real'/args.dataset,args.methods,identity)


if __name__=='__main__':main()
