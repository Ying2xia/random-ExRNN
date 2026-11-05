"""Run the fixed five-setting supplementary experiment plan."""
from __future__ import annotations
import release_utils as util
import argparse
from dataclasses import asdict
from pathlib import Path
import run_followup_experiments as fitting
from extremilefed.experiments import SimulationJob
from run_real_data_experiments import decode_config


def build_tasks(replications=30, smoke=False, suite='all'):
    configs = {}
    for path in (util.ROOT/'configs/supplementary').glob('*.json'):
        scenario, error = path.stem.split('_',1)
        configs[(int(scenario.replace('scenario','')),error)] = util.configuration('supplementary',path.stem)[0]
    return [task.specification() for task in fitting.build_tasks(configs,replications,smoke)
            if (task.job.scenario,task.job.error) in fitting.REPRESENTATIVE and fitting.task_selected(task,suite)]


def worker(argument):
    spec, folder, signature = argument
    util.configure()
    task = fitting.Task(spec['variant'],spec['family'],SimulationJob(**spec['job']),
                        decode_config(spec['config']),spec['depth'],tuple(spec['methods']),spec['smoke'])
    path = Path(folder)/'records'/(task.key+'.json')
    if util.cached(path,signature,spec):
        return task.key+' cached'
    if task.family == 'reference':
        # Refit the published reference instead of depending on a local old-result cache.
        result = fitting.experiments.run_job(task.job,task.config,task.methods)
        result['followup'] = {'specification':spec,'reference_recomputed':True}
        fitting.original.finite_tree(result)
    else:
        result = fitting.run_task(task)
    util.publish(path,result,signature,spec)
    return task.key+' complete'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('all','depth','sensitivity','budget','quadrature'),default='all')
    parser.add_argument('--replications',type=int,default=30)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--max-tasks',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    util.check_positive(parser,replications=args.replications,workers=args.workers)
    if args.replications>30:parser.error('the fixed supplementary plan has at most 30 replications')
    if args.max_tasks is not None:util.check_positive(parser,max_tasks=args.max_tasks)
    tasks=build_tasks(1 if args.smoke else args.replications,args.smoke,args.suite)
    if args.max_tasks is not None:tasks=tasks[:args.max_tasks]
    output=args.output or util.ROOT/'outputs'/('smoke_supplement' if args.smoke else 'supplement')
    if not args.run:
        print({'task_count':len(tasks),'suite':args.suite,'smoke':args.smoke,'output':str(output)})
        return
    util.execute(worker,tasks,output,'supplement_smoke' if args.smoke else 'supplement',args.workers)


if __name__=='__main__':main()
