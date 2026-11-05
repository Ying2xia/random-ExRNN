"""Run the fixed one-hidden-layer main study; default prints the plan."""
from __future__ import annotations
import release_utils as util
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from depth_main_formal import run_formal
from extremilefed.experiments import ALL_METHODS
from run_followup_experiments import smoke_config


def build_tasks(settings, replications, smoke=False, methods=ALL_METHODS):
    tasks = []
    for setting in settings:
        config, snapshot = util.configuration('main', setting)
        for rep in range(replications):
            selected = replace(config, ernn=replace(config.ernn,
                seed=config.ernn.seed + snapshot['model_seed_stride'] * rep))
            job = {**snapshot['job_replication_zero'], 'replication': rep}
            if smoke:
                selected = smoke_config(selected)
                selected = replace(selected, rounds=1)
                job.update(n_train=100, n_test=32, clients=4)
            for method in methods:
                tasks.append({'id': f'formal/{setting}/{method}/rep{rep:03d}',
                    'stage':'formal','setting':setting,'rep':rep,'method':method,'depth':1,
                    'config':asdict(selected),'job':job,'smoke':smoke})
    return tasks


def worker(argument):
    task, folder, signature = argument
    util.configure()
    path = Path(folder) / 'records' / (task['id'] + '.json')
    if util.cached(path, signature, task):
        return task['id'] + ' cached'
    result = run_formal(task, Path(folder), {'release_schema':1})
    util.publish(path, result, signature, task)
    return task['id'] + ' complete'


def main():
    settings = sorted(p.stem for p in (util.ROOT/'configs/main').glob('*.json'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', nargs='+', choices=settings)
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument('--replications', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    util.check_positive(parser, workers=args.workers, replications=args.replications)
    if args.replications > 100:
        parser.error('the fixed paper plan has at most 100 replications')
    chosen = args.settings or (['scenario1_normal'] if args.smoke else settings)
    tasks = build_tasks(chosen, 1 if args.smoke else args.replications, args.smoke, args.methods)
    output = args.output or util.ROOT/'outputs'/('smoke_main' if args.smoke else 'main')
    if not args.run:
        print({'task_count':len(tasks),'settings':chosen,'smoke':args.smoke,'output':str(output)})
        return
    util.execute(worker, tasks, output, 'main_smoke' if args.smoke else 'main', args.workers)


if __name__ == '__main__':
    main()
