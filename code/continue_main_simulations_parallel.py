"""Resume missing formal replications in bounded, independent CPU processes.

Default is check-only; --run starts training. Each process retains the selected
float64 model, seeds and training budgets. The parent alone publishes results.
"""
from __future__ import annotations

import os

# Set before NumPy/PyTorch imports, including in macOS spawn children.
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for _name in THREAD_VARIABLES:
    os.environ[_name] = '1'

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import multiprocessing as mp
from multiprocessing import resource_tracker
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback
import warnings

import torch
from threadpoolctl import threadpool_limits

import continue_main_simulations as serial
from extremilefed.development import implementation_hash
from extremilefed.experiments import ALL_METHODS, run_job
from extremilefed.validation import validate_formal_coverage

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUTPUT = serial.OUTPUT
RECORDS = ROOT / '工作记录与备份/2026-10-07_并行优化'
load_local_configs = serial.load_local_configs
verify_result = serial.verify_result
atomic_json = serial.atomic_json
sha256 = serial.sha256
jobs = serial.jobs
exclusive_lock = serial.exclusive_lock


def now():
    return datetime.now().astimezone().isoformat()


def configure_threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return threadpool_limits(limits=1)


def _worker_entry(job, config, sender):
    # Ctrl-C belongs to the parent: first drain, second terminate owned children.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    started = time.monotonic()
    observed = []
    envelope = {'ok': False, 'job_key': job.key}
    try:
        limiter = configure_threads()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always', RuntimeWarning)
            result = run_job(job, config, ALL_METHODS)
            verify_result(result, job, config)
        envelope.update(ok=True, result=result)
    except BaseException as error:
        envelope.update(error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
    finally:
        envelope['warnings'] = [
            {'category': w.category.__name__, 'message': str(w.message),
             'filename': w.filename, 'lineno': w.lineno} for w in observed]
        from threadpoolctl import threadpool_info
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        envelope['metadata'] = {
            'pid': os.getpid(), 'seconds': time.monotonic() - started,
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {key: os.environ.get(key) for key in THREAD_VARIABLES},
            'threadpools': threadpool_info(),
            'peak_rss_mib': peak / (1024 ** 2 if sys.platform == 'darwin' else 1024),
            'finished_at': now()}
        try:
            sender.send(envelope)
        finally:
            sender.close()


@dataclass
class Running:
    job: object
    process: object
    connection: object
    started: float


def start_worker(job, config):
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry, args=(job, config, sender), name=job.key)
    # Python 3.9's first tracker launch unblocks SIGINT instead of restoring the
    # caller's mask. Start it before applying the mask inherited by our worker.
    resource_tracker.ensure_running()
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        try:
            # The child inherits this mask through Python/torch import, before
            # _worker_entry can install SIG_IGN. Restore the parent immediately.
            process.start()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
        sender.close()
        raise
    sender.close()
    return Running(job, process, receiver, time.monotonic())


def receive_worker(running):
    try:
        envelope = running.connection.recv()
    except (EOFError, OSError) as error:
        raise RuntimeError(f'{running.job.key}: worker exited without a complete result '
                           f'(exitcode={running.process.exitcode})') from error
    return envelope


def close_worker(running):
    running.connection.close()
    running.process.join(timeout=5)
    if running.process.is_alive():
        running.process.terminate()
        running.process.join(timeout=5)
    if running.process.is_alive():
        running.process.kill()
        running.process.join()


def stop_workers(active):
    # Stop only processes created by this invocation, never another launcher.
    for running in active:
        if running.process.is_alive():
            running.process.terminate()
    for running in active:
        close_worker(running)


class StopController:
    def __init__(self):
        self.requested = False

    def __call__(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt('Second Ctrl-C: terminating unfinished replications.')
        self.requested = True


def scan(configs):
    coverage = validate_formal_coverage(OUTPUT)
    if coverage['unexpected'] or coverage['invalid']:
        raise ValueError(f"Invalid/unexpected existing results: {coverage['invalid'] + coverage['unexpected']}")
    existing, missing = {}, []
    for job in jobs():
        path = OUTPUT / (job.key + '.json')
        if path.exists():
            verify_result(json.loads(path.read_text()), job, configs[(job.scenario, job.error)])
            existing[path.name] = sha256(path)
        else:
            missing.append(job)
    return existing, missing


def persist_result(running, envelope, config, launch, workers, manifest):
    job = running.job
    atomic_json(launch / (job.key + '_warnings.json'), envelope.get('warnings', []))
    if envelope.get('job_key') != job.key:
        raise ValueError(f'Worker returned a different job: {job.key}')
    if not envelope.get('ok'):
        atomic_json(launch / (job.key + '_failed.json'), envelope)
        raise RuntimeError(f"{job.key}: {envelope.get('error_type')}: {envelope.get('error')}")
    result = envelope['result']
    verify_result(result, job, config)
    metadata = envelope['metadata']
    if metadata['torch_threads'] != 1 or metadata['torch_interop_threads'] != 1:
        raise ValueError(f'Worker thread policy changed: {job.key}')
    result['execution_context'] = {
        'mode': 'parallel_replications', 'max_workers': workers, 'threads_per_worker': 1,
        'start_method': 'spawn', 'launch_directory': str(launch),
        'timing_note': 'Method elapsed_seconds measured under concurrent CPU load; '
                       'not an isolated serial timing benchmark.'}
    path = OUTPUT / (job.key + '.json')
    atomic_json(path, result, overwrite=False)
    manifest['completed_jobs'].append(job.key)
    atomic_json(launch / (job.key + '_completed.json'),
                dict(metadata, sha256=sha256(path), runtime_warnings=len(envelope.get('warnings', [])),
                     parent_wall_seconds=time.monotonic() - running.started))
    atomic_json(launch / 'manifest.json', manifest)


def run_parallel(selected, configs, launch, workers, manifest):
    """Bounded scheduling: at most workers jobs in flight, no queued training jobs."""
    stop = StopController()
    previous_handler = signal.signal(signal.SIGINT, stop)
    active, failures = [], []
    next_job = 0
    stop_announced = False
    try:
        while active or (next_job < len(selected) and not stop.requested and not failures):
            if stop.requested and not stop_announced:
                manifest['stop_requested_at'] = now()
                atomic_json(launch / 'manifest.json', manifest)
                print(f'STOP REQUESTED: saving the {len(active)} running replications before exit. '
                      'Press Ctrl-C again to discard unfinished work and stop immediately.', flush=True)
                stop_announced = True
            while len(active) < workers and next_job < len(selected) and not stop.requested and not failures:
                job = selected[next_job]
                if (OUTPUT / (job.key + '.json')).exists():
                    raise RuntimeError(f'Unexpected concurrent writer: {job.key}')
                active.append(start_worker(job, configs[(job.scenario, job.error)]))
                next_job += 1
                print(f'[{next_job}/{len(selected)}] START {job.key}; active={len(active)}/{workers}', flush=True)
            completed = False
            for running in list(active):
                if running.connection.poll() or not running.process.is_alive():
                    completed = True
                    try:
                        envelope = receive_worker(running)
                        persist_result(running, envelope, configs[(running.job.scenario, running.job.error)],
                                       launch, workers, manifest)
                        print(f'SAVED {running.job.key}; this launch={len(manifest["completed_jobs"])}/'
                              f'{len(selected)}; {time.monotonic()-running.started:.1f}s', flush=True)
                    except Exception as error:
                        failure = {'job': running.job.key, 'type': type(error).__name__, 'error': str(error)}
                        failures.append(failure)
                        manifest.setdefault('failures', []).append(failure)
                        atomic_json(launch / 'manifest.json', manifest)
                        print(f'FAILED {running.job.key}: {error}. No new jobs will start; '
                              'remaining running jobs will finish.', file=sys.stderr, flush=True)
                    finally:
                        close_worker(running)
                        active.remove(running)
            if not completed and active:
                time.sleep(0.1)
        if failures:
            raise RuntimeError(f'{len(failures)} worker/result failures; see {launch}')
        return stop.requested
    finally:
        # Ignore further Ctrl-C during bounded cleanup so children cannot be orphaned.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop_workers(active)
        finally:
            signal.signal(signal.SIGINT, previous_handler)


def source_hashes():
    return {'implementation': implementation_hash(), 'parallel_launcher': sha256(__file__),
            'serial_helpers': sha256(serial.__file__)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='train missing formal replications')
    parser.add_argument('--workers', type=int, default=min(10, max(1, (os.cpu_count() or 1) - 2)),
                        help='number of independent single-threaded CPU processes (default: up to 10)')
    parser.add_argument('--max-new-jobs', type=int)
    parser.add_argument('--scenario', type=int, choices=(1, 2, 3))
    parser.add_argument('--error', choices=serial.ERRORS)
    args = parser.parse_args(argv)
    if args.workers <= 0 or (args.max_new_jobs is not None and args.max_new_jobs <= 0):
        parser.error('workers and max-new-jobs must be positive')
    limiter = configure_threads()
    with exclusive_lock(OUTPUT):
        configs, migration = load_local_configs()
        existing, missing = scan(configs)
        selected = [j for j in missing if (args.scenario is None or j.scenario == args.scenario)
                    and (args.error is None or j.error == args.error)]
        if args.max_new_jobs:
            selected = selected[:args.max_new_jobs]
        workers = min(args.workers, len(selected)) if selected else args.workers
        print(f'Validated {len(configs)} configurations; {len(existing)}/1200 completed; '
              f'{len(missing)} missing. Workers={workers}; CPU threads/worker=1; float64.', flush=True)
        if not args.run:
            print(f'CHECK ONLY: {len(selected)} missing replications selected; no training started.', flush=True)
            return
        if not selected:
            print('Nothing to run for the selected scope.', flush=True)
            return
        launch = RECORDS / 'formal_launches' / (datetime.now().astimezone().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
        manifest = {
            'started_at': now(), 'output': str(OUTPUT), 'source_hashes': source_hashes(),
            'environment': serial.environment(), 'migration': migration, 'existing_sha256': existing,
            'execution': {'mode': 'parallel_replications', 'workers': workers, 'threads_per_worker': 1,
                          'start_method': 'spawn', 'worker_reuse': False,
                          'thread_environment': {key: os.environ[key] for key in THREAD_VARIABLES},
                          'logical_cpus': os.cpu_count(), 'machine': platform.machine(),
                          'timing_comparable_to_serial': False},
            'planned_jobs': [j.key for j in selected], 'completed_jobs': [], 'status': 'running'}
        atomic_json(launch / 'manifest.json', manifest)
        try:
            stopped = run_parallel(selected, configs, launch, workers, manifest)
            final_existing, final_missing = scan(configs)
            manifest.update(status=('stopped_after_current_jobs' if stopped else
                                    ('complete' if not final_missing else 'batch_complete')),
                            final_existing_count=len(final_existing), final_missing_count=len(final_missing))
            print(f'DONE: {len(final_existing)}/1200 complete; {len(final_missing)} remaining.', flush=True)
        except BaseException as error:
            manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error_type=type(error).__name__, error=str(error))
            raise
        finally:
            try:
                deviations = [name for name, digest in existing.items()
                              if not (OUTPUT / name).is_file() or sha256(OUTPUT / name) != digest]
                manifest['existing_results_unchanged'] = not deviations
                manifest['existing_result_deviations'] = deviations
                manifest['final_result_file_count'] = len(list(OUTPUT.glob('scenario*_rep*.json')))
                manifest['source_hashes_after'] = source_hashes()
                manifest['provenance_changed'] = manifest['source_hashes_after'] != manifest['source_hashes']
                if deviations or manifest['provenance_changed']:
                    manifest['status'] = 'audit_failed'
            except Exception as error:
                manifest.update(status='audit_failed', final_audit_error=repr(error))
            manifest['finished_at'] = now()
            atomic_json(launch / 'manifest.json', manifest)
        if manifest['status'] == 'audit_failed':
            raise RuntimeError(f'Final provenance audit failed; see {launch / "manifest.json"}')


if __name__ == '__main__':
    main()
