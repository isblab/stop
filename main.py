import multiprocessing as mp
import sys
import os
import time
import warnings
import pickle

from utils import parse_file, Executor, MPLogger
from optimizer import Optimizer


def executor_wrapper(*args):
    exc = Executor(*args)
    exc.handle_processes()


def logger_wrapper(*args):
    lg = MPLogger(*args)
    lg.handle_processes()


def main(argv):
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()
    if len(argv) < 2:
        print("FATAL ERROR: A param file must be provided for the program to continue.")
        quit(1)
    elif len(argv) > 2:
        print("FATAL ERROR: Too many command line arguments passed. Only 1 argument expected (param file path)")
        quit(1)

    filename = argv[1]
    if not os.path.isfile(filename):
        print(f'FATAL ERROR: File-path is incorrect: {filename}')
        quit(1)
    params = parse_file(filename)
    if isinstance(params['analysis_wrapper'], str):
        sys.path += [os.getcwd()]
        exec(params['analysis_wrapper'])
        params['analysis_wrapper'] = eval(params['analysis_wrapper'].split('import ')[-1])
    params['max_wait'] = int(params['max_wait'])

    logger_queue = mp.Queue()
    executor_queue = mp.Queue()
    executor_queue_out = mp.Queue()
    main_death_event = mp.Event()
    main_death_event.clear()
    opt = Optimizer(logger_queue, executor_queue, executor_queue_out, **params)
    path = params['path']
    path = f'{path}/logs'
    if not os.path.isdir(path):
        os.mkdir(path)

    executor_process = mp.Process(target=executor_wrapper, daemon=True,
                                  args=(main_death_event, logger_queue, executor_queue, executor_queue_out, opt.max_np,
                                        path, params['max_wait']))
    executor_process.start()
    logger_process = mp.Process(target=logger_wrapper, daemon=True,
                                args=(f'{path}/main_{int(time.time())}.log', logger_queue, main_death_event, 'w',
                                      params['max_wait']))
    logger_process.start()
    opt.initial_setup()
    cont = True
    terminate = False
    while cont:
        opt.load_block()
        terminate = opt.handle_processes()
        cont = opt.update_state()
        if terminate:
            cont = False
    if not terminate:
        opt.report()
        opt.plots()
        with open(f'{path}/saved_optimizer_param_groups', 'wb') as f:
            pickle.dump(opt.parameter_group_objs, f)
        with open(f'{path}/saved_optimizer_analysis_results', 'wb') as f:
            pickle.dump(opt.analysis_results, f)
    main_death_event.set()
    waiting_time = max(params['max_wait'], 60)
    print(f'Waiting {waiting_time}s for processes to safely quit')
    time.sleep(waiting_time)
    if (executor_process.is_alive()) or (logger_process.is_alive()):
        print('Waiting another 60s for processes to safely quit')
        time.sleep(60)
    if (executor_process.is_alive()) or (logger_process.is_alive()):
        warnings.warn('Processes have not quit yet. Quitting the main process anyway.', UserWarning)
    logger_queue.close()
    executor_queue.close()
    del logger_queue, executor_queue
    print("That's all folks!")


if __name__ == '__main__':
    main(sys.argv)
