"""This module contains functions for running tasks in parallel."""

# Standard libraries
import logging
import multiprocessing as mp
import signal
import time
from typing import Callable, Sequence, TypeVar

# Internal libraries
from ring_torque.progress import CustomProgress


log = logging.getLogger(__name__)


T = TypeVar("T")


def init_pool():
    """Replaces the pool's child processes' interrupt handler.
    This is to properly handle keyboard interrupts in the main process instead of
    each child process handling it individually.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_tasks_with_pool(
    num_processes: int,
    job: Callable[[T], None],
    tasks: Sequence[T],
    description: str,
):
    """Process the given tasks with a multiprocessing pool.

    Parameters
    ----------
    num_processes : int
        The number of processes to use.
    job : Callable[[T], None]
        The function to be called on each task. It must take a single argument of type `T` and return nothing.
    tasks : Sequence[T]
        A list of tasks to be completed by the pool. Each task must be of type `T`.
    description : str
        The description of the work to be done.
    """
    # Return early if there are no tasks
    if len(tasks) == 0:
        log.info("No work to do!")
        return

    # Limit the number of processes to the number of logical cores and the number of tasks
    actual_num_processes = min(num_processes, mp.cpu_count(), len(tasks))
    if actual_num_processes != num_processes:
        log.warning(
            f"Limiting number of processes to {actual_num_processes} (requested {num_processes})"
        )

    if actual_num_processes == 1:
        log.info("As the number of processes is 1, running synchronously")
        # Synchronous version
        for task in tasks:
            job(task)
        return
    else:
        log.info(
            f"Initialising pool with {actual_num_processes}/{mp.cpu_count()} (processes/logical cores)"
        )
        with mp.Pool(actual_num_processes, initializer=init_pool) as process_pool:
            try:
                # TODO: Create proper progress bars that take into account progress made in each job
                with CustomProgress() as progress:
                    progress_task = progress.add_task(description, total=len(tasks))
                    results = []
                    for task in tasks:
                        result = process_pool.apply_async(
                            job,
                            args=(task,),
                            callback=lambda _: progress.advance(
                                progress_task, advance=1
                            ),
                        )
                        results.append(result)
                    while True:
                        ready = True
                        for result in results:
                            ready &= result.ready()
                        if ready:
                            break
                        time.sleep(1)
            except KeyboardInterrupt:
                log.info("Forceful exit! Cleaning up the processing pool...")
                process_pool.terminate()
                process_pool.join()
                log.info("Cleanup complete.")
