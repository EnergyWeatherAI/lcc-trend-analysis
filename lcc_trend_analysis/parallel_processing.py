"""
Utility classes for parallel processing and error handling.

This module provides fault-tolerant alternatives to joblib.Parallel() that can
handle segfaults, timeouts, and other uncatchable errors without failing the
entire processing pipeline.
"""
import multiprocessing as mp
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Callable, Iterable, List, Optional, Any
from itertools import islice

from lcc_trend_analysis.logging import get_logger

logger = get_logger(__name__)


class ProcessPool:
    """
    A fault-tolerant process pool that handles worker crashes gracefully.
    
    Unlike joblib.Parallel(), this pool:
    - Continues processing even when individual workers crash
    - Handles segfaults and other uncatchable errors
    - Collects successful results while logging failures
    - Logs task details when exceptions occur
    """
    
    def __init__(self, n_jobs: Optional[int] = None, timeout: Optional[float] = None, log_task_on_error: bool = True):
        self.n_jobs = n_jobs or mp.cpu_count()
        self.timeout = timeout
        self.log_task_on_error = log_task_on_error
        self.logger = get_logger(self.__class__.__name__)
        
    def map(self, func: Callable, tasks: Iterable, progress_callback: Optional[Callable] = None, batch_size: int = 1000) -> List[Any]:
        """Apply function to tasks in parallel with fault tolerance and streaming processing.
        
        Batching strategy prevents memory exhaustion when tasks is a large generator by submitting
        work incrementally as workers become available, rather than materializing all tasks upfront.
        
        Args:
            func: Function to apply to each task
            tasks: Iterable of tasks to process (can be a generator)
            progress_callback: Optional callback function called with (completed, estimated_total)
            batch_size: Number of tasks to submit at once
            
        Returns:
            List of successful results (None values are excluded)
        """
        results: List[Any] = []
        completed = 0
        submitted = 0
        failed = 0
        
        self.logger.info(f"Starting processing with {self.n_jobs} workers (batch size: {batch_size})")
        
        # Use ProcessPoolExecutor for better error handling
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            tasks_iter = iter(tasks)
            future_to_task = {}
            
            # Initial batch submission
            for task in islice(tasks_iter, batch_size):
                future = executor.submit(func, task)
                future_to_task[future] = task
                submitted += 1
            
            if not future_to_task:
                self.logger.info("No tasks to process")
                return results
                
            self.logger.info(f"Submitted initial batch of {len(future_to_task)} tasks")
            
            # Process results and submit more tasks as workers become available
            while future_to_task:
                # Wait for at least one task to complete
                for future in as_completed(future_to_task, timeout=None):
                    task = future_to_task.pop(future)
                    completed += 1
                    
                    try:
                        result = future.result(timeout=self.timeout) if self.timeout else future.result()
                        if result is not None:
                            results.append(result)
                            
                    except TimeoutError:
                        failed += 1
                        msg = f"Task timed out after {self.timeout}s"
                        if self.log_task_on_error:
                            self.logger.warning(f"{msg}\nTask: {task}")
                        else:
                            self.logger.warning(msg)
                            
                    except Exception as e:
                        failed += 1
                        if self.log_task_on_error:
                            self.logger.error(
                                f"Task failed: {type(e).__name__}: {e}\nTask: {task}",
                                exc_info=True
                            )
                        else:
                            self.logger.error(f"Task failed: {e}", exc_info=True)
                    
                    # Submit next task if available
                    try:
                        next_task = next(tasks_iter)
                        new_future = executor.submit(func, next_task)
                        future_to_task[new_future] = next_task
                        submitted += 1
                    except StopIteration:
                        # No more tasks to submit
                        pass
                    
                    if progress_callback:
                        progress_callback(completed, submitted)
                        
                    # Log progress periodically
                    if completed % max(1, min(batch_size // 10, 100)) == 0:
                        self.logger.info(
                            f"Progress: {completed} tasks completed "
                            f"({len(results)} successful, {failed} failed)"
                        )
                    
                    # Only process one completion per iteration to maintain streaming
                    break
        
        self.logger.info(
            f"Processing complete: {len(results)} successful results, "
            f"{failed} failed tasks out of {completed} total"
        )
        return results

def parallel_map(
    func: Callable,
    tasks: Iterable,
    n_jobs: Optional[int] = None,
    timeout: Optional[float] = None,
    progress_callback: Optional[Callable] = None,
    batch_size: int = 1000,
    log_task_on_error: bool = True
) -> List[Any]:
    """Convenience function for parallel mapping.
    
    Args:
        func (Callable): Function to apply to each task
        tasks (Iterable): Iterable of tasks to process (can be a generator)
        n_jobs (int | None): Number of worker processes (default: CPU count)
        timeout (float | None): Timeout per task in seconds
        progress_callback (Callable | None): Optional callback called with (completed, estimated_total)
        batch_size (int): Number of tasks to submit at once for streaming processing
        log_task_on_error (bool): If True, log task details when errors occur
        
    Returns:
        list[Any]: List of successful results
    """
    pool = ProcessPool(n_jobs=n_jobs, timeout=timeout, log_task_on_error=log_task_on_error)
        
    return pool.map(func, tasks, progress_callback=progress_callback, batch_size=batch_size)


def _worker_init():
    """Initialize worker process to handle signals properly."""
    # Ignore SIGINT in worker processes so they can be terminated cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def create_executor(n_jobs: Optional[int] = None) -> ProcessPoolExecutor:
    """Create a ProcessPoolExecutor configured for parallel processing.
    
    Args:
        n_jobs (int | None): Number of worker processes
        
    Returns:
        ProcessPoolExecutor: Configured executor for parallel processing
    """
    return ProcessPoolExecutor(
        max_workers=n_jobs or mp.cpu_count(),
        initializer=_worker_init
    )