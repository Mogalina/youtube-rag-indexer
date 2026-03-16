import signal
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor, Future

from utils.logger import get_logger, setup_logger
from utils.config import load_config
from pipeline.queue import JobQueue
from pipeline.summarizer import Summarizer
from pipeline.embedder import Embedder
from pipeline.worker import process_job


logger = get_logger(__name__)


class Runner:
    """
    Background pipeline runner.

    This class is responsible for running the pipeline in the background.
    It continuously polls the queue and dispatches jobs to the thread pool.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the runner.

        Args:
            config_path: Path to the configuration file
        """
        self.config = load_config(config_path)
        pipeline_config = self.config["pipeline"]

        # Runner configuration
        self._num_workers: int = pipeline_config["num_workers"]
        self._poll_interval: int = pipeline_config["poll_interval_sec"]
        self._queue = JobQueue(pipeline_config["queue_db"])

        # Thread pool for worker threads
        self._executor: ThreadPoolExecutor | None = None
        self._active_futures: dict[int, Future] = {}
        self._futures_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

        # Load models once and share across worker threads
        logger.info("Loading summarizer model")
        self._summarizer = Summarizer(config_path)
        logger.info("Loading embedder model")
        self._embedder = Embedder(config_path)
        logger.info("Models loaded successfully")

    def start(self) -> None:
        """Start the background polling thread and thread pool."""
        stale = self._queue.recover_stale()
        if stale:
            logger.info(f"Recovered {stale} stale job(s)")

        # Create thread pool for worker threads
        self._executor = ThreadPoolExecutor(
            max_workers=self._num_workers,
            thread_name_prefix="pipeline"
        )

        # Create polling thread
        self._poll_thread = threading.Thread(
            target=self._poll_loop, 
            daemon=True, 
            name="queue-poller"
        )
        self._poll_thread.start()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(f"Runner started with {self._num_workers} worker(s)")

    def wait(self) -> None:
        """Block until the runner is stopped."""
        if self._poll_thread:
            self._poll_thread.join()

    def stop(self) -> None:
        """Gracefully stop the runner after active jobs finish."""
        logger.info("Shutting down and waiting for active jobs to finish")

        # Set stop event
        self._stop_event.set()

        # Wait for all active jobs to finish
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info("All jobs finished")

    def _handle_signal(
        self, 
        signum: int, 
        frame: FrameType
    ) -> None:
        """
        Handle signals for graceful shutdown.
        
        Args:
            signum: Signal number
            frame: Frame object
        """
        logger.info(f"\nSignal {signum} received")
        self.stop()
        sys.exit(0)

    def _poll_loop(self) -> None:
        """
        Continuously claim and submit pending jobs.
        
        This method runs in a background thread and continuously polls the queue
        for new jobs. When a job is found, it is submitted to the thread pool
        for processing.
        """
        while not self._stop_event.is_set():
            # Reap completed jobs
            self._reap_completed()

            # Don't submit more jobs than the pool can handle
            with self._futures_lock:
                active = len(self._active_futures)

            # Submit new jobs if there are available slots
            if active < self._num_workers:
                job = self._queue.claim_next()
                if job:
                    self._submit(job)
                else:
                    time.sleep(self._poll_interval)
            else:
                time.sleep(1)

    def _submit(self, job: dict) -> None:
        """
        Submit a job to the thread pool.
        
        Args:
            job: Job to submit
        """
        job_id = job["id"]

        # Submit job to thread pool
        future = self._executor.submit(
            process_job,
            job,
            self._queue,
            self._summarizer,
            self._embedder,
        )

        # Add future to active map
        with self._futures_lock:
            self._active_futures[job_id] = future
        logger.info(f"[Runner] Submitted job {job_id}: {job['url']}")

    def _reap_completed(self) -> None:
        """
        Remove completed futures from the active map.
        
        This method checks for completed futures and removes them from the
        active map. It also logs any exceptions that occurred during job
        processing.
        """
        with self._futures_lock:
            # Find completed futures
            completed_jobs = []
            for job_id, future in self._active_futures.items():
                if future.done():
                    completed_jobs.append(job_id)
            
            # Remove completed futures and log exceptions
            for job_id in completed_jobs:
                future = self._active_futures.pop(job_id)
                exc = future.exception()
                if exc:
                    logger.error(f"Job {job_id} raised an unhandled exception: {exc}")
