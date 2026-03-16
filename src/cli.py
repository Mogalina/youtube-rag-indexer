import argparse
import os
import signal
import subprocess
import sys
import time

from utils.logger import setup_logger, get_logger
from utils.config import load_config, PROJECT_ROOT
from utils.cli import get_queue, print_status
from pipeline.runner import Runner


logger = get_logger(__name__)

# Process ID file for the background runner
PID_FILE = PROJECT_ROOT / "logs" / "tubx_runner.pid"


def cmd_add(args, config: dict) -> None:
    """
    Add YouTube URLs to the job queue.

    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logging_config = config.get("logging", {})
    setup_logger(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("log_file"),
        rotation=logging_config.get("rotation", "100 MB"),
        retention=logging_config.get("retention", "30 days"),
    )
    queue = get_queue(config)

    for url in args.urls:
        job_id = queue.enqueue(url.strip())
        if job_id is not None:
            print(f"  Enqueued job {job_id}: {url}")
        else:
            print(f"  Already queued (skipped): {url}")


def cmd_status(args, config: dict) -> None:
    """
    Show the status of all jobs in the queue.

    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logging_config = config.get("logging", {})
    setup_logger(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("log_file"),
        rotation=logging_config.get("rotation", "100 MB"),
        retention=logging_config.get("retention", "30 days"),
    )
    queue = get_queue(config)
    jobs = queue.get_all()
    print_status(jobs)


def cmd_run(args, config: dict) -> None:
    """
    Start the background pipeline runner.

    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    if args.daemon:
        # Check if the runner is already running
        if PID_FILE.exists():
            try:
                # Get the PID from the file
                pid = int(PID_FILE.read_text().strip())
                
                # Check if the process is still running
                os.kill(pid, 0)
                
                print(f"Runner is already up and running: {pid}")
                return
            except (ProcessLookupError, ValueError):
                # Clean up the stale PID file
                PID_FILE.unlink()

        # Launch in background
        print("Starting runner in background...")
        
        # Call ourselves without the --daemon flag
        cmd = [sys.executable, "-m", "cli", "run"]
        
        # Send stdout and stderr go to devnull because they are already logged to file
        with open(os.devnull, 'wb') as devnull:
            process = subprocess.Popen(
                cmd,
                stdout=devnull,
                stderr=devnull,
                cwd=str(PROJECT_ROOT / "src"),
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                start_new_session=True
            )
        
        # Save the process identifier
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(process.pid))

        print(f"Runner started in background: {process.pid}")
        return

    # Normal foreground execution
    logging_config = config.get("logging", {})
    setup_logger(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("log_file"),
        rotation=logging_config.get("rotation", "100 MB"),
        retention=logging_config.get("retention", "30 days"),
    )
    
    # Save current process identifier if running in foreground
    if not args.daemon:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))

    try:
        runner = Runner()
        runner.start()
        runner.wait()
    finally:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()


def cmd_stop(args, config: dict) -> None:
    """Stop the background runner."""
    if not PID_FILE.exists():
        print("No runner is currently running.")
        return

    try:
        # Get the process identifier
        pid = int(PID_FILE.read_text().strip())
        print(f"Stopping runner: {pid}")
        
        # Send termination signal
        os.kill(pid, signal.SIGTERM)
        
        # Wait for the runner to stop
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                print("Runner stopped successfully.")
                if PID_FILE.exists():
                    PID_FILE.unlink()
                return
        
        print("Runner is taking too long to stop. It will finish its current job first.")
    except (ProcessLookupError, ValueError):
        print("Runner was not running. Cleaning up stale process identifier file.")

        # Remove the process identifier file
        if PID_FILE.exists():
            PID_FILE.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube Indexer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add YouTube URLs to the job queue
    add_parser = subparsers.add_parser("add", help="Enqueue YouTube URLs for processing")
    add_parser.add_argument("urls", nargs="+", metavar="URL", help="YouTube video URL(s)")

    # Show job queue status and progress
    status_parser = subparsers.add_parser("status", help="Show job queue status and progress")

    # Start the background pipeline runner
    run_parser = subparsers.add_parser("run", help="Start the background pipeline runner")
    run_parser.add_argument("--daemon", action="store_true", help="Run in background")

    # Stop the background runner
    subparsers.add_parser("stop", help="Stop the background runner")

    # Parse arguments
    args = parser.parse_args()
    config = load_config()

    # Execute the command
    if args.command == "add":
        cmd_add(args, config)
    elif args.command == "status":
        cmd_status(args, config)
    elif args.command == "run":
        cmd_run(args, config)
    elif args.command == "stop":
        cmd_stop(args, config)


if __name__ == "__main__":
    main()
