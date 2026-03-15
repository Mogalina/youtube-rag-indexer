import argparse

from utils.logger import setup_logger, get_logger
from utils.config import load_config
from utils.cli import get_queue, print_status
from pipeline.runner import Runner


logger = get_logger(__name__)


def cmd_add(args, config: dict) -> None:
    """
    Add YouTube URLs to the job queue.

    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    setup_logger(level="INFO", format_type="text")
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
    setup_logger(level="INFO", format_type="text")
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
    setup_logger(level="INFO", format_type="text")
    runner = Runner()
    runner.start()
    runner.wait()   


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube Indexer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add YouTube URLs to the job queue
    add_p = subparsers.add_parser("add", help="Enqueue YouTube URLs for processing")
    add_p.add_argument("urls", nargs="+", metavar="URL", help="YouTube video URL(s)")

    # Show job queue status and progress
    subparsers.add_parser("status", help="Show job queue status and progress")

    # Start the background pipeline runner
    subparsers.add_parser("run", help="Start the background pipeline runner")

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


if __name__ == "__main__":
    main()
