from pipeline.queue import JobQueue

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
from rich import box


def get_queue(config: dict) -> JobQueue:
    """
    Get the job queue from the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        JobQueue instance
    """
    return JobQueue(config["pipeline"]["queue_db"])


def print_status(jobs: list[dict]) -> None:
    """
    Print the status of all jobs in the queue.

    Args:
        jobs: List of job dictionaries
    """
    console = Console()
    if not jobs:
        console.print("No jobs in queue.", style="yellow")
        return

    counts = {
        "pending": 0,
        "processing": 0,
        "done": 0,
        "failed": 0
    }

    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1

    total = len(jobs)
    done = counts["done"]

    # Job completion progress bar
    console.print()
    with Progress(
        TextColumn("[bold]Progress[/bold]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total} completed)"),
        console=console,
    ) as progress:
        progress.add_task("Progress", total=total, completed=done)

    console.print()
    console.print(
        f"[dim]{counts['pending']} pending[/] • "
        f"[yellow]{counts['processing']} processing[/] • "
        f"[green]{done} done[/] • "
        f"[red]{counts['failed']} failed[/]"
    )
    console.print()

    # Jobs table
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("ID", justify="right")
    table.add_column("URL")
    table.add_column("VIDEO ID")
    table.add_column("STATUS")
    table.add_column("STEP")
    table.add_column("UPDATED")

    status_styles = {
        "pending": "dim",
        "processing": "yellow",
        "done": "green",
        "failed": "red",
    }

    for job in jobs:
        status = job["status"]
        style = status_styles.get(status, "")
        updated = (job["updated_at"] or "")[:19].replace("T", " ")
        
        table.add_row(
            str(job["id"]),
            job["url"],
            job["video_id"] or "-",
            f"[{style}]{status}[/]",
            job["step"] or "-",
            updated
        )

    console.print(table)
    console.print()
