import sqlite3
import threading

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger


logger = get_logger(__name__)

# Valid job statuses
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# Processing steps (in order)
STEPS = ["fetching", "summarizing", "embedding", "saving"]

# Create table schema
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    url        TEXT    NOT NULL UNIQUE,
    video_id   TEXT,
    status     TEXT    NOT NULL DEFAULT 'pending',
    step       TEXT,
    error      TEXT,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
)
"""


class JobQueue:
    """
    Thread-safe database job queue.

    Uses Write-Ahead Logging (WAL) journal mode for concurrent read-write access 
    across threads.
    """

    def __init__(self, db_path: str):
        """
        Initialize the job queue.

        Args:
            db_path: Path to the SQLite database file
        """
        self._database_path = Path(db_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connect(self) -> sqlite3.Connection:
        """
        Context manager for database connections.

        Yields a database connection with WAL journal mode and foreign key support.
        Commits on success, rolls back on exception.

        Returns:
            Database connection
        """
        conn = sqlite3.connect(str(self._database_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
        logger.debug(f"Job queue initialized at {self._database_path}")

    def enqueue(
        self, 
        url: str, 
        video_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Idempotently add a URL to the queue.

        Args:
            url: YouTube URL to enqueue
            video_id: Optional video identifier (if already known)

        Returns:
            Job identifier if newly inserted, None if already existed
        """
        now = datetime.utcnow().isoformat()

        with self._lock, self._connect() as conn:
            # Check if URL already exists
            existing = conn.execute(
                "SELECT id FROM jobs WHERE url = ?", 
                (url,)
            ).fetchone()
            
            # If URL already exists, return None    
            if existing:
                logger.debug(f"URL {url} already queued for job {existing['id']}")
                return None
            
            # Insert new job
            cursor = conn.execute(
                """
                INSERT INTO jobs (
                    url, 
                    video_id, 
                    status, 
                    created_at, 
                    updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (url, video_id, STATUS_PENDING, now, now),
            )

            job_id = cursor.lastrowid
            logger.info(f"Enqueued job {job_id}: {url}")
            
            return job_id

    def claim_next(self) -> Optional[dict]:
        """
        Atomically claim the next pending job, marking it as processing.

        Returns:
            Job dictionary or None if queue is empty
        """
        now = datetime.utcnow().isoformat()

        with self._lock, self._connect() as conn:
            # Select the next pending job
            row = conn.execute(
                """
                SELECT * FROM jobs 
                WHERE status = ? 
                ORDER BY created_at 
                LIMIT 1
                """,
                (STATUS_PENDING,),
            ).fetchone()
            
            # If no pending jobs, return None
            if not row:
                return None
            
            # Update job to processing and set first step
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, step = ?, updated_at = ? 
                WHERE id = ?
                """,
                (STATUS_PROCESSING, STEPS[0], now, row["id"]),
            )
            
            return dict(row)

    def update_job(
        self, 
        job_id: int, 
        status: str, 
        step: Optional[str] = None, 
        error: Optional[str] = None
    ) -> None:
        """
        Update job status and current processing step.

        Args:
            job_id: Job identifier
            status: New status (pending, processing, done, failed)
            step: Optional current processing step
            error: Optional error message if job failed
        """
        now = datetime.utcnow().isoformat()
        
        with self._lock, self._connect() as conn:
            # Update job status and current processing step
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, step = ?, error = ?, updated_at = ? 
                WHERE id = ?
                """,
                (status, step, error, now, job_id),
            )

    def set_video_id(self, job_id: int, video_id: str) -> None:
        """
        Store the resolved video identifier once it's known.

        Args:
            job_id: Job identifier
            video_id: Video identifier
        """
        now = datetime.utcnow().isoformat()
        
        with self._lock, self._connect() as conn:
            # Update job with resolved video identifier
            conn.execute(
                """
                UPDATE jobs 
                SET video_id = ?, updated_at = ? 
                WHERE id = ?
                """,
                (video_id, now, job_id),
            )

    def recover_stale(self) -> int:
        """
        Reset any jobs stuck in 'processing' back to 'pending'.
        Call this on startup to handle crash recovery.

        Returns:
            Number of jobs recovered
        """
        now = datetime.utcnow().isoformat()
        
        with self._lock, self._connect() as conn:
            # Reset stale processing jobs back to pending
            cursor = conn.execute(
                """
                UPDATE jobs 
                SET status = ?, step = NULL, error = NULL, updated_at = ? 
                WHERE status = ?
                """,
                (STATUS_PENDING, now, STATUS_PROCESSING),
            )
            count = cursor.rowcount
        
        if count:
            logger.warning(f"Recovered {count} stale job(s) back to pending")
        
        return count

    def get_all(self) -> list[dict]:
        """
        Return all jobs ordered by creation time.

        Returns:
            List of job dictionaries
        """
        with self._connect() as conn:
            # Get all jobs ordered by creation time
            rows = conn.execute(
                """
                SELECT * FROM jobs 
                ORDER BY created_at ASC
                """
            ).fetchall()

            # Convert rows to list of dictionaries
            return [dict(r) for r in rows]

    def pending_count(self) -> int:
        """
        Return number of pending jobs.

        Returns:
            Number of pending jobs
        """
        with self._connect() as conn:
            # Get count of pending jobs
            row = conn.execute(
                """
                SELECT COUNT(*) FROM jobs 
                WHERE status = ?
                """,
                (STATUS_PENDING,)
            ).fetchone()
            
            # Return count of pending jobs
            return row[0]
