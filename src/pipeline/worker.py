import pickle
import faiss
import numpy as np

from logger import get_logger
from config import load_config, PROJECT_ROOT
from pipeline.queue import JobQueue, STATUS_DONE, STATUS_FAILED
from pipeline.summarizer import Summarizer
from pipeline.embedder import Embedder

from youtube import get_video_id, get_full_transcript


logger = get_logger(__name__)


def _chunk_text(
    text: str,
    max_tokens: int,
    overlap: int,
) -> list[str]:
    """
    Chunk text into non-overlapping segments of `max_tokens` length.

    Args:
        text: Input text to chunk
        max_tokens: Maximum number of tokens per chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of non-overlapping text chunks
    """
    # Split text into words
    words = text.split()

    # Ensure step to avoid empty chunks
    step = max(1, max_tokens - overlap)

    # Chunk text into non-overlapping segments
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_tokens])
        chunks.append(chunk)

    return chunks


def _save_to_faiss(
    video_id: str,
    chunks: list[str],
    vectors: np.ndarray,
    config: dict,
) -> None:
    """
    Append embeddings to the FAISS index and update the metadata store.

    Args:
        video_id: Identifier of the video being processed
        chunks: List of text chunks to embed
        vectors: Array of embedding vectors
        config: Configuration dictionary
    """
    database_config = config["database"]
    index_directory = PROJECT_ROOT / database_config["path"]
    index_directory.mkdir(parents=True, exist_ok=True)

    index_path = index_directory / "index.faiss"
    metadata_path = index_directory / "metadata.pkl"
    max_bytes = int(database_config["max_size_gb"] * 1024 ** 3)

    # Load or create index
    dim = vectors.shape[1]
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        index = faiss.IndexFlatIP(dim)

    # Check size limit
    current_size = sum(f.stat().st_size for f in index_directory.rglob("*") if f.is_file())
    additional = vectors.nbytes + len(chunks) * 256
    if current_size + additional > max_bytes:
        raise RuntimeError(
            f"Adding vectors for '{video_id}' would exceed the "
            f"{database_config['max_size_gb']}GB index limit."
        )

    # Load or create metadata
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            metadata: list[dict] = pickle.load(f)
    else:
        metadata = []

    index.add(vectors)
    for i, chunk in enumerate(chunks):
        metadata.append({
            "video_id": video_id,
            "chunk_index": i,
            "text": chunk
        })

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(
        f"Saved {len(chunks)} vectors for video '{video_id}', "
        f"total vectors in index: {index.ntotal}"
    )


def process_job(
    job: dict,
    queue: JobQueue,
    summarizer: Summarizer,
    embedder: Embedder,
) -> None:
    """
    Process a single job end-to-end.

    Args:
        job: Job dict from the database
        queue: JobQueue instance for status updates
        summarizer: Loaded Summarizer
        embedder: Loaded Embedder
    """
    job_id = job["id"]
    url = job["url"]
    config = load_config()
    chunking_config = config["chunking"]

    try:
        # Fetch transcript
        queue.update_job(job_id, "processing", step="fetching")
        logger.info(f"Job {job_id} fetching transcript: {url}")
        transcript = get_full_transcript(url, language="en")
        video_id = get_video_id(url)
        queue.set_video_id(job_id, video_id)

        # Summarize transcript
        queue.update_job(job_id, "processing", step="summarizing")
        logger.info(f"Job {job_id} summarizing transcript ({len(transcript.split())} words)")
        summary = summarizer.summarize(transcript)

        # Embed summary
        queue.update_job(job_id, "processing", step="embedding")
        chunks = _chunk_text(
            summary,
            chunking_config["max_tokens"],
            chunking_config["overlap_tokens"],
        )
        if not chunks:
            chunks = [summary]
        logger.info(f"Job {job_id} embedding {len(chunks)} chunk(s)")
        vectors = embedder.embed(chunks)

        # Save to FAISS index and metadata
        queue.update_job(job_id, "processing", step="saving")
        logger.info(f"Job {job_id} saving to FAISS index")
        _save_to_faiss(video_id, chunks, vectors, config)

        # Mark as done
        queue.update_job(job_id, STATUS_DONE, step="saving")
        logger.info(f"Job {job_id} done: {url}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        queue.update_job(job_id, STATUS_FAILED, error=str(e))
