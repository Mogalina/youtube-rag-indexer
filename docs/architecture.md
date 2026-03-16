# Architecture

This document describes the internal design of **YouTube Indexer**: its components, data flow, concurrency model, storage layout, and configuration system.

## System Overview

**YouTube Indexer** is structured as a producer-consumer pipeline. A persistent `SQLite` database acts as the shared job queue. The CLI is the producer, enqueuing URLs for processing. The runner is the consumer, dispatching jobs to a thread pool where each worker executes the full transcript-to-vector pipeline.

### CLI (`tubx add`)

The user adds a YouTube URL. The CLI inserts a new job into the **SQLite job queue** with status `pending`.

### SQLite Job Queue

This database stores all jobs and their states: 
- `pending` 
- `processing` 
- `done` 
- `failed`

### Runner (Poll Thread)

A background polling thread continuously checks the queue. When it finds a `pending` job, it **claims it** and sends it for processing.

### Thread Pool Executor

The runner submits the job to a pool of worker threads (`worker-1` … `worker-N`).

The number of workers is defined by `pipeline.num_workers`, allowing multiple videos to be processed in parallel.

### Worker Pipeline (Per video)

Each worker runs the full pipeline:

- **Fetch transcript** – retrieves the video subtitles.
- **Summarize** – compresses the transcript into shorter text chunks.
- **Embed chunks** – converts text chunks into vector embeddings.
- **Save to FAISS** – stores the vectors and metadata in the vector index.

### Completion

When the pipeline finishes, the job is marked `done`. If an error occurs, it is marked `failed`.

### Ask (`tubx ask`)

A synchronous retrieval-augmented generation (RAG) flow:
1. **Question** is passed to the CLI.
2. **Embedder** converts the question to a vector.
3. **Searcher** finds the most relevant transcript chunks in FAISS.
4. **ChatEngine** (`microsoft/phi-2`) generates an answer based on the retrieved context.

## Component Reference

### src/cli.py

Entry point registered as the `tubx` command via `pyproject.toml`. Parses subcommands (`add`, `status`, `run`), loads configuration, and delegates to the appropriate handler. Console output is produced only by the CLI layer; all internal logging is routed exclusively to the log file.

### src/utils/config.py

Loads `config/config.yaml` relative to the project root using `lru_cache`, ensuring the YAML file is parsed at most once per process lifetime regardless of how many modules call `load_config()`.

### src/utils/logger.py

Configures loguru with a single file sink. No output is written to standard error or standard output. The log file rotates at 100 MB, old files are compressed to ZIP, and archives are retained for 30 days. The `enqueue=True` option makes log writes asynchronous and thread-safe.

### src/utils/cli.py

UI helpers for the `status` command. Renders a rich progress bar (job completion fraction) and a formatted table of all jobs using the `rich` library. Status labels are color-coded: dim for pending, yellow for processing, green for done, red for failed.

### src/utils/youtube.py

Thin wrapper around `youtube-transcript-api`. `get_video_id()` extracts the `v` query parameter from a YouTube URL. `get_transcript()` fetches the transcript for the resolved video ID in the requested language and joins the segment objects into a single whitespace-separated string.

### src/pipeline/queue.py

Thread-safe `SQLite` job queue backed by **Write-Ahead Logging** (`WAL`) journal mode, which allows concurrent reads from multiple threads alongside a single writer. All mutating operations acquire a Python-level `threading.Lock` in addition to `SQLite`'s own serialization, preventing race conditions during claim-and-update sequences.

Job states:

| Status        | Meaning                            |
|---------------|------------------------------------|
| `pending`     | Waiting to be claimed by a worker  |
| `processing`  | Currently held by a worker thread  |
| `done`        | Successfully indexed               |
| `failed`      | Encountered an unrecoverable error |

On startup, `recover_stale()` resets any jobs left in `processing` state from a previous crash back to `pending`, providing exactly-once delivery guarantees across restarts.

### src/pipeline/summarizer.py

Wraps `google/flan-t5-small` via **Hugging Face Transformers**. Long transcripts are split into word-count chunks (default 512 words each) before being passed to the model. Each chunk is summarized independently with beam search (`num_beams=4`) and a maximum output of 128 new tokens. The per-chunk summaries are concatenated to form the final summary string. The model is moved to the most capable available device (`CUDA` > `MPS` > `CPU`) at initialization time.

### src/pipeline/embedder.py

Wraps `google/embedding-gemma-300m` via **Hugging Face Transformers**. Texts are processed in configurable batches (default 32). For each batch, token embeddings from the final hidden state are mean-pooled with attention mask weighting to produce a single sentence vector. The resulting vectors are L2-normalized before being returned as a float32 NumPy array, making them compatible with FAISS `IndexFlatIP` interpreted as cosine similarity.

### src/pipeline/worker.py

Implements the full per-job processing function `process_job()`. Updates the job's `step` field in the queue at each stage so `tubx status` accurately reflects in-progress state. Also provides `_chunk_text()` for splitting summaries into overlapping chunks before embedding, and `_save_to_faiss()` for atomically appending to the on-disk FAISS index and metadata store.

### src/pipeline/chat.py

Wraps `microsoft/phi-2` (2.7B parameters). It implements a local answering engine that uses an "Instruction" prompt format to ground the model's response in the provided context. It uses `torch.float16` on CUDA devices for performance and memory efficiency.

### src/pipeline/searcher.py

Provides similarity search functionality for the FAISS index. It performs lazy loading of the index and metadata store to minimize startup time for non-search commands.

### src/pipeline/runner.py

Manages the `ThreadPoolExecutor` and a single daemon polling thread. The poll loop calls `claim_next()` to atomically mark a job as `processing`, then submits it to the executor. Completed futures are reaped on every loop iteration. `SIGINT` and `SIGTERM` are intercepted to trigger a graceful shutdown sequence: the stop event is set, the executor waits for all in-flight futures to resolve, and the process exits cleanly.

## Data Flow

1. `input`: YouTube video URL
2. `queue.enqueue()`: INSERT into jobs (status=pending)
3. `queue.claim_next()`: UPDATE status=processing, step=fetching
4. `get_transcript()`: HTTP to YouTube Transcript API
5. `queue.set_video_id()`: UPDATE video_id
6. `summarizer.summarize()`: Local inference (FLAN-T5-small)
7. `embedder.embed()`: Local inference (embedding-gemma-300m)
8. `_save_to_faiss()`: Append to index.faiss + metadata.pkl
9. `queue.update_job(done)`: UPDATE status=done, step=done

**Question Answering Flow:**

1. `input`: User question string
2. `embedder.embed()`: Convert question to vector
3. `searcher.search()`: Query FAISS for top-K results
4. `context`: Format retrieved chunks into a prompt
5. `chat.answer()`: Generate response using Phi-2
6. `output`: Printed answer panel in terminal

## Concurrency Model

The runner spawns exactly one polling thread and one `ThreadPoolExecutor`. Worker threads share the pre-loaded `Summarizer` and `Embedder` instances. Both models are loaded in FP32 and are stateless during inference (no gradient tracking), so sharing them across threads is safe. The SQLite queue serializes all writes through the Python lock and WAL mode, eliminating write contention. **FAISS** index writes in `_save_to_faiss()` are not separately locked; because each job is claimed exclusively by a single worker, parallel writes to the FAISS index are not possible under the current design.

## Configuration Reference

```yaml
summary:
  model_id: string          # HuggingFace model ID for the summarizer
  local_model_dir: path     # Local cache directory
  max_tokens: int           # Words per input chunk
  summary_max_tokens: int   # Max new tokens per output chunk

embedding:
  model_id: string          # HuggingFace model ID for the embedder
  local_model_dir: path     # Local cache directory
  batch_size: int           # Texts per forward pass
  max_tokens: int           # Max tokenizer sequence length

database:
  path: path                # FAISS index directory
  max_size_gb: float        # Hard cap on total index size

chunking:
  max_tokens: int           # Max words per summary chunk for embedding
  overlap_tokens: int       # Overlapping words between consecutive chunks

pipeline:
  num_workers: int          # ThreadPoolExecutor size
  queue_db: path            # SQLite database file
  poll_interval_sec: int    # Idle poll delay in seconds

logging:
  log_file: path            # Output log file path
  rotation: string          # Rotation threshold
  retention: string         # Archive retention
  level: string             # Minimum log level

chat:
  model_id: string          # HuggingFace model ID for the chat model
  local_model_dir: path     # Local cache directory
  max_new_tokens: int       # Generation limit
  top_k: int                # Context chunks to retrieve
```
