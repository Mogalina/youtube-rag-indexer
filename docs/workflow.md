# Workflow

This document describes the end-to-end lifecycle of a single indexing job, the 
processing steps performed by each worker, the models involved, their published 
performance benchmarks, and the design decisions that govern each stage.

## Job Lifecycle

Every URL submitted to youtube-rag-indexer passes through seven discrete states:

1. **Enqueued (`pending`)**  
   When a user adds a YouTube URL, a new job is inserted into the queue with the 
   status `pending`.

2. **Claimed (`processing`)**  
   The runner claims the job and assigns it to a worker thread. The job status 
   becomes `processing`, and the pipeline begins.

3. **Fetching step**  
   The worker retrieves the transcript for the YouTube video.

4. **Summarizing step**  
   The transcript is summarized into shorter text segments suitable for downstream 
   processing.

5. **Embedding step**  
   The summarized text chunks are converted into vector embeddings.

6. **Saving step**  
   The embeddings and their associated metadata are appended to the vector index.

7. **Done step (`done`)**  
   If the pipeline finishes successfully, the job status is set to `done` and the 
   step is marked as `done`. If an error occurs at any stage, the job is marked 
   `failed`.

The `step` column in the job queue provides fine-grained visibility into which stage is currently executing. The `tubx status` command reads this column live and 
displays it in the jobs table.

## Stage 1: Transcript Fetching

The worker calls `get_transcript(url)`, which extracts the video ID from the URL
query string and invokes `YouTubeTranscriptApi.list()` to retrieve the available
transcript tracks. The requested language (default: `en`) is selected; if
unavailable, the English track is auto-translated. Individual timed segments are
joined into a single whitespace-separated string.

Transcript fetching is the only stage that requires a network connection. All
subsequent stages run entirely on local hardware.

Typical transcript length for a 30-minute technical talk: 4,000 to 8,000 words.

## Stage 2: Summarization

### Model: google/flan-t5-small

FLAN-T5-small is the 80-million parameter variant of the FLAN-T5 family,
developed by Google. It is an encoder-decoder Transformer fine-tuned with
instruction tuning across 1,836 tasks spanning question answering,
summarization, translation, and classification. The instruction tuning
approach ("Finetuned LANguage Net") substantially improves zero-shot and
few-shot generalization compared to the base T5 model.

**Model specifications:**

| Property             | Value                             |
|----------------------|-----------------------------------|
| Architecture         | T5 (encoder-decoder Transformer)  |
| Parameters           | 80 million                        |
| Training objective   | Instruction tuning on 1,836 tasks |
| Input token limit    | 512 tokens                        |
| Storage footprint    | approximately 300 MB              |
| Minimum RAM (CPU)    | 4 GB                              |
| License              | Apache 2.0                        |

**Published benchmark results:**

| Benchmark                         | Score                            |
|-----------------------------------|----------------------------------|
| MMLU (5-shot)                     | 75.2% (FLAN-T5 family average)   |
| ROUGE-1 (summarization fine-tune) | 27.3 to 55.7 (dataset-dependent) |
| ROUGE-2 (summarization fine-tune) | 9.6 to 45.7 (dataset-dependent)  |
| ROUGE-L (summarization fine-tune) | 22.2 to 52.3 (dataset-dependent) |

FLAN-T5 models perform approximately twice as well as the base T5 models on MMLU, 
BBH (Big-Bench Hard), and MGSM (multilingual math reasoning) benchmarks, as 
reported by Google.

**Inference throughput (CPU):**

| Hardware                 | Throughput                                           |
|--------------------------|------------------------------------------------------|
| AWS ml.c5.large (2 vCPU) | approximately 277 tokens/second at 512-token context |
| Modern laptop CPU        | 100+ sequences/second for short inputs               |

FP16 and INT8 quantization are supported for edge deployments, reducing memory 
by up to 50% with minimal quality degradation.

**Implementation in this project:**

Long transcripts are divided into word-count windows of `summary.max_tokens` words 
(default: 512). The prompt `"summarize: <chunk>"` is passed to the tokenizer, and 
output is generated with beam search (`num_beams=4`) capped at `summary.summary_max_tokens` 
new tokens (default: 128). Per-chunk outputs are concatenated to form the full summary.

## Stage 3: Embedding

### Model: google/embedding-gemma-300m

embedding-gemma-300m is a 300-million parameter dense text embedding model built 
on the Gemma 3 architecture. It is specifically designed for retrieval use cases 
and achieves state-of-the-art results on the Massive Text Embedding Benchmark (MTEB) 
for the under-500-million parameter class.

**Model specifications:**

| Property           | Value                                            |
|--------------------|--------------------------------------------------|
| Architecture       | Gemma 3 (decoder-only Transformer, encoder mode) |
| Parameters         | 300 million                                      |
| Context window     | 2,048 tokens                                     |
| Training languages | 100+                                             |
| Output dimensions  | 1,152 (configurable via MRL truncation)          |
| Memory (quantized) | less than 200 MB                                 |
| License            | Gemma Terms of Use                               |

**Published MTEB benchmark results:**

| Benchmark Suite                       | Mean Score |
|---------------------------------------|------------|
| MTEB Multilingual v2 (250+ languages) | 61.15      |
| MTEB English v2                       | 69.67      |
| MTEB Code v1                          | 68.76      |

The model uses Matryoshka Representation Learning (MRL), which allows the output 
embedding dimension to be truncated to 128, 256, or 512 dimensions with minimal 
quality loss. This is particularly useful for reducing index storage when exact 
recall is less critical.

**Implementation in this project:**

Summary text is split into overlapping chunks of `chunking.max_tokens` words 
(default: 512) with `chunking.overlap_tokens` words of overlap (default: 64). 
Each chunk is independently tokenized and passed through the model. Token embeddings 
from the final hidden state are mean-pooled using the attention mask to handle 
padding, and the resulting sentence vectors are L2-normalized to float32. Batches 
of up to `embedding.batch_size` chunks (default: 32) are processed per forward 
pass to bound peak GPU/CPU memory usage.

L2-normalization ensures that inner-product similarity (FAISS `IndexFlatIP`) is 
numerically equivalent to cosine similarity.

## Stage 4: Vector Storage

### Index: FAISS IndexFlatIP

FAISS (Facebook AI Similarity Search) is a library for efficient dense vector 
similarity search. The `IndexFlatIP` index performs exact exhaustive inner-product 
search.

**Characteristics:**

| Property          | Value                               |
|-------------------|-------------------------------------|
| Search type       | Exact nearest-neighbor (exhaustive) |
| Recall            | 100% (no approximation)             |
| Search complexity | O(N) per query                      |
| Recommended scale | Up to approximately 100,000 vectors |
| Storage format    | Binary flat file (`index.faiss`)    |

For larger collections, the index type can be upgraded to `IndexIVFFlat` 
(cluster-based approximate search) or `IndexIVFPQ` (compressed approximate search) 
with no changes to the surrounding pipeline logic.

**Metadata store:**

A parallel `metadata.pkl` file (Python pickle) stores a list of dicts, one per 
indexed vector:

* `video_id`: The identifier of the video.
* `chunk_index`: The index of the chunk.
* `text`: The text of the chunk.

The position of each entry in the list corresponds to the FAISS internal vector ID,
enabling exact text retrieval from a similarity search result. A size guard 
(`database.max_size_gb`) prevents the index from exceeding the configured disk budget.

## Chunking Strategy

Rather than embedding the full summary as a single vector, the pipeline embeds 
multiple overlapping chunks. This approach improves retrieval granularity: a query 
about a specific topic within a long video is more likely to surface a relevant match 
than if the entire video were compressed into a single vector.

With `max_tokens=512` and `overlap_tokens=64`, the effective stride is 448 words per 
chunk. A 2,000-word summary produces approximately 5 chunks, each overlapping by 
64 words with its neighbors.

## Device Selection

All model objects (`Summarizer`, `Embedder`) perform automatic device selection at 
initialization:

1. **CUDA (NVIDIA GPU)** if `torch.cuda.is_available()`
2. **MPS (Apple Silicon)** if `torch.backends.mps.is_available()`
3. **CPU** otherwise

Models are loaded once in the runner and shared across all worker threads. Inference 
uses `torch.no_grad()` throughout, eliminating gradient bookkeeping overhead.

## Crash Recovery

If the process terminates abnormally while jobs are in `processing` state, the next 
invocation of `tubx run` calls `queue.recover_stale()` before loading any models. 
This resets all `processing` jobs back to `pending`, ensuring they are retried. 
Because transcript fetching and inference are deterministic and side-effect-free (the 
FAISS index is not written until the final step), partial work is safely discarded and 
recomputed.

## Logging

All internal log records are written in JSON format to `logs/tubx.log`. The log 
file rotates when it reaches 100 MB. Rotated files are compressed to ZIP and retained 
for 30 days before deletion. Log writes are enqueued asynchronously via loguru's 
`enqueue=True` flag, preventing I/O from blocking worker threads. No log output 
reaches the terminal.

## Summary of Processing Parameters (Defaults)

| Parameter                | Default   | Config Key                   |
|--------------------------|-----------|------------------------------|
| Summarizer chunk size    | 512 words | `summary.max_tokens`         |
| Summarizer output tokens | 128       | `summary.summary_max_tokens` |
| Beam search width        | 4         | hardcoded in Summarizer      |
| Embedder batch size      | 32 texts  | `embedding.batch_size`       |
| Embedder token limit     | 2048      | hardcoded in Embedder        |
| Embedding chunk size     | 512 words | `chunking.max_tokens`        |
| Embedding chunk overlap  | 64 words  | `chunking.overlap_tokens`    |
| Thread pool workers      | 4         | `pipeline.num_workers`       |
| Poll interval (idle)     | 5 seconds | `pipeline.poll_interval_sec` |
| FAISS index size limit   | 10 GB     | `database.max_size_gb`       |
| Log rotation threshold   | 100 MB    | `logging.rotation`           |
