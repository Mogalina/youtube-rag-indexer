# YouTube Indexer

## Overview

YouTube Indexer is a locally executed pipeline for transforming YouTube video content into a searchable semantic knowledge base. Given one or more YouTube video URLs, the system fetches transcripts, summarizes them using a compact instruction-tuned language model, embeds the summaries into dense vector representations, and stores them in a persistent **Facebook AI Similarity Search** (`FAISS`) index. The result is a structured, queryable repository of video knowledge that runs entirely on the local machine without requiring external API calls for inference.

The project is designed for researchers, developers, and content analysts who need to index large collections of YouTube content for downstream **retrieval-augmented generation** (`RAG`) workflows, semantic search, or question-answering systems. All inference is performed locally using open-weights models from **Hugging Face**, making the system suitable for air-gapped or privacy-sensitive environments.

## Motivation

Public video platforms produce an enormous volume of spoken information. YouTube alone hosts hundreds of millions of hours of content across lectures, technical talks, interviews, and tutorials. Despite this, the content remains largely unsearchable at a semantic level. YouTube Indexer addresses this gap by providing a lightweight, extensible indexing pipeline that converts spoken content into dense vector embeddings amenable to similarity search.

## Installation

```bash
# Clone the repository
git clone https://github.com/Mogalina/youtube-rag-indexer.git
cd youtube-rag-indexer

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

The `pip install -e .` step registers the `tubx` command globally within the virtual environment.

## Usage

All interaction occurs through the `tubx` command-line interface.

**Add videos to the queue:**

```bash
tubx add https://www.youtube.com/watch?v=<VIDEO_ID>
```

**Inspect queue status:**

```bash
tubx status
```

Displays a progress bar and a formatted table of all jobs with their current status (`pending`, `processing`, `done`, `failed`), current processing step, and last update timestamp.

**Start the processing pipeline:**

```bash
tubx run
```

Loads the summarization and embedding models into memory once, then continuously processes queued jobs using a configurable thread pool. 

**Run in background:**

```bash
tubx run --daemon
```

Starts the runner in the background and saves its process ID (PID). You can safely close your terminal.

**Stop the background runner:**

```bash
tubx stop
```

Gracefully stops the background runner after it finishes its current job.

## Requirements

- `Python` 3.10 or later
- `PyTorch` (with optional `MPS` or `CUDA` support)
- Internet access for initial transcript fetching and model downloading (local model caching supported thereafter)
