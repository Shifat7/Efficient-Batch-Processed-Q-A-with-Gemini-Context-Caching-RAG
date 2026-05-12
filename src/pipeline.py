"""RAG pipeline orchestrator."""
from __future__ import annotations
import time

from src.ingest import load_and_segment_transcript
from src.summarize import summarize_chunk
from src.embed_retrieve import build_index, retrieve
from src.qa import batch_gemini_qa
from src.utils import log_metrics, Timer


def run_pipeline(
    transcript_path: str,
    questions: list[str],
    top_k: int = 3,
    max_chunk_words: int = 800,
    overlap_words: int = 100,
) -> dict[str, str]:
    """End-to-end RAG pipeline with batch Gemini prediction.

    Steps:
        1. Ingest: load and chunk the transcript.
        2. Summarise: produce a short summary of each chunk.
        3. Embed + index: build a FAISS index over raw chunks.
        4. Retrieve: for each question fetch top_k relevant chunks.
        5. Batch QA: answer ALL questions in a single Gemini call.
        6. Metrics: log timing and counts.

    Args:
        transcript_path: Path to a plain-text transcript file.
        questions: List of question strings.
        top_k: Number of chunks to retrieve per question as context.
        max_chunk_words: Maximum words per chunk during ingestion.
        overlap_words: Word overlap between consecutive chunks.

    Returns:
        Dict mapping each question to its answer string.

    Raises:
        FileNotFoundError: If transcript_path does not exist.
    """
    if not questions:
        return {}

    start = time.time()

    # 1. Ingest
    with Timer("ingest"):
        chunks = load_and_segment_transcript(
            transcript_path,
            max_words=max_chunk_words,
            overlap_words=overlap_words,
        )

    # 2. Summarise (used for richer context in future; computed now for metrics)
    with Timer("summarise"):
        summaries = [summarize_chunk(c) for c in chunks]

    # 3. Embed + index
    with Timer("embed"):
        model, index, _ = build_index(chunks)

    # 4. Retrieve — merge top_k chunks across all questions into one context block
    with Timer("retrieve"):
        seen: set[str] = set()
        context_chunks: list[str] = []
        for q in questions:
            for chunk in retrieve(q, model, index, chunks, top_k=top_k):
                if chunk not in seen:
                    seen.add(chunk)
                    context_chunks.append(chunk)
        context = "\n\n".join(context_chunks)

    # 5. Batch QA — single Gemini call for all questions
    with Timer("gemini_batch"):
        answers = batch_gemini_qa(questions, context)

    # 6. Metrics
    elapsed = time.time() - start
    log_metrics({
        "total_time_s": round(elapsed, 4),
        "num_chunks": len(chunks),
        "num_summaries": len(summaries),
        "num_questions": len(questions),
        "context_chunks_used": len(context_chunks),
    })

    return answers
