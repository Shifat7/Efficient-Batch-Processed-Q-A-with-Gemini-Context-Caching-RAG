# Handover Document — Optimized Gemini Context Caching RAG

> **Status:** Work in Progress — Skeleton / Proposal-Stage Implementation  
> **Last Updated:** May 2026  
> **Author:** Shifat Rahman  
> **Original Challenge:** [GSoC DeepMind Q4 — Batch Prediction + Context Caching](https://gist.github.com/dynamicwebpaige/92f7739ad69d2863ac7e2032fe52fbad#4-batch-prediction-with-long-context-and-context-caching-code-sample)

---

## Overview

This project is a local demo implementing batch prediction with Gemini APIs, leveraging long context and context caching to efficiently answer multiple questions about a single video/document source. The use case: extract information from a video lecture or documentary by submitting a batch of potentially interconnected questions against a cached context.

The architecture is sound and the scaffolding is in place. As of the current state, the embedding/retrieval layer is functional but the core Gemini API integration (context caching, batch prediction, and summarisation) are all stubbed out and need to be implemented.

---

## Repository Structure

```
.
├── src/
│   ├── __main__.py          # Entry point
│   ├── ingest.py            # Transcript loading and chunking
│   ├── embed_retrieve.py    # FAISS-based semantic retrieval
│   ├── pipeline.py          # Orchestration of the full RAG pipeline
│   ├── summarize.py         # STUB — chunk summarisation
│   ├── qa_mock.py           # STUB — Gemini QA (mocked)
│   └── utils.py             # Metrics logging, token estimation, Timer
├── inputs/                  # Drop transcript files here
├── tests/                   # Empty — tests to be written
├── pyproject.toml           # Poetry project config
├── poetry.lock
├── gsoc proposal.pdf        # Original submitted GSoC proposal
└── README.md
```

**Run the project:**
```bash
poetry install
poetry run python src
```

---

## Current State Assessment

### What Works ✅

| Module | Status | Notes |
|--------|--------|-------|
| `ingest.py` — chunking | ✅ Functional | Sliding window with configurable `max_words=800`, `overlap_words=100` |
| `embed_retrieve.py` — FAISS index | ✅ Functional | `all-MiniLM-L6-v2` + `faiss.IndexFlatL2`, `top_k=3` retrieval |
| `utils.py` — Timer + logging | ✅ Functional | JSON metrics saved to `logs/` with timestamp |
| `pipeline.py` — orchestration skeleton | ✅ Structural | Wires all modules together correctly; runs end-to-end on mocked components |

### What Is Stubbed / Missing ❌

| Module | Status | Notes |
|--------|--------|-------|
| `qa_mock.py` | ❌ Stub | Returns a hardcoded string. No Gemini API call. |
| `summarize.py` | ❌ Stub | Truncates to 50 words with `...`. No real summarisation. |
| Gemini Context Caching | ❌ Not implemented | `CachedContent` / `google.generativeai` not used anywhere in codebase |
| Batch Prediction | ❌ Not implemented | Questions are answered one-by-one in a `for` loop in `pipeline.py` |
| Tests | ❌ Empty | `/tests` directory exists but contains no test files |
| Error handling | ❌ Minimal | No API error handling, no retry logic, no input validation |
| Token estimation | ⚠️ Rough | `utils.py` uses `len(text) // 4` — acceptable for English prose but unreliable for code or non-Latin text |

### Known Logic Bug in `ingest.py`

The `load_and_segment_transcript` function has a subtle bug: it first builds `chunks` via a sliding word window, then separately checks `paragraphs` and appends any paragraph not already contained in a chunk. However the containment check (`if not any(paragraph in chunk ...)`) does a substring match on joined word strings, which can produce false positives or miss duplicates depending on whitespace. The paragraph preservation logic should either be removed (the word chunks already cover all text) or be replaced with a proper paragraph-aware chunking strategy.

---

## Pipeline Architecture

```
Input: transcript file (TXT)
         │
         ▼
  [ingest.py] load_and_segment_transcript()
  Sliding window chunks (800 words, 100-word overlap)
         │
         ▼
  [summarize.py] summarize_chunk()          ← STUB — not used downstream
         │
         ▼
  [embed_retrieve.py] build_index()
  SentenceTransformer('all-MiniLM-L6-v2') + FAISS IndexFlatL2
         │
         ▼
  For each question:
    [embed_retrieve.py] retrieve()           ← top-k=3 chunks
    [qa_mock.py] mock_gemini_qa()            ← STUB — returns dummy string
         │
         ▼
  Output: dict { question → answer }
```

The critical gap is that `mock_gemini_qa()` needs to be replaced with a real Gemini API call that takes the retrieved `ctx_text` and uses a cached context object for efficiency.

---

## Previous Implementations / Design Decisions

### Why `all-MiniLM-L6-v2` + FAISS

A pragmatic choice for local dev: `all-MiniLM-L6-v2` is a fast, well-benchmarked sentence embedding model with a low memory footprint. `faiss.IndexFlatL2` does brute-force L2 search — perfectly fine for small-to-medium transcript sizes (thousands of chunks). For production at scale, `IndexIVFFlat` or `IndexHNSWFlat` would be faster with approximate nearest-neighbour search.

### Why `top_k=3`

A sensible default for retrieving relevant context per question without overloading the Gemini prompt. This can be made configurable as a parameter to `retrieve()`.

### Chunking Strategy

Word-based sliding window rather than sentence or paragraph-based. This ensures no chunk exceeds the maximum word count, but it can split mid-sentence. A future improvement would be to use a sentence boundary splitter (e.g., `nltk.sent_tokenize`) as the primary unit before applying the word cap.

---

## Priority Roadmap for Future Contributors

### P0 — Core Integration (Project doesn't work without these)

1. **Replace `mock_gemini_qa` with real Gemini API calls**
   - Install `google-generativeai` (or `google-genai` for the newer SDK)
   - Use `gemini-1.5-pro` or `gemini-2.0-flash` depending on budget/latency requirements
   - Pass `ctx_text` as context in the prompt
   - Handle `google.api_core.exceptions` (rate limits, quota exhaustion, invalid requests)

2. **Implement Gemini Context Caching**
   - This is the core differentiator of the project (the namesake feature)
   - Upload the full transcript or a pre-built summary once using `genai.caching.CachedContent.create()`
   - Set a sensible TTL (e.g., 1 hour for a session)
   - Pass the `cached_content` reference to all subsequent QA calls in the session
   - Log cache hit/miss rates in `utils.log_metrics()`
   - Example pattern:
     ```python
     cache = genai.caching.CachedContent.create(
         model='models/gemini-1.5-pro-001',
         system_instruction='You are an expert assistant...',
         contents=[full_transcript_text],
         ttl=datetime.timedelta(hours=1),
     )
     model = genai.GenerativeModel.from_cached_content(cache)
     response = model.generate_content(question)
     ```

3. **Implement Batch Prediction**
   - Instead of calling Gemini once per question in a loop, collect all questions and submit them in a single prompt with the cached context
   - Structure the prompt to instruct the model to answer all questions in a numbered list
   - Parse the structured response back into a `{ question: answer }` dict
   - This reduces API calls from `N` (one per question) to `1` — the primary optimisation goal of the GSoC challenge

### P1 — Quality of Life

4. **Fix `summarize_chunk` with real summarisation**
   - Replace the 50-word truncation with either a Gemini summarisation call (using the same cached context) or a lightweight local model (e.g., `facebook/bart-large-cnn` via HuggingFace)
   - Wire summaries into the retrieval prompt as condensed context alongside raw chunks

5. **Fix the `ingest.py` paragraph deduplication bug**
   - Simplify: remove the paragraph preservation loop at the bottom of `load_and_segment_transcript` — it introduces duplicates and the sliding window already covers all content
   - Or: rewrite chunking to be paragraph-aware using `\n\n` splits first, then applying word caps per paragraph group

6. **Add a GEMINI_API_KEY config layer**
   - Currently there's no `.env` / `config.py` / `pyproject.toml` secret management
   - Add `python-dotenv` to `pyproject.toml` and load `GEMINI_API_KEY` from a `.env` file
   - Add `.env` to `.gitignore` (it's already listed but double-check)

7. **Improve token estimation**
   - Replace `len(text) // 4` with Gemini's native token counting: `model.count_tokens(text)`
   - This gives exact token counts and is necessary for managing context window limits

### P2 — Tests and Robustness

8. **Write unit tests in `/tests`**
   - `test_ingest.py`: test chunking with known inputs (check chunk count, overlap, edge cases: empty string, single-word input, text shorter than `max_words`)
   - `test_embed_retrieve.py`: test that `retrieve()` returns exactly `top_k` results and that known-similar text scores higher than dissimilar text
   - `test_pipeline.py`: integration test with a mock transcript and mocked Gemini calls

9. **Add retry logic for API calls**
   - Wrap Gemini calls with exponential backoff using `tenacity` library
   - Log retries in `utils.py`

10. **Timestamp-to-answer mapping (stretch goal)**
    - The README lists "links to relevant timestamps in the video" as a feature
    - This requires the transcript to include timestamp markers (e.g., from `yt-dlp` VTT output)
    - Parse timestamps in `ingest.py` and propagate them through chunks so retrieved context includes `[HH:MM:SS]` markers in the answer

### P3 — Extended Features

11. **Video transcript ingestion via `yt-dlp`**
    - Automate transcript download from YouTube URLs using `yt-dlp --write-auto-sub`
    - Add a `download_transcript(url)` function in `ingest.py`

12. **Persistent caching across sessions**
    - Store Gemini `CachedContent` names/IDs in a local JSON file keyed by transcript hash
    - On re-run with the same transcript, reuse the existing cache rather than re-uploading

13. **Multi-document support**
    - Extend the pipeline to handle multiple transcript files and route questions to the most relevant document before retrieving chunks

---

## Environment Setup

```toml
# pyproject.toml — current dependencies
[tool.poetry.dependencies]
python = "^3.11"
sentence-transformers = "*"
faiss-cpu = "*"
```

**Dependencies to add for P0:**
```toml
google-generativeai = ">=0.7"  # or google-genai for newer SDK
python-dotenv = ">=1.0"
```

**Dependencies to add for P1/P2:**
```toml
tenacity = ">=8.0"       # retry logic
pytest = ">=8.0"         # tests
nltk = ">=3.9"           # sentence tokenisation (optional)
```

---

## API Key Setup

1. Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. Load it in your module:
   ```python
   from dotenv import load_dotenv
   import os
   load_dotenv()
   api_key = os.getenv("GEMINI_API_KEY")
   ```

---

## References

- [GSoC DeepMind Challenge Q4 — Original prompt](https://gist.github.com/dynamicwebpaige/92f7739ad69d2863ac7e2032fe52fbad#4-batch-prediction-with-long-context-and-context-caching-code-sample)
- [Gemini Context Caching Docs](https://ai.google.dev/gemini-api/docs/caching)
- [google-generativeai Python SDK](https://github.com/google-gemini/generative-ai-python)
- [FAISS Documentation](https://faiss.ai/)
- [SentenceTransformers — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Original GSoC proposal (PDF)](https://github.com/Shifat7/Efficient-Batch-Processed-Q-A-with-Gemini-Context-Caching-RAG/blob/main/gsoc%20proposal.pdf)
