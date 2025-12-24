# Cognitive Digital Brain

Neuromorphic text reasoning stack that wires Gemini 2.0 Flash into brain-inspired "regions" (language, memory, reasoning, executive, integration) connected by an async event bus. The primary entry point is `main.py`, which spins up the regions and exposes both an interactive shell and a single-shot CLI.

## Highlights
- Event-driven pipeline (`core/event_bus.py`) that lets regions fire and react to events concurrently with Prometheus metrics instrumentation.
- Brain-inspired modules in `brain_regions/` covering language comprehension, working/episodic memory, multi-path reasoning (logical, analogical, causal, creative), a global workspace integrator, and an executive prefrontal cortex orchestrator.
- Gemini 2.0 Flash integration via the new `google-genai` client (`brain_regions/gemini/gemini_service.py`) with shared configs for free-form and structured JSON prompting.
- Working memory + hippocampus layers that compress, retrieve, and consolidate salient context before reasoning kicks off.
- Rich docs (`technical_docs/`, `initial_setup_documentation/`) and runnable examples/tests to illustrate typical flows.

## Architecture In Practice
1. **Input & comprehension** - `CognitiveTextBrain.process_text` emits `input_received`; `LanguageComprehension` (spaCy + SentenceTransformer) parses tokens, entities, sentence embeddings, and raises `language_comprehension_complete`.
2. **Memory upkeep** - `WorkingMemory` listens for comprehension events, scores salience, compresses overflow items with Gemini summarization, and exposes retrieval APIs while updating `working_memory_updated`.
3. **Reasoning fan-out** - Prefrontal Cortex subscribes to `new_task`, decomposes the goal into sub-goals (JSON planned by Gemini), and fires events such as `reasoning_request` that `LogicalReasoning`/other modules fulfill. Gemini-driven `ReasoningEngine` can run deductive, inductive, analogical, causal, and counterfactual paths in parallel.
4. **Global broadcast** - `GlobalWorkspace` runs an attention competition, integrates the highest-salience region outputs, and re-broadcasts the synthesized state so modules can react or spawn new goals.
5. **Executive synthesis** - Prefrontal Cortex evaluates the executed plan, triggers contingencies when needed, and emits `task_complete` with the final response used by the CLI.
6. **Observability** - Structured logging via `core/logging.py` + `structlog`, Prometheus counters/histograms (`core/metrics.py`), and optional dockerized Prometheus/Grafana services keep an eye on region health.

## Repository Map
| Path | Purpose |
| --- | --- |
| `main.py` | Async entry point + CLI shell for CognitiveTextBrain. |
| `core/` | Interfaces, structlog setup, event bus, and shared metrics. |
| `brain_regions/` | All region implementations (language, memory, reasoning, executive, integration, Gemini adapters). |
| `config/settings.py`, `config/prompt.yaml` | Runtime knobs loaded via `pydantic-settings` + curated system prompts per region. |
| `utils/` | Text cleaning, vector ops, neural converters, helper utilities. |
| `examples/` | Scripts showing simple reasoning, creative prompting, and complex multi-turn problems. |
| `test/` | Pytest suites for reasoning modules plus `test/benchmarks/` for latency/accuracy measurements. |
| `initial_setup_documentation/` & `technical_docs/` | Step-by-step bring-up notes, architecture deep-dives, module-by-module explanations, and manual test playbooks. |
| `prev/` | Frozen prototypes and reference entry points for earlier experiments. |
| `docker-compose.yml`, `Dockerfile` | Containerized runtime plus optional Neo4j + Prometheus + Grafana stack (supply your own `monitoring/` configs before enabling observability services). |
| `CHANGES_LITE.md`, `SETUP_LITE.md`, `paths.md` | Historical change log and onboarding shortcuts. |

## Prerequisites
- Python 3.10+
- A Google Gemini API key with access to the `gemini-2.0-flash-exp` model (or whichever you configure).
- System packages for PyTorch and FAISS (CPU build is listed in `requirements.txt`).
- Runtime models: `python -m spacy download en_core_web_sm` and `python -m nltk.downloader punkt stopwords`.

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

### Configure Environment Variables
Create a `.env` (the repo does not ship a `.env.example`) with the settings consumed in `config/settings.py`:
```
GEMINI_API_KEY=sk-...
GEMINI_MODEL=gemini-2.0-flash-exp  # optional override
LOG_LEVEL=INFO
REASONING_TIMEOUT=180
WORKING_MEMORY_CAPACITY=7
HIPPOCAMPUS_SIZE=100000
PARALLEL_REASONING_PATHS=5
```
Quick sanity check:
```bash
python check_api_key.py
```

### Run The Brain
```bash
python main.py "What would happen if gravity was twice as strong?"
```
- Run without arguments to enter interactive chat; type `help` for prompt ideas and `exit` to quit.
- Explore richer flows via the example scripts:
  - `python -m examples.simple_reasoning`
  - `python -m examples.complex_problem`
  - `python -m examples.creative_task`
- To keep everything containerized, build and run: `docker compose up cognitive-brain`. Supply `GEMINI_API_KEY` in your shell or `.env` so the container inherits it. The compose file also defines Neo4j/Prometheus/Grafana services; create `./monitoring/prometheus.yml` and Grafana provisioning files before bringing them up.

## Configuration Surface
`Config/settings.py` exposes knobs through environment variables or `.env`:
- `GEMINI_API_KEY`, `GEMINI_MODEL`: forwarded to `brain_regions/gemini/gemini_service.py`.
- `LOG_LEVEL`: structlog + stdlib logging level.
- `REASONING_TIMEOUT`: `CognitiveTextBrain.process_text` timeout.
- `WORKING_MEMORY_CAPACITY`, `HIPPOCAMPUS_SIZE`: memory buffers.
- `PARALLEL_REASONING_PATHS`: fan-out width for `ReasoningEngine`.
Update prompt personalities per-region in `config/prompt.yaml` when you need different behaviors for comprehension, memory compression, executive planning, etc.

## Monitoring & Ops
- Logs: structured JSON via `structlog` (see `core/logging.py`). Pipe them into your preferred log stack or pretty-print via `jq`.
- Metrics: `core/metrics.py` defines Prometheus counters/histograms/gauges. Expose them by wiring `prometheus_client.start_http_server(port)` wherever you deploy or embed them into an API server.
- External services: `docker-compose.yml` includes Neo4j for future graph memory, Prometheus, and Grafana. Comment out services you do not need; create the `monitoring/` folder before launching observability containers.

## Testing & Quality Gates
```bash
pytest -q
pytest test/benchmarks/latency_tests.py -k smoke
pytest test/benchmarks/accuracy_tests.py -k causal
```
- Tests in `test/` use `pytest-asyncio`; they call real Gemini endpoints by default, so set `GEMINI_API_KEY` when running locally. For CI you can monkeypatch `GeminiService` or provide a fake client via dependency injection.
- Benchmark suites provide coarse latency/accuracy snapshots for reasoning paths; expect higher variance without caching.

## Documentation & Next Steps
- `technical_docs/architecture.md`, `technical_docs/technical_documentation.md`: architecture narrative, module-by-module diagrams, and reasoning walkthroughs.
- `initial_setup_documentation/`: historical onboarding notes plus manual validation scripts.
- `paths.md`: curated map of how modules connect (useful when extending/adding brain regions).
- `prev/*.py`: legacy driver scripts retained for reference; start from `main.py` for all new work.

### Roadmap Ideas
- Persist Prometheus metrics via the provided compose stack and export `metrics_registry` through an HTTP endpoint.
- Integrate Neo4j or Chroma memory stores (dependencies already declared) to persist semantic memories beyond runtime buffers.
- Expand `ReasoningEngine` strategies (e.g., probabilistic programming) or add scoring hooks so Prefrontal Cortex can prioritize higher-fidelity paths.

Have fun exploring how each brain-inspired component cooperates, and update only this README when documenting additional capabilities.

