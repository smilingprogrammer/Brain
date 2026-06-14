# BrainOfThought Fellowship Recording Guide

Use this repo as the primary work sample. It shows agent architecture, not just a single prompt around an LLM.

## Before Recording

```bash
cd /Users/abdulsobur/Desktop/Projects/empty-anything/BrainOfThought
cp .env.example .env
source .venv/bin/activate
```

If you want live Gemini responses, edit `.env`, set `BRAIN_DEMO_MODE=false`, and add a real `GEMINI_API_KEY`.
The project now tries `gemini-3-flash-preview` first, then falls back to `gemini-3.1-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-2.5-flash-lite`, and `gemini-3.5-flash`.
If all live calls are rate-limited or unavailable, `GEMINI_LOCAL_FALLBACK=true` keeps the walkthrough deterministic.

## Sanity Checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_asyncio.plugin test/test_reasoning.py -q
python main_lite.py "If all cats are animals and all animals need food, do cats need food?"
```

If the live API is quota-limited, the command will log Gemini errors and then use the local fallback. That is acceptable for recording as long as you say the code attempts live Gemini first and falls back locally to keep the demo reproducible.
For a cleaner recording terminal, use:

```bash
LOG_LEVEL=CRITICAL python main.py "If all cats are animals and all animals need food, do cats need food?"
```

## What To Show In The Video

1. Start on `README.md` and explain the architecture in one sentence:
   "BrainOfThought decomposes an LLM agent into explicit regions: language comprehension, working memory, executive planning, reasoning, global workspace integration, and final synthesis."

2. Show these files:
   - `main.py`: system entry point and event flow.
   - `core/event_bus.py`: async event bus connecting modules.
   - `brain_regions/memory/working_memory.py`: memory and compression logic.
   - `brain_regions/executive/prefrontal_cortex.py`: planning and routing.
   - `brain_regions/reasoning/logical_reasoning.py`: reasoning and proof validation.
   - `brain_regions/gemini/gemini_service.py`: live Gemini integration, model fallback, and deterministic local fallback.

3. Run the end-to-end command:

```bash
LOG_LEVEL=CRITICAL python main_lite.py "If all cats are animals and all animals need food, do cats need food?"
```

4. Connect it to the fellowship:
   - The project makes agent internals explicit instead of hiding everything inside one prompt.
   - Explicit module boundaries make it easier to add validators, monitors, permission checks, and trace-based evals.
   - This connects directly to Mining Enforceable Specifications and Agent Permission System.
   - The fallback behavior is also relevant to secure synthesis: external model failure is treated as an expected state, not an unhandled crash.

5. Mention supporting work:
   - KubeController shows rule-driven monitors and constrained actions.
   - FOSSology/Safaa shows experience with noisy real-world outputs and evaluation pipelines.
