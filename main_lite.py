"""Clean entry point for the lightweight BrainOfThought demo."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prev.main_lite import main


if __name__ == "__main__":
    asyncio.run(main())
