"""Compatibility command; implementation lives in facetad."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module("facetad.evaluate", run_name="__main__")
