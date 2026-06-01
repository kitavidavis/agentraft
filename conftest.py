"""Root conftest: put the repo root on sys.path so tests can import the top-level
``benchmarks`` package (which is a dev/eval tool, not part of the installed wheel)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
