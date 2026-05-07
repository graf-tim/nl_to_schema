"""Sorgt dafür, dass `pytest` aus dem Projektroot importieren kann."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
