"""Vercel serverless entry point: serves the Drone Design Studio API with the same handler as the local app."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DDS_SAVED", "browser")   # no lasting disk on Vercel; saved builds stay in the visitor's browser

from drone_app import Handler  # noqa: E402


class handler(Handler):
    pass
