"""
Elo Lab

Version 15.1 — Keyless NBA live slate (CDN + ESPN fallback)

Public package interface.
"""

from .engine import run_game

__version__ = "15.1"

__all__ = [
    "run_game",
]