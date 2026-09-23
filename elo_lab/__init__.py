"""
Elo Lab

Version 15.0 — Live slate: ingest completed scores + lock UI

Public package interface.
"""

from .engine.game_runner import run_game

__version__ = "15.0"

__all__ = [
    "run_game",
]