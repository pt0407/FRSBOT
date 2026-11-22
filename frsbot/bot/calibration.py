"""Measured controller parameters; no backend dependencies."""
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def profiles():
    return json.loads((Path(__file__).resolve().parents[2] / "data" / "bot_profiles.json").read_text())


def profile(team):
    return profiles().get(str(team), {})
