from pathlib import Path

from frsbot.baseline import route_follow
from frsbot.scrape import events, harvest
from frsbot.sim import push
from frsbot.tools import build_viewer


def test_repository_paths_after_move():
    root = Path(__file__).resolve().parents[1]
    assert route_follow.ROOT == push.ROOT == build_viewer.ROOT == root
    assert harvest.DATA == root / "data"
    assert events.DATA == root / "data" / "events"
