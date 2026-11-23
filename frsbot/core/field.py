"""Match timing measured from replay metadata; time zero is auto start."""
import math

from frsbot.core.state import Phase

ALLIANCES = ("blue", "red")
# Midpoints of x/z extrema for where==3 FUEL above y=1.7 m across 45 cached
# 30 Hz replays (89,967 blue / 1,249 red samples); max y gives rim height.
# Rounded to 1 mm, with sub-mm centre asymmetry suppressed. Tuples: (x, z, rim y).
HUB_CENTRES = {"blue": (-3.645, 0.0, 1.829), "red": (3.645, 0.0, 1.829)}
TIMELINE = (
    (0.0, Phase.AUTO), (20.0, Phase.GAP), (23.0, Phase.TRANSITION),
    (33.0, Phase.SHIFT1), (58.0, Phase.SHIFT2), (83.0, Phase.SHIFT3),
    (108.0, Phase.SHIFT4), (133.0, Phase.ENDGAME), (163.0, Phase.SETTLE), (166.0, Phase.DONE),
)


def phase_at(t: float) -> Phase:
    if not math.isfinite(t):
        raise ValueError("match time must be finite")
    phase = Phase.COUNTDOWN
    for start, candidate in TIMELINE:
        if t < start:
            break
        phase = candidate
    return phase


def is_active(alliance: str, t: float, first_inactive: str | None) -> bool:
    if alliance not in ALLIANCES or first_inactive not in (*ALLIANCES, None):
        raise ValueError("alliance and first_inactive must be blue or red")
    phase = phase_at(t)
    if phase in (Phase.AUTO, Phase.TRANSITION, Phase.ENDGAME):
        return True
    if phase in (Phase.SHIFT1, Phase.SHIFT2, Phase.SHIFT3, Phase.SHIFT4):
        if first_inactive is None:
            raise ValueError("first_inactive is required during shifts")
        first_off = phase in (Phase.SHIFT1, Phase.SHIFT3)
        return (alliance != first_inactive) if first_off else (alliance == first_inactive)
    return False


def seconds_to_toggle(alliance: str, t: float, first_inactive: str | None) -> float:
    """Time to an actual activity change, or infinity if no changes remain."""
    active = is_active(alliance, t, first_inactive)
    for start, _ in TIMELINE:
        if start > t and is_active(alliance, start, first_inactive) != active:
            return start - t
    return math.inf
