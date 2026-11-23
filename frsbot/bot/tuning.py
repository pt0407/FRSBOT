"""ImAI's tunable constants. Defaults are the shipped values.

IMAI_PARAMS='{"collect_frac": 0.4}' overrides them for a sweep; worker processes inherit the environment.
"""
import json
import os

DEFAULTS = dict(collect_frac=0.3, collect_min=10.0, replan_s=0.5, avoid_slot_m=1.5,
                hub_near_scale=0.8, goal_commit=1.25, avoid_point_m=0.8,
                stage_full_frac=0.95, arrive_margin_s=0.0, intake_needs_fuel=0,
                score_full_frac=0.85,
                learned_hint=0, hint_threshold=0.6,
                cluster_bias=0.5,
                collect_near_frac=0.6, collect_cos_floor=0.05,
                trip_sequencing=0, trip_fill=0.85)
PARAMS = {**DEFAULTS, **json.loads(os.environ.get("IMAI_PARAMS", "{}"))}
