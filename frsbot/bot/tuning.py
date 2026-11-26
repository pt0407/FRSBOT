"""ImAI's tunable constants. Defaults are the shipped values.

IMAI_PARAMS='{"collect_frac": 0.4}' overrides them for a sweep; worker processes inherit the environment.
"""
import json
import os

DEFAULTS = dict(collect_frac=0.3, collect_min=10.0, replan_s=0.5, avoid_slot_m=1.5,
                hub_near_scale=0.8, goal_commit=1.25, avoid_point_m=0.8,
                stage_full_frac=0.95, arrive_margin_s=0.0, intake_needs_fuel=0,
                learned_hint=0, hint_threshold=0.6,
                cluster_bias=0.5,
                collect_near_frac=0.6, collect_cos_floor=0.05, collect_cruise_frac=0.68, lane_bias=1.0, pass_frac=0.7, swoop_m=1.8,
                trip_sequencing=0, trip_fill=0.85, trip_shift_aware=1,
                sweep_min_m=0.6, sweep_scale=0.8, collect_arrive_m=0.3, sweep_lane_m=0, fire_sweep=1,
                active_score_frac=0.85, use_total_capacity=0,
                early_score_until=0.0,
                drive_dist_gain=2.0, drive_brake_frac=0.4,
                pro_auto=0, active_hub_bias=0.08, score_radius_frac=0.5, free_fire=0, free_fire_speed=0.8,
                collect_stall_s=1.6, collect_dead_s=9.0,
                mow_cone_min=99, mow_speed_frac=0.35, approach_fire=1,
                pre_fire_s=2.0, post_fire_s=0.35,
                pile_exit=8,
                score_floor=10, learned_drive=0, learned_terminal_m=1.5,
                learned_blend=.6, learned_turn_blend=None)
PARAMS = {**DEFAULTS, **json.loads(os.environ.get("IMAI_PARAMS", "{}"))}
