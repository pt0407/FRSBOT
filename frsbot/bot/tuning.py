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
                trip_sequencing=0, trip_fill=0.85, trip_shift_aware=1,
                sweep_min_m=0.6, sweep_scale=0.8, collect_arrive_m=0.3,
                active_score_frac=0.85, use_total_capacity=0,
                early_score_until=0.0,
                drive_dist_gain=2.0, drive_brake_frac=0.4,
                pro_auto=0, active_hub_bias=0.08, score_radius_frac=0.5, free_fire=0, free_fire_speed=0.8,
                conveyor=0, conv_full=75.0, dump_floor=8.0, conv_spot_x=1.45, conv_spot_z=1.9,
                conv_blob_x=1.75, conv_blob_r=2.2, sweep_speed_frac=0.8, conv_dump_min=25.0,
                collect_stall_s=1.6, collect_dead_s=9.0,
                seed_inactive=0, seed_hi_frac=0.8, seed_lo_frac=0.45, seed_lead_s=10.0,
                seed_floor=15,
                mow_cone_min=99, mow_speed_frac=0.35, approach_fire=1,
                patrol=0, patrol_min=25, patrol_exit=12, patrol_speed_frac=0.22,
                patrol_low=12, patrol_high=95,
                patrol_fire_speed=0.9, seed_speed_frac=0.35,
                bank=0, pre_fire_s=2.0, post_fire_s=0.35, feed_lead_s=1.5,
                pile_min=14, pile_exit=8, zone_speed_frac=0.5, zone_fill=0.85,
                score_floor=10, score_close_m=2.1, feed_min_cluster=4)
PARAMS = {**DEFAULTS, **json.loads(os.environ.get("IMAI_PARAMS", "{}"))}
