"""Extracted bundle rules, not the official rulebook or a WASM equivalence claim.

Sources are the public assets cached under data/engine. The JS judge and Rust
engine are separate implementations; constants below are from the JS judge.
"""

# 2c3zgga2f0b2g.js, module 2960, W.minorFoul / W.majorFoul;
# 1o_abmr6co2n1.js, B() credits the opposite alliance, board.foul().
MINOR_FOUL_POINTS = 15
MAJOR_FOUL_POINTS = 30
FOUL_CREDIT = "opponent"

# 1o_abmr6co2n1.js, rules UI list and judge methods autoContact,
# controlled, pinning, blockade, towerContact. No separate tech-foul type.
IMPLEMENTED_RULES = ("G403", "G408", "G418", "G419", "G420")
TECH_FOUL_TYPE = None

# autoContact / towerContact use pair(), A(0, 1): immediate, 1 s clear to rearm.
# 2lnu-vtgpos5u.js contactFouls: G403 only in auto, offender's entire
# bumper on opponent half; G420 only in endgame, victim touching its tower.
AUTO_CONTACT_RULE = "G403"
AUTO_CONTACT_POINTS = MAJOR_FOUL_POINTS
TOWER_CONTACT_RULE = "G420"
TOWER_CONTACT_POINTS = MAJOR_FOUL_POINTS
CONTACT_REARM_SECONDS = 1.0
TOWER_CONTACT_GRANTS_L3_IF_HANGING = True

# 1o_abmr6co2n1.js judge.pinning: run counts its initial recognition second;
# first call at 3 s is minor, then major every 3 s. Clearance is bumper gap
# OR either robot moving from the original pin positions; strictly >1.5 s.
PIN_RULE = "G418"
PIN_RECOGNITION_SECONDS = 1.0
PIN_CALL_SECONDS = 3.0
PIN_FIRST_POINTS = MINOR_FOUL_POINTS
PIN_REPEAT_POINTS = MAJOR_FOUL_POINTS
PIN_CLEAR_DISTANCE_METERS = 1.8288
PIN_CLEAR_SECONDS = 1.5

# 2lnu-vtgpos5u.js contactFouls and 1o_abmr6co2n1.js exports W/E:
# hung victims, victims on a bump, or moving >=0.4 m per trail interval
# cannot start a pin. Detection also requires pushing and heldBehind geometry.
PIN_VICTIM_TRAIL_DISTANCE_METERS = 0.4
PIN_PUSH_THRESHOLD = 0.5

# 1o_abmr6co2n1.js blockade / zoneBlockade, default blockadePens/Ways false.
# Two partners blocking BOTH trenches or BOTH bumps (or trapping one victim),
# not a blanket limit on defender count or opponent-zone occupancy. Phase/hub
# exceptions are applied by zoneBlockade and 2lnu-vtgpos5u.js blockadeStage.
BLOCKADE_RULE = "G419"
BLOCKADE_POINTS = MAJOR_FOUL_POINTS
BLOCKADE_PARTNERS = 2
BLOCKADE_ZONE_LOITER_SECONDS = 0.25
BLOCKADE_HOLD_SECONDS = 0.25
BLOCKADE_REPEAT_SECONDS = 3.0
BLOCKADE_CLEAR_SECONDS = 1.0
BLOCKADE_TRAP_SECONDS = 0.5
BLOCKADE_PENS_DEFAULT = False
BLOCKADE_WAYS_DEFAULT = False
BLOCKADE_PENS_HOLD_SECONDS = 0.5
BLOCKADE_PENS_TRAP_SECONDS = 3.0

# 2lnu-vtgpos5u.js blockadeStage and 1o_abmr6co2n1.js zoneBlockade:
# stage 1 suppresses zone lists entirely; stages 2/3 exclude own-side zones.
# Stage 3: first 2 s of our active shift, or last 3 s of opponent active
# shifts 1-3. Stage 1: rest of our active shift until its final 3 s.
# Stage 2: transition/endgame. Otherwise stage 0. This is NOT a zone ban.
BLOCKADE_STAGE_DEFAULT = 0
BLOCKADE_STAGE_ACTIVE_EXEMPT = 1
BLOCKADE_STAGE_BOTH_ACTIVE = 2
BLOCKADE_STAGE_GRACE = 3
BLOCKADE_ACTIVE_START_GRACE_SECONDS = 2.0
BLOCKADE_SHIFT_END_GRACE_SECONDS = 3.0

# judge.controlled: catching hub-return FUEL before carpet; consecutive catch
# window resets after a gap >1.5 s. These override the generic 15/30 values.
HUB_CATCH_RULE = "G408"
HUB_CATCH_POINTS_BY_COUNT = {1: 5, 6: 10, 15: 15}
HUB_CATCH_RESET_SECONDS = 1.5

# judge.expansion is empty; hubCrossing returns no fouls. No generic protected
# hub/HP-area contact rule or shift-based opponent-zone ban was found in JS.
GENERIC_PROTECTED_ZONE_CONTACT_RULE = None
OPPONENT_ZONE_SHIFT_BAN = False
