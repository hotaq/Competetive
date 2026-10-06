from __future__ import annotations
import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

# ============================================================
# Shared Configuration
# ============================================================

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
TOTAL_STEPS = 500
SIM_HORIZON = 110
ROUTE_SEARCH_HORIZON = 60
HORIZON = SIM_HORIZON
LAUNCH_CLEARANCE = 0.1

EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
LATE_REMAINING_TURNS = 70          # was 60 — enter late-game earlier
VERY_LATE_REMAINING_TURNS = 25
TOTAL_WAR_REMAINING_TURNS = 55     # was 38 — endgame push starts sooner

SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
INTERCEPT_TOLERANCE = 1

SAFE_OPENING_PROD_THRESHOLD = 4
SAFE_OPENING_TURN_LIMIT = 10
ROTATING_OPENING_MAX_TURNS = 13
ROTATING_OPENING_LOW_PROD = 2
FOUR_PLAYER_ROTATING_REACTION_GAP = 3
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.55  # was 0.62 — less overcommit in 4P
FOUR_PLAYER_ROTATING_TURN_LIMIT = 10

COMET_MAX_CHASE_TURNS = 10

ATTACK_COST_TURN_WEIGHT = 0.50
SNIPE_COST_TURN_WEIGHT = 0.42
INDIRECT_VALUE_SCALE = 0.15
INDIRECT_FRIENDLY_WEIGHT = 0.35
INDIRECT_NEUTRAL_WEIGHT = 0.9
INDIRECT_ENEMY_WEIGHT = 1.25

STATIC_NEUTRAL_VALUE_MULT = 1.4
STATIC_HOSTILE_VALUE_MULT = 1.65
ROTATING_OPENING_VALUE_MULT = 0.9
HOSTILE_TARGET_VALUE_MULT = 2.05   # was 1.95
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.55
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
EARLY_NEUTRAL_VALUE_MULT = 1.2
COMET_VALUE_MULT = 0.65
SNIPE_VALUE_MULT = 1.12
SWARM_VALUE_MULT = 1.05
REINFORCE_VALUE_MULT = 1.35
CRASH_EXPLOIT_VALUE_MULT = 1.18
FINISHING_HOSTILE_VALUE_MULT = 1.3  # was 1.2
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92
EXPOSED_PLANET_VALUE_MULT = 2.0    # was 1.75

# NEW: weakest enemy targeting multipliers
WEAKEST_ENEMY_VALUE_MULT_4P = 2.2   # bonus for targeting weakest in 4P
WEAKEST_ENEMY_VALUE_MULT_2P = 1.25  # bonus for targeting weakest in 1v1
GANG_UP_VALUE_MULT = 1.4            # bonus for gang-up attacks on weakened planets
GANG_UP_VALUE_MULT_4P = 1.85         # stronger in 4P to exploit inter-enemy fights
GANG_UP_POST_BATTLE_DELAY = 2       # turns after battle to arrive
GANG_UP_POST_BATTLE_DELAY_4P = 1    # arrive faster in 4P cleanup windows
GANG_UP_ETA_WINDOW = 4              # tolerance window for gang-up timing
GANG_UP_ETA_WINDOW_4P = 6           # wider 4P gang-up timing window

NEUTRAL_MARGIN_BASE = 2
NEUTRAL_MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 8
HOSTILE_MARGIN_BASE = 3
HOSTILE_MARGIN_PROD_WEIGHT = 2
HOSTILE_MARGIN_CAP = 12
HOSTILE_REINFORCE_HORIZON = 8
HOSTILE_REINFORCE_RATIO = 0.22
HOSTILE_REINFORCE_CAP = 12
STATIC_TARGET_MARGIN = 4
CONTESTED_TARGET_MARGIN = 5
FOUR_PLAYER_TARGET_MARGIN = 2       # was 3 — more efficient in 4P
LONG_TRAVEL_MARGIN_START = 18
LONG_TRAVEL_MARGIN_DIVISOR = 3
LONG_TRAVEL_MARGIN_CAP = 8
COMET_MARGIN_RELIEF = 6
FINISHING_HOSTILE_SEND_BONUS = 3

STATIC_TARGET_SCORE_MULT = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.84
DENSE_STATIC_NEUTRAL_COUNT = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.86
SNIPE_SCORE_MULT = 1.12
SWARM_SCORE_MULT = 1.06
CRASH_EXPLOIT_SCORE_MULT = 1.05

FOLLOWUP_MIN_SHIPS = 8
LOW_VALUE_COMET_PRODUCTION = 1
LATE_CAPTURE_BUFFER = 5
VERY_LATE_CAPTURE_BUFFER = 3

DEFENSE_LOOKAHEAD_TURNS = 28
DEFENSE_COST_TURN_WEIGHT = 0.4
DEFENSE_FRONTIER_SCORE_MULT = 1.12
DEFENSE_SEND_MARGIN_BASE = 1
DEFENSE_SEND_MARGIN_PROD_WEIGHT = 1
DEFENSE_SHIP_VALUE = 0.55

REINFORCE_ENABLED = True
REINFORCE_MIN_PRODUCTION = 2
REINFORCE_MAX_TRAVEL_TURNS = 22
REINFORCE_SAFETY_MARGIN = 2
REINFORCE_MAX_SOURCE_FRACTION = 0.75
REINFORCE_MIN_FUTURE_TURNS = 40
REINFORCE_HOLD_LOOKAHEAD = 20
REINFORCE_COST_TURN_WEIGHT = 0.35

RECAPTURE_LOOKAHEAD_TURNS = 10
RECAPTURE_COST_TURN_WEIGHT = 0.52
RECAPTURE_VALUE_MULT = 0.88
RECAPTURE_FRONTIER_MULT = 1.08
RECAPTURE_PRODUCTION_WEIGHT = 0.6
RECAPTURE_IMMEDIATE_WEIGHT = 0.4

REAR_SOURCE_MIN_SHIPS = 16
REAR_SOURCE_MIN_SHIPS_4P = 22
REAR_DISTANCE_RATIO = 1.25
REAR_STAGE_PROGRESS = 0.78
REAR_SEND_RATIO_TWO_PLAYER = 0.62
REAR_SEND_RATIO_FOUR_PLAYER = 0.45  # was 0.7 — less overcommit in 4P
REAR_SEND_MIN_SHIPS = 10
REAR_MAX_TRAVEL_TURNS = 40

PARTIAL_SOURCE_MIN_SHIPS = 6
MULTI_SOURCE_TOP_K = 5
MULTI_SOURCE_ETA_TOLERANCE = 2
MULTI_SOURCE_PLAN_PENALTY = 0.97
HOSTILE_SWARM_ETA_TOLERANCE = 1
THREE_SOURCE_SWARM_ENABLED = True
THREE_SOURCE_MIN_TARGET_SHIPS = 20
THREE_SOURCE_ETA_TOLERANCE = 1
THREE_SOURCE_PLAN_PENALTY = 0.93

PROACTIVE_DEFENSE_HORIZON = 12
PROACTIVE_DEFENSE_RATIO = 0.28      # was 0.20 — more defensive
MULTI_ENEMY_PROACTIVE_HORIZON = 14
MULTI_ENEMY_PROACTIVE_HORIZON_4P = 18
MULTI_ENEMY_PROACTIVE_RATIO = 0.35  # was 0.24 — much more defensive in 4P
MULTI_ENEMY_PROACTIVE_RATIO_4P = 0.48
MULTI_ENEMY_STACK_WINDOW = 4        # was 3 — wider detection window
MULTI_ENEMY_STACK_WINDOW_4P = 6
REACTION_SOURCE_TOP_K_MY = 4
REACTION_SOURCE_TOP_K_ENEMY = 4
PROACTIVE_ENEMY_TOP_K = 3

CRASH_EXPLOIT_ENABLED = True
CRASH_EXPLOIT_MIN_TOTAL_SHIPS = 7   # was 10 — exploit smaller crashes too
CRASH_EXPLOIT_ETA_WINDOW = 3        # was 2 — wider window
CRASH_EXPLOIT_POST_CRASH_DELAY = 1

LATE_IMMEDIATE_SHIP_VALUE = 0.75
WEAK_ENEMY_THRESHOLD = 110          # was 60 — detect weak enemies earlier
WEAK_ENEMY_THRESHOLD_4P = 180        # detect 4P elimination windows earlier
ELIMINATION_BONUS = 55.0            # was 28.0 — much stronger elimination drive
ELIMINATION_BONUS_4P = 90.0         # stronger 4P kill commitment

BEHIND_DOMINATION = -0.20
AHEAD_DOMINATION = 0.15
FINISHING_DOMINATION = 0.28
FINISHING_PROD_RATIO = 1.15
AHEAD_ATTACK_MARGIN_BONUS = 0.12
BEHIND_ATTACK_MARGIN_PENALTY = 0.05
FINISHING_ATTACK_MARGIN_BONUS = 0.12

DOOMED_EVAC_TURN_LIMIT = 24
DOOMED_MIN_SHIPS = 8

SOFT_ACT_DEADLINE = 0.82
HEAVY_PHASE_MIN_TIME = 0.16
OPTIONAL_PHASE_MIN_TIME = 0.08
HEAVY_ROUTE_PLANET_LIMIT = 32

# Counter-planner layer derived from replay analysis. Keep this as a thin
# modifier over the proven v2 planner: classify opponent behavior, then nudge
# target values / reserves instead of rewriting the whole strategy.
COUNTER_QUIET_AFTER_STEP = 210
COUNTER_QUIET_TURNS = 30
COUNTER_PASSIVE_HOSTILE_MULT = 1.18
COUNTER_PASSIVE_NEUTRAL_MULT = 1.10

COUNTER_LATE_SWARM_START = 430
COUNTER_LATE_SWARM_WINDOW = 20
COUNTER_LATE_SWARM_SHIPS = 500
COUNTER_LATE_SWARM_DEFENSE_MULT = 1.20
COUNTER_LATE_SWARM_EXPOSED_ENEMY_MULT = 1.25
COUNTER_LATE_SWARM_COMET_MULT = 0.55

COUNTER_COMET_WINDOWS = (50, 150, 250, 350, 450)
COUNTER_COMET_WINDOW_RADIUS = 12
COUNTER_MIN_COMET_LIFE = 12
COUNTER_ROTATOR_HOLD_MARGIN = 3

# Anti-stall layer from evolve_live seed 100-103 analysis.  The counter bot was
# equal in the opening, then stopped expanding when old_base_v2 pulled ahead on
# production.  These knobs deliberately release proactive reserve and boost
# short profitable captures only when a single enemy has a clear map-control
# lead, so the default v2 snowball stays intact on neutral/even boards.
RECOVERY_PRESSURE_START_STEP = 90
RECOVERY_PROD_GAP = 5
RECOVERY_PLANET_GAP = 2
RECOVERY_RESERVE_MULT = 0.35
RECOVERY_TARGET_VALUE_MULT = 1.45
RECOVERY_HIGH_PROD_MULT = 1.18
RECOVERY_MIN_TARGET_PROD = 1
RECOVERY_MAX_ETA = 25
RECOVERY_SOURCE_MIN_SHIPS = 6
RECOVERY_SCORE_MULT = 1.9
RECOVERY_SEVERE_PROD_GAP = 16
RECOVERY_SEVERE_PLANET_GAP = 5
RECOVERY_SEVERE_EXACT_KEEP_MULT = 0.72

# Prevent early low-value trickle: seed 102 showed repeated small fleets sent at
# a 54-ship, +2 neutral over 20+ turns.  That ties up ships without tempo.
BAD_TRICKLE_TURN_LIMIT = 60
BAD_TRICKLE_MAX_PROD = 2
BAD_TRICKLE_MIN_DEFICIT = 15
BAD_TRICKLE_MIN_ETA = 18
OPENING_UNDERPOWERED_TURN_LIMIT = 45
OPENING_UNDERPOWERED_MIN_ETA = 24
OPENING_UNDERPOWERED_MAX_PROD = 4
OPENING_UNDERPOWERED_MIN_DEFICIT = 1

CONTESTED_HIGH_PROD_LOOKBACK = 15
CONTESTED_HIGH_PROD_MIN = 3
CONTESTED_HIGH_PROD_VALUE_MULT = 1.22

# W1-B fresh frontier recapture / denial: 4P-only tactical window for
# prod-2+ enemy planets that were just captured and are still weakly held.
# This copies the episode_75526435 pattern (p20/p23 and p11/p8 steals)
# without reopening broad big-first or global low-value capture loosening.
FRESH_RECAPTURE_ENABLED = True
FRESH_RECAPTURE_START_STEP = 18
FRESH_RECAPTURE_LOOKBACK = 12
FRESH_RECAPTURE_MIN_PRODUCTION = 2
FRESH_RECAPTURE_MAX_GARRISON = 13
FRESH_RECAPTURE_GARRISON_PROD_MULT = 5
FRESH_RECAPTURE_MAX_ETA = 12
FRESH_RECAPTURE_HOLD_TURNS = 8
FRESH_RECAPTURE_MIN_SOURCE_SHIPS = 8
FRESH_RECAPTURE_SOURCE_TOP_K = 4
FRESH_RECAPTURE_VALUE_MULT = 1.85
FRESH_RECAPTURE_SCORE_MULT = 2.15
FRESH_RECAPTURE_MAX_SEND_FRACTION = 0.78
FRESH_RECAPTURE_GENERIC_VALUE_MULT = 1.55
FRESH_RECAPTURE_GENERIC_SCORE_MULT = 1.35
FRESH_RECAPTURE_HOLD_MARGIN = 3

# Soft cluster-target layer for Orbit Wars Lab testing.  This keeps the current
# W1-B planner's expansion/defense intact, but nudges target choice and capture
# margin toward high-production local anchors that public agents fight over.
CLUSTER_TARGET_ENABLED = True
CLUSTER_TARGET_START_STEP = 18
CLUSTER_TARGET_END_STEP = 260
CLUSTER_TARGET_RADIUS = 23.0
CLUSTER_TARGET_MIN_PRODUCTION = 3
CLUSTER_TARGET_MIN_SCORE = 1.15
CLUSTER_TARGET_FRONTIER_DISTANCE = 24.0
CLUSTER_TARGET_NEUTRAL_FRONTIER_DISTANCE = 18.0
CLUSTER_TARGET_VALUE_MULT = 1.11
CLUSTER_TARGET_STATIC_VALUE_MULT = 1.05
CLUSTER_TARGET_SCORE_MULT = 1.05
CLUSTER_TARGET_SEND_MARGIN_CAP = 4

# 1v1 mirror-slugfest tempo / anchor-hold capture layer.  The tamrazov-style
# loss is not an opening failure: both agents reach a high-mass symmetric
# midgame, then our planner flips prod-3/5 anchors with tiny surplus and loses
# them to in-flight counters.  Keep this gated to mirror slugfests so it does
# not disturb the current wins against asymmetric expanders.
ONEVONE_TEMPO_ENABLED = True
ONEVONE_TEMPO_START_STEP = 72
ONEVONE_TEMPO_END_STEP = 300
ONEVONE_TEMPO_MIN_MY_PROD = 16
ONEVONE_TEMPO_MIN_COMBINED_STRENGTH = 1400
ONEVONE_TEMPO_MAX_PROD_DELTA = 7
ONEVONE_TEMPO_MAX_PLANET_DELTA = 2
ONEVONE_TEMPO_MAX_TOTAL_DELTA_RATIO = 0.12
ONEVONE_TEMPO_MAX_TOTAL_DELTA_FLOOR = 170
ONEVONE_TEMPO_RESERVE_MULT = 0.58
ONEVONE_TEMPO_HIGH_PROD_VALUE_MULT = 1.22
ONEVONE_TEMPO_HIGH_PROD_SCORE_MULT = 1.20
ONEVONE_TEMPO_WEAK_GARRISON_MULT = 1.10
ONEVONE_HOLD_CAPTURE_MIN_PROD = 3
ONEVONE_HOLD_CAPTURE_TURNS = 10
ONEVONE_HOLD_CAPTURE_MAX_ETA = 34
ONEVONE_HOLD_CAPTURE_MARGIN_CAP = 7

# ============================================================
# Shared Types
# ============================================================

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


@dataclass(frozen=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[ShotOption] = field(default_factory=list)

# ============================================================
# Physics
# ============================================================

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def orbital_radius(planet):
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static_planet(planet):
    return orbital_radius(planet) + planet.radius >= ROTATION_LIMIT


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def launch_point(sx, sy, sr, angle):
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx, sy, sr, tx, ty, tr,
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance


def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )


def predict_comet_position(planet_id, comets, turns):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None


def comet_remaining_life(planet_id, comets):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def travel_time(sx, sy, sr, tx, ty, tr, ships):
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        return 10**9
    return est[1]


def predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids):
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def target_can_move(target, initial_by_id, comet_ids):
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    return r + init.radius < ROTATION_LIMIT


def search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    best = None
    best_score = None
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target, candidate_turns, initial_by_id, ang_vel, comets, comet_ids,
        )
        if pos is None:
            continue
        est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_target_position(
            target, actual_turns, initial_by_id, ang_vel, comets, comet_ids,
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x, src.y, src.radius, actual_pos[0], actual_pos[1], target.radius, ships,
        )
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue

        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        if not target_can_move(target, initial_by_id, comet_ids):
            return None
        return search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids,
        )

    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        pos = predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            if not target_can_move(target, initial_by_id, comet_ids):
                return None
            return search_safe_intercept(
                src, target, ships, initial_by_id, ang_vel, comets, comet_ids,
            )
        if (
            abs(ntx - tx) < 0.3
            and abs(nty - ty) < 0.3
            and abs(next_est[1] - turns) <= INTERCEPT_TOLERANCE
        ):
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids,
        )
    return final_est[0], final_est[1], tx, ty

# ============================================================
# World Model
# ============================================================

def fleet_target_planet(fleet, planets):
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def build_arrival_ledger(fleets, planets):
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets)
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def resolve_arrival_event(owner, garrison, arrivals):
    by_owner = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(arrivals, horizon):
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events


def simulate_planet_timeline(planet, arrivals, player, horizon):
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:
        def survives_with_keep(keep):
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(sim_owner, sim_garrison, group)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(timeline, arrival_turn):
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


def count_players(planets, fleets):
    owners = set()
    for planet in planets:
        if planet.owner != -1:
            owners.add(planet.owner)
    for fleet in fleets:
        owners.add(fleet.owner)
    return max(2, len(owners))


def nearest_distance_to_set(px, py, planets):
    if not planets:
        return 10**9
    return min(dist(px, py, planet.x, planet.y) for planet in planets)


def indirect_features(planet, planets, player):
    friendly = 0.0
    neutral = 0.0
    enemy = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            friendly += factor
        elif other.owner == -1:
            neutral += factor
        else:
            enemy += factor
    return friendly, neutral, enemy


def cluster_target_score(planet, planets, player):
    score = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d <= 0.5 or d > CLUSTER_TARGET_RADIUS:
            continue
        proximity = max(0.0, (CLUSTER_TARGET_RADIUS - d) / CLUSTER_TARGET_RADIUS)
        if other.owner == player:
            owner_weight = 0.80
        elif other.owner == -1:
            owner_weight = 1.00
        else:
            owner_weight = 1.15
        score += float(other.production) * owner_weight * proximity
    return score


def detect_exposed_enemy_planets(fleets, enemy_planets):
    exposed = set()
    for planet in enemy_planets:
        outbound = sum(
            int(f.ships)
            for f in fleets
            if f.owner == planet.owner and f.from_planet_id == planet.id and f.ships >= 5
        )
        if outbound >= 12 and outbound >= planet.ships * 0.8:  # lowered threshold
            exposed.add(planet.id)
    return exposed


def _compute_weakest_enemy(enemy_planets, owner_strength, owner_production):
    """Return the player ID of the weakest enemy (lowest ships + future production)."""
    enemy_owners = set(p.owner for p in enemy_planets)
    if not enemy_owners:
        return None
    return min(
        enemy_owners,
        key=lambda owner: owner_strength.get(owner, 0) + owner_production.get(owner, 0) * 15,
    )


class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)
        self.counter_profile = {}
        self.recent_enemy_captures = {}

        self.planet_by_id = {planet.id: planet for planet in planets}
        self.my_planets = [planet for planet in planets if planet.owner == player]
        self.enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        self.neutral_planets = [planet for planet in planets if planet.owner == -1]
        self.static_neutral_planets = [
            planet for planet in self.neutral_planets if is_static_planet(planet)
        ]

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < OPENING_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_total_war = self.remaining_steps < TOTAL_WAR_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        self.owner_planet_count = defaultdict(int)
        for planet in planets:
            if planet.owner != -1:
                self.owner_strength[planet.owner] += int(planet.ships)
                self.owner_production[planet.owner] += int(planet.production)
                self.owner_planet_count[planet.owner] += 1
        for fleet in fleets:
            self.owner_strength[fleet.owner] += int(fleet.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.my_planet_count = self.owner_planet_count.get(player, 0)
        self.enemy_total = sum(
            strength for owner, strength in self.owner_strength.items() if owner != player
        )
        self.max_enemy_strength = max(
            (strength for owner, strength in self.owner_strength.items() if owner != player),
            default=0,
        )
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(
            production
            for owner, production in self.owner_production.items()
            if owner != player
        )
        self.max_enemy_prod = max(
            (
                production
                for owner, production in self.owner_production.items()
                if owner != player
            ),
            default=0,
        )
        self.max_enemy_planets = max(
            (
                count
                for owner, count in self.owner_planet_count.items()
                if owner != player
            ),
            default=0,
        )

        # Weakest enemy tracking (key for 4P elimination strategy)
        self._weakest_enemy = _compute_weakest_enemy(
            self.enemy_planets, self.owner_strength, self.owner_production
        )
        self._weakest_enemy_strength = (
            self.owner_strength.get(self._weakest_enemy, 0)
            if self._weakest_enemy is not None else 0
        )

        self.arrivals_by_planet = build_arrival_ledger(fleets, planets)
        self.base_timeline = {
            planet.id: simulate_planet_timeline(
                planet,
                self.arrivals_by_planet[planet.id],
                player,
                HORIZON,
            )
            for planet in planets
        }
        self.keep_needed_map = {
            planet.id: self.base_timeline[planet.id]["keep_needed"] for planet in planets
        }
        self.min_owned_map = {
            planet.id: self.base_timeline[planet.id]["min_owned"] for planet in planets
        }
        self.first_enemy_map = {
            planet.id: self.base_timeline[planet.id]["first_enemy"] for planet in planets
        }
        self.fall_turn_map = {
            planet.id: self.base_timeline[planet.id]["fall_turn"] for planet in planets
        }
        self.holds_full_map = {
            planet.id: self.base_timeline[planet.id]["holds_full"] for planet in planets
        }
        self.indirect_feature_map = {
            planet.id: indirect_features(planet, planets, player) for planet in planets
        }
        self.cluster_target_score_map = {
            planet.id: cluster_target_score(planet, planets, player) for planet in planets
        }
        self.cluster_target_ids = set()
        self.frontier_target_ids = set()
        for planet in planets:
            if planet.id in self.comet_ids or int(planet.production) < CLUSTER_TARGET_MIN_PRODUCTION:
                continue
            local_score = self.cluster_target_score_map.get(planet.id, 0.0)
            if (
                local_score >= CLUSTER_TARGET_MIN_SCORE
                or (is_static_planet(planet) and int(planet.production) >= CLUSTER_TARGET_MIN_PRODUCTION + 1)
            ):
                self.cluster_target_ids.add(planet.id)
            nearest_enemy = nearest_distance_to_set(planet.x, planet.y, self.enemy_planets)
            nearest_neutral = nearest_distance_to_set(planet.x, planet.y, self.neutral_planets)
            if (
                nearest_enemy <= CLUSTER_TARGET_FRONTIER_DISTANCE
                or nearest_neutral <= CLUSTER_TARGET_NEUTRAL_FRONTIER_DISTANCE
            ):
                self.frontier_target_ids.add(planet.id)

        # Opponent proximity map: in 4P, nearby opponents are the ones who can punish
        # our frontier soonest. Use this as a soft target-priority multiplier instead
        # of blindly attacking every enemy with the same weight.
        self.opp_planets = defaultdict(list)
        for enemy in self.enemy_planets:
            self.opp_planets[enemy.owner].append(enemy)
        self.opp_min_dist = {}
        for owner, owner_planets in self.opp_planets.items():
            self.opp_min_dist[owner] = min(
                (dist(my.x, my.y, enemy.x, enemy.y) for my in self.my_planets for enemy in owner_planets),
                default=999.0,
            )

        self.exposed_planet_ids = detect_exposed_enemy_planets(fleets, self.enemy_planets)
        self.shot_cache = {}
        self.probe_candidate_cache = {}
        self.best_probe_cache = {}
        self.reaction_cache = {}
        self.exact_need_cache = {}

        self.total_visible_ships = sum(int(planet.ships) for planet in planets) + sum(
            int(fleet.ships) for fleet in fleets
        )
        self.total_production = sum(int(planet.production) for planet in planets)

    def opponent_priority(self, owner):
        if not self.opp_min_dist:
            return 1.0
        my_dist = self.opp_min_dist.get(owner, 999.0)
        distances = list(self.opp_min_dist.values())
        near = min(distances)
        far = max(distances)
        spread = max(1.0, far - near)
        norm = (my_dist - near) / spread
        proximity = 1.22 - 0.42 * norm
        total_enemy = sum(self.owner_strength.get(o, 0) for o in self.opp_min_dist) or 1
        strength_share = self.owner_strength.get(owner, 0) / total_enemy
        return max(0.82, min(1.28, proximity + strength_share * 0.08))

    def is_static(self, planet_id):
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        return comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id, spent_total):
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def plan_shot(self, src_id, target_id, ships):
        ships = int(ships)
        key = (src_id, target_id, ships)
        if key in self.shot_cache:
            return self.shot_cache[key]
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        result = aim_with_prediction(
            src, target, ships, self.initial_by_id, self.ang_vel, self.comets, self.comet_ids,
        )
        self.shot_cache[key] = result
        return result

    def probe_ship_candidates(self, src_id, target_id, source_cap, hints=()):
        source_cap = max(1, int(source_cap))
        normalized_hints = tuple(
            int(math.ceil(hint)) for hint in hints if hint is not None
        )
        cache_key = (src_id, target_id, source_cap, normalized_hints)
        cached = self.probe_candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        target = self.planet_by_id[target_id]
        target_ships = max(1, int(math.ceil(target.ships)))

        values = set(range(1, min(6, source_cap) + 1))
        values.update({
            source_cap,
            max(1, source_cap // 2),
            max(1, source_cap // 3),
            min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
            min(source_cap, target_ships + 1),
            min(source_cap, target_ships + 2),
            min(source_cap, target_ships + 4),
            min(source_cap, target_ships + 8),
        })

        for hint in normalized_hints:
            base = max(1, min(source_cap, hint))
            for delta in (-2, -1, 0, 1, 2):
                candidate = base + delta
                if 1 <= candidate <= source_cap:
                    values.add(candidate)

        result = sorted(values)
        self.probe_candidate_cache[cache_key] = result
        return result

    def best_probe_aim(
        self,
        src_id,
        target_id,
        source_cap,
        hints=(),
        min_turn=None,
        max_turn=None,
        anchor_turn=None,
        max_anchor_diff=None,
    ):
        cache_key = (
            src_id,
            target_id,
            max(1, int(source_cap)),
            tuple(hints),
            min_turn,
            max_turn,
            anchor_turn,
            max_anchor_diff,
        )
        if cache_key in self.best_probe_cache:
            return self.best_probe_cache[cache_key]

        best = None
        best_key = None

        for ships in self.probe_ship_candidates(src_id, target_id, source_cap, hints=hints):
            aim = self.plan_shot(src_id, target_id, ships)
            if aim is None:
                continue

            angle, turns, dist_to_target, path_target = aim
            if min_turn is not None and turns < min_turn:
                continue
            if max_turn is not None and turns > max_turn:
                continue
            if (
                anchor_turn is not None
                and max_anchor_diff is not None
                and abs(turns - anchor_turn) > max_anchor_diff
            ):
                continue

            if anchor_turn is None:
                key = (turns, ships)
            else:
                key = (abs(turns - anchor_turn), turns, ships)

            if best_key is None or key < best_key:
                best_key = key
                best = (ships, (angle, turns, dist_to_target, path_target))

        self.best_probe_cache[cache_key] = best
        return best

    def reaction_times(self, target_id):
        cached = self.reaction_cache.get(target_id)
        if cached is not None:
            return cached

        target = self.planet_by_id[target_id]
        my_t = 10**9
        for planet in self.my_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            my_t = min(my_t, aim[1])

        enemy_t = 10**9
        for planet in self.enemy_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            enemy_t = min(enemy_t, aim[1])

        cached = (my_t, enemy_t)
        self.reaction_cache[target_id] = cached
        return cached

    def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)

        arrivals = [
            item
            for item in self.arrivals_by_planet.get(target_id, [])
            if item[0] <= cutoff
        ]
        arrivals.extend(
            item
            for item in planned_commitments.get(target_id, [])
            if item[0] <= cutoff
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)

        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return state_at_timeline(dyn, cutoff)

    def projected_timeline(self, target_id, horizon, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [
            item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= horizon
        ]
        arrivals.extend(
            item for item in planned_commitments.get(target_id, []) if item[0] <= horizon
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        target = self.planet_by_id[target_id]
        return simulate_planet_timeline(target, arrivals, self.player, horizon)

    def hold_status(self, target_id, planned_commitments=None, horizon=HORIZON):
        planned_commitments = planned_commitments or {}
        if planned_commitments.get(target_id):
            tl = self.projected_timeline(
                target_id, horizon, planned_commitments=planned_commitments,
            )
        else:
            tl = self.base_timeline[target_id]
        return {
            "keep_needed": tl["keep_needed"],
            "min_owned": tl["min_owned"],
            "first_enemy": tl["first_enemy"],
            "fall_turn": tl["fall_turn"],
            "holds_full": tl["holds_full"],
        }

    def _ownership_search_cap(self, eval_turn):
        productive_cap = self.total_production * max(2, eval_turn + 2)
        return max(32, int(self.total_visible_ships + productive_cap + 32))

    def min_ships_to_own_by(
        self,
        target_id,
        eval_turn,
        attacker_owner,
        arrival_turn=None,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            if upper_bound is not None:
                return max(1, int(upper_bound)) + 1
            return self._ownership_search_cap(eval_turn) + 1

        normalized_extra = tuple(
            (max(1, int(math.ceil(turns))), owner, int(ships))
            for turns, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
        )

        cache_key = None
        if (
            arrival_turn == eval_turn
            and not planned_commitments.get(target_id)
            and not normalized_extra
        ):
            cache_key = (target_id, eval_turn, attacker_owner)
            cached = self.exact_need_cache.get(cache_key)
            if cached is not None:
                return cached

        owner_before, ships_before = self.projected_state(
            target_id,
            eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=normalized_extra,
        )
        if owner_before == attacker_owner:
            if cache_key is not None:
                self.exact_need_cache[cache_key] = 0
            return 0

        def owns_at(ships):
            owner_after, _ = self.projected_state(
                target_id,
                eval_turn,
                planned_commitments=planned_commitments,
                extra_arrivals=normalized_extra + ((arrival_turn, attacker_owner, int(ships)),),
            )
            return owner_after == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_at(hi):
                return hi + 1
        else:
            hi = max(1, int(math.ceil(ships_before)) + 1)
            search_cap = self._ownership_search_cap(eval_turn)
            while hi <= search_cap and not owns_at(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not owns_at(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid):
                hi = mid
            else:
                lo = mid + 1

        if cache_key is not None:
            self.exact_need_cache[cache_key] = lo
        return lo

    def min_ships_to_own_at(
        self,
        target_id,
        arrival_turn,
        attacker_owner,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        return self.min_ships_to_own_by(
            target_id,
            arrival_turn,
            attacker_owner,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        planet_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        upper_bound=None,
    ):
        planned_commitments = planned_commitments or {}
        target = self.planet_by_id[planet_id]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        if target.owner != self.player:
            return self.min_ships_to_own_by(
                planet_id,
                hold_until,
                self.player,
                arrival_turn=arrival_turn,
                planned_commitments=planned_commitments,
                upper_bound=upper_bound,
            )

        def holds_with_reinforcement(ships):
            timeline = self.projected_timeline(
                planet_id,
                hold_until,
                planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, self.player, int(ships)),),
            )
            for turn in range(arrival_turn, hold_until + 1):
                if timeline["owner_at"].get(turn) != self.player:
                    return False
            return True

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with_reinforcement(hi):
                return hi + 1
        else:
            hi = 1
            search_cap = self._ownership_search_cap(hold_until)
            while hi <= search_cap and not holds_with_reinforcement(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not holds_with_reinforcement(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with_reinforcement(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def ships_needed_to_capture(
        self,
        target_id,
        arrival_turn,
        planned_commitments=None,
        extra_arrivals=(),
    ):
        return self.min_ships_to_own_at(
            target_id,
            arrival_turn,
            self.player,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
        )

# ============================================================
# Strategy
# ============================================================

def planet_distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def nearest_sources_to_target(target, sources, top_k):
    if top_k <= 0 or len(sources) <= top_k:
        return sources
    return sorted(
        sources,
        key=lambda src: (planet_distance(src, target), -int(src.ships), src.id),
    )[:top_k]


def min_legal_reaction_time(target, sources, world):
    best = 10**9
    for src in sources:
        seeded = world.best_probe_aim(src.id, target.id, max(1, int(src.ships)))
        if seeded is None:
            continue
        _, aim = seeded
        best = min(best, aim[1])
    return best


def policy_reaction_times(target_id, policy):
    return policy["reaction_time_map"].get(target_id, (10**9, 10**9))


def candidate_time_valid(target, turns, world, remaining_buffer):
    if turns > world.remaining_steps - remaining_buffer:
        return False
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        if turns >= life or turns > COMET_MAX_CHASE_TURNS:
            return False
        if (
            world.counter_profile.get("late_swarm_danger")
            or world.counter_profile.get("comet_bait_chaser")
        ):
            if life - turns < COUNTER_MIN_COMET_LIFE:
                return False
    return True


def stacked_enemy_proactive_keep(planet, world):
    horizon = MULTI_ENEMY_PROACTIVE_HORIZON_4P if world.is_four_player else MULTI_ENEMY_PROACTIVE_HORIZON
    stack_window = MULTI_ENEMY_STACK_WINDOW_4P if world.is_four_player else MULTI_ENEMY_STACK_WINDOW
    ratio = MULTI_ENEMY_PROACTIVE_RATIO_4P if world.is_four_player else MULTI_ENEMY_PROACTIVE_RATIO
    threats = []
    for enemy in world.enemy_planets:
        seeded = world.best_probe_aim(enemy.id, planet.id, max(1, int(enemy.ships)))
        if seeded is None:
            continue
        _, aim = seeded
        eta = aim[1]
        if eta > horizon:
            continue
        threats.append((eta, int(enemy.ships)))

    if not threats:
        return 0

    threats.sort()
    best_stacked = 0
    left = 0
    running = 0
    for right in range(len(threats)):
        running += threats[right][1]
        while threats[right][0] - threats[left][0] > stack_window:
            running -= threats[left][1]
            left += 1
        best_stacked = max(best_stacked, running)

    return int(best_stacked * ratio)


def swarm_eta_tolerance(options, target, world):
    if len(options) >= 3:
        return THREE_SOURCE_ETA_TOLERANCE
    if target.owner not in (-1, world.player):
        return HOSTILE_SWARM_ETA_TOLERANCE
    return MULTI_SOURCE_ETA_TOLERANCE


def detect_enemy_crashes(world):
    crashes = []
    for target_id, arrivals in world.arrivals_by_planet.items():
        enemy_events = [
            (int(math.ceil(eta)), owner, int(ships))
            for eta, owner, ships in arrivals
            if owner not in (-1, world.player) and ships > 0
        ]
        enemy_events.sort()
        for i in range(len(enemy_events)):
            eta_a, owner_a, ships_a = enemy_events[i]
            for j in range(i + 1, len(enemy_events)):
                eta_b, owner_b, ships_b = enemy_events[j]
                if owner_a == owner_b:
                    continue
                if abs(eta_a - eta_b) > CRASH_EXPLOIT_ETA_WINDOW:
                    break
                if ships_a + ships_b < CRASH_EXPLOIT_MIN_TOTAL_SHIPS:
                    continue
                crashes.append({
                    "target_id": target_id,
                    "crash_turn": max(eta_a, eta_b),
                    "owners": (owner_a, owner_b),
                    "ships": (ships_a, ships_b),
                })
    return crashes


def detect_enemy_planet_battles(world):
    """
    Detect where an enemy fleet is attacking another enemy's owned planet.
    These create attack windows — the winning side will be weakened.
    Returns list of dicts with target_id, battle_turn, estimated post_battle_ships.
    """
    battles = []
    for target in world.enemy_planets:
        attacking_fleets = [
            (int(math.ceil(eta)), owner, int(ships))
            for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if owner not in (-1, world.player) and owner != target.owner and ships > 0
        ]
        if not attacking_fleets:
            continue

        for eta, attacker_owner, attacker_ships in attacking_fleets:
            # Rough post-battle estimate (production growth + garrison vs attacker)
            garrison_at_eta = target.ships + target.production * eta
            surviving_attacker = max(0, attacker_ships - garrison_at_eta)
            surviving_defender = max(0, garrison_at_eta - attacker_ships)

            if attacker_ships > garrison_at_eta:
                post_owner = attacker_owner
                post_ships = surviving_attacker
            else:
                post_owner = target.owner
                post_ships = surviving_defender

            # Only interesting if planet will be weak after battle
            if post_ships < 25:
                battles.append({
                    "target_id": target.id,
                    "battle_turn": eta,
                    "post_owner": post_owner,
                    "post_ships": post_ships,
                    "original_owner": target.owner,
                })

    return battles


def onevone_tempo_active(world):
    if not ONEVONE_TEMPO_ENABLED:
        return False
    if getattr(world, "is_four_player", False) or world.is_total_war:
        return False
    if world.step < ONEVONE_TEMPO_START_STEP or world.step > ONEVONE_TEMPO_END_STEP:
        return False
    if not world.enemy_planets or world.my_prod < ONEVONE_TEMPO_MIN_MY_PROD:
        return False
    combined_strength = world.my_total + world.max_enemy_strength
    if combined_strength < ONEVONE_TEMPO_MIN_COMBINED_STRENGTH:
        return False
    if abs(world.my_prod - world.max_enemy_prod) > ONEVONE_TEMPO_MAX_PROD_DELTA:
        return False
    if abs(world.my_planet_count - world.max_enemy_planets) > ONEVONE_TEMPO_MAX_PLANET_DELTA:
        return False
    total_delta_limit = max(
        ONEVONE_TEMPO_MAX_TOTAL_DELTA_FLOOR,
        int(combined_strength * ONEVONE_TEMPO_MAX_TOTAL_DELTA_RATIO),
    )
    return abs(world.my_total - world.max_enemy_strength) <= total_delta_limit


def onevone_hold_capture_target(target, arrival_turns, world):
    if not onevone_tempo_active(world):
        return False
    if target.owner == world.player:
        return False
    if target.id in world.comet_ids:
        return False
    if int(target.production) < ONEVONE_HOLD_CAPTURE_MIN_PROD:
        return False
    if int(math.ceil(arrival_turns)) > ONEVONE_HOLD_CAPTURE_MAX_ETA:
        return False
    return True


def onevone_capture_eval_turn(target, arrival_turns, world):
    arrival = int(math.ceil(arrival_turns))
    if not onevone_hold_capture_target(target, arrival, world):
        return arrival
    return min(HORIZON, arrival + ONEVONE_HOLD_CAPTURE_TURNS)


def onevone_capture_eval_fn(target, world):
    return lambda turns, target=target, world=world: onevone_capture_eval_turn(target, turns, world)


def onevone_ships_to_own_capture(target, arrival_turns, world, planned_commitments, upper_bound=None):
    arrival = max(1, int(math.ceil(arrival_turns)))
    eval_turn = onevone_capture_eval_turn(target, arrival, world)
    if eval_turn <= arrival:
        return world.min_ships_to_own_at(
            target.id,
            arrival,
            world.player,
            planned_commitments=planned_commitments,
            upper_bound=upper_bound,
        )
    return world.min_ships_to_own_by(
        target.id,
        eval_turn,
        world.player,
        arrival_turn=arrival,
        planned_commitments=planned_commitments,
        upper_bound=upper_bound,
    )


def build_policy_state(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    indirect_wealth_map = {}
    for target_id, features in world.indirect_feature_map.items():
        friendly, neutral, enemy = features
        indirect_wealth_map[target_id] = (
            friendly * INDIRECT_FRIENDLY_WEIGHT
            + neutral * INDIRECT_NEUTRAL_WEIGHT
            + enemy * INDIRECT_ENEMY_WEIGHT
        )

    reserve = {}
    attack_budget = {}
    reaction_time_map = {}
    recovery_pressure = recovery_pressure_active(world)

    for target in world.planets:
        if expired():
            break
        if target.owner == world.player:
            continue
        my_sources = nearest_sources_to_target(target, world.my_planets, REACTION_SOURCE_TOP_K_MY)
        enemy_sources = nearest_sources_to_target(target, world.enemy_planets, REACTION_SOURCE_TOP_K_ENEMY)
        my_t = min_legal_reaction_time(target, my_sources, world)
        enemy_t = min_legal_reaction_time(target, enemy_sources, world)
        reaction_time_map[target.id] = (my_t, enemy_t)

    for planet in world.my_planets:
        if expired():
            break
        exact_keep = world.keep_needed_map.get(planet.id, 0)

        proactive_keep = 0
        for enemy in nearest_sources_to_target(planet, world.enemy_planets, PROACTIVE_ENEMY_TOP_K):
            enemy_aim = world.plan_shot(enemy.id, planet.id, max(1, int(enemy.ships)))
            if enemy_aim is None:
                continue
            enemy_eta = enemy_aim[1]
            if enemy_eta > PROACTIVE_DEFENSE_HORIZON:
                continue
            eta_discount = max(
                0.5,
                1.0 - (enemy_eta / max(1, PROACTIVE_DEFENSE_HORIZON)) * 0.5,
            )
            proactive_keep = max(
                proactive_keep,
                int(enemy.ships * PROACTIVE_DEFENSE_RATIO * eta_discount),
            )
        stacked_keep = stacked_enemy_proactive_keep(planet, world)
        proactive_keep = max(proactive_keep, stacked_keep)
        if recovery_pressure:
            if (
                world.max_enemy_prod - world.my_prod >= RECOVERY_SEVERE_PROD_GAP
                or world.max_enemy_planets - world.my_planet_count >= RECOVERY_SEVERE_PLANET_GAP
            ):
                exact_keep = int(math.floor(exact_keep * RECOVERY_SEVERE_EXACT_KEEP_MULT))
            proactive_keep = int(math.floor(proactive_keep * RECOVERY_RESERVE_MULT))
        elif onevone_tempo_active(world):
            # Exact visible defense still wins, but release anticipatory reserve
            # so front planets keep sending in the midgame instead of waiting.
            fall_turn = world.fall_turn_map.get(planet.id)
            if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
                proactive_keep = int(math.floor(proactive_keep * ONEVONE_TEMPO_RESERVE_MULT))
        if (
            world.counter_profile.get("late_swarm_danger")
            and int(planet.production) >= 2
            and not recovery_pressure
        ):
            proactive_keep = int(math.ceil(proactive_keep * COUNTER_LATE_SWARM_DEFENSE_MULT))

        if world.is_total_war:
            exact_keep = min(exact_keep, max(1, exact_keep // 2))
            proactive_keep = min(proactive_keep, max(1, proactive_keep // 2))

        reserve[planet.id] = min(int(planet.ships), max(exact_keep, proactive_keep))
        attack_budget[planet.id] = max(0, int(planet.ships) - reserve[planet.id])

    return {
        "indirect_wealth_map": indirect_wealth_map,
        "reserve": reserve,
        "attack_budget": attack_budget,
        "reaction_time_map": reaction_time_map,
    }


def build_modes(world):
    domination = (world.my_total - world.enemy_total) / max(1, world.my_total + world.enemy_total)
    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead or (
        world.max_enemy_strength > 0 and world.my_total > world.max_enemy_strength * 1.25
    )
    is_finishing = (
        domination > FINISHING_DOMINATION
        and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
        and world.step > 80
    )

    attack_margin_mult = 1.0
    if is_ahead:
        attack_margin_mult += AHEAD_ATTACK_MARGIN_BONUS
    if is_behind:
        attack_margin_mult -= BEHIND_ATTACK_MARGIN_PENALTY
    if is_finishing:
        attack_margin_mult += FINISHING_ATTACK_MARGIN_BONUS

    return {
        "domination": domination,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": attack_margin_mult,
    }


def is_safe_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN


def is_contested_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN


def recovery_pressure_active(world):
    """True when we need to spend ships to recover map control."""
    if world.step < RECOVERY_PRESSURE_START_STEP or not world.enemy_planets:
        return False
    prod_gap = world.max_enemy_prod - world.my_prod
    planet_gap = world.max_enemy_planets - world.my_planet_count
    return prod_gap >= RECOVERY_PROD_GAP or planet_gap >= RECOVERY_PLANET_GAP


def bad_early_trickle_target(target, arrival_turns, send_cap, world):
    """Skip long, low-prod partial captures that were causing seed-102 stalls."""
    if world.step >= BAD_TRICKLE_TURN_LIMIT:
        return False
    if target.owner != -1 or target.id in world.comet_ids:
        return False
    if target.production > BAD_TRICKLE_MAX_PROD:
        return False
    if arrival_turns < BAD_TRICKLE_MIN_ETA:
        return False
    return int(target.ships) - int(send_cap) >= BAD_TRICKLE_MIN_DEFICIT


def recently_contested_high_prod(target, world):
    if target.production < CONTESTED_HIGH_PROD_MIN:
        return False
    for eta, owner, ships in world.arrivals_by_planet.get(target.id, []):
        if eta <= CONTESTED_HIGH_PROD_LOOKBACK and owner != target.owner and ships > 0:
            return True
    fall_turn = world.fall_turn_map.get(target.id)
    return fall_turn is not None and fall_turn <= CONTESTED_HIGH_PROD_LOOKBACK


def fresh_enemy_capture_event(target, world):
    """Return a recent enemy-capture event if this is a narrow W1-B target."""
    if not FRESH_RECAPTURE_ENABLED:
        return None
    if not getattr(world, "is_four_player", False):
        return None
    if world.step < FRESH_RECAPTURE_START_STEP:
        return None
    if target.owner in (-1, world.player):
        return None
    if target.id in world.comet_ids:
        return None
    if target.production < FRESH_RECAPTURE_MIN_PRODUCTION:
        return None
    event = getattr(world, "recent_enemy_captures", {}).get(int(target.id))
    if not event:
        return None
    if int(event.get("owner", -99)) != int(target.owner):
        return None
    age = int(world.step) - int(event.get("step", world.step))
    if age < 0 or age > FRESH_RECAPTURE_LOOKBACK:
        return None
    max_garrison = max(
        FRESH_RECAPTURE_MAX_GARRISON,
        int(target.production) * FRESH_RECAPTURE_GARRISON_PROD_MULT,
    )
    if int(target.ships) > max_garrison:
        return None
    return event


def cluster_target_window_active(world):
    if not CLUSTER_TARGET_ENABLED or world.is_total_war:
        return False
    return CLUSTER_TARGET_START_STEP <= world.step <= CLUSTER_TARGET_END_STEP


def is_cluster_target(target, world):
    if not cluster_target_window_active(world):
        return False
    if target.owner == -1:
        return False
    if target.id in world.comet_ids:
        return False
    if int(target.production) < CLUSTER_TARGET_MIN_PRODUCTION:
        return False
    if target.id in getattr(world, "cluster_target_ids", set()):
        return True
    return int(target.production) >= 4 and target.id in getattr(world, "frontier_target_ids", set())


def opening_filter(target, arrival_turns, needed, src_available, world, policy):
    if not world.is_opening or target.owner != -1:
        return False
    if bad_early_trickle_target(target, arrival_turns, src_available, world):
        return True
    # In losing maps the bot repeatedly sent underpowered opening fleets across
    # 30+ turn lanes, then reached the target still short and lost tempo.  In
    # 1v1 openings, wait until a source can actually cover the needed capture
    # instead of trickling partial fleets into long neutral targets.
    if (
        not world.is_four_player
        and world.step < OPENING_UNDERPOWERED_TURN_LIMIT
        and target.id not in world.comet_ids
        and arrival_turns >= OPENING_UNDERPOWERED_MIN_ETA
        and target.production <= OPENING_UNDERPOWERED_MAX_PROD
        and int(needed) > int(src_available)
        and int(target.ships) - int(src_available) >= OPENING_UNDERPOWERED_MIN_DEFICIT
    ):
        return True
    if target.id in world.comet_ids:
        return False
    if world.is_static(target.id):
        return False

    my_t, enemy_t = policy_reaction_times(target.id, policy)
    reaction_gap = enemy_t - my_t
    if (
        target.production >= SAFE_OPENING_PROD_THRESHOLD
        and arrival_turns <= SAFE_OPENING_TURN_LIMIT
        and reaction_gap >= SAFE_NEUTRAL_MARGIN
    ):
        return False

    if world.is_four_player:
        affordable = needed <= max(
            PARTIAL_SOURCE_MIN_SHIPS,
            int(src_available * FOUR_PLAYER_ROTATING_SEND_RATIO),
        )
        if (
            affordable
            and arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT
            and reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP
        ):
            return False
        return True

    return arrival_turns > ROTATING_OPENING_MAX_TURNS or target.production <= ROTATING_OPENING_LOW_PROD


def target_value(target, arrival_turns, mission, world, modes, policy):
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit
    value += policy["indirect_wealth_map"][target.id] * turns_profit * INDIRECT_VALUE_SCALE

    if world.is_static(target.id):
        value *= STATIC_NEUTRAL_VALUE_MULT if target.owner == -1 else STATIC_HOSTILE_VALUE_MULT
    else:
        value *= ROTATING_OPENING_VALUE_MULT if world.is_opening else 1.0

    if is_cluster_target(target, world):
        value *= CLUSTER_TARGET_VALUE_MULT
        if world.is_static(target.id):
            value *= CLUSTER_TARGET_STATIC_VALUE_MULT

    if target.owner not in (-1, world.player):
        base_mult = OPENING_HOSTILE_TARGET_VALUE_MULT if world.is_opening else HOSTILE_TARGET_VALUE_MULT
        value *= base_mult * world.opponent_priority(target.owner)
        if world.counter_profile.get("passive_mirror"):
            value *= COUNTER_PASSIVE_HOSTILE_MULT

    if target.owner == -1:
        if is_safe_neutral(target, policy):
            value *= SAFE_NEUTRAL_VALUE_MULT
            if world.counter_profile.get("passive_mirror"):
                value *= COUNTER_PASSIVE_NEUTRAL_MULT
        elif is_contested_neutral(target, policy):
            value *= CONTESTED_NEUTRAL_VALUE_MULT
        if world.is_early:
            value *= EARLY_NEUTRAL_VALUE_MULT
        if (
            recovery_pressure_active(world)
            and target.production >= RECOVERY_MIN_TARGET_PROD
            and arrival_turns <= RECOVERY_MAX_ETA
        ):
            value *= RECOVERY_TARGET_VALUE_MULT

    if target.id in world.comet_ids:
        value *= COMET_VALUE_MULT
        if (
            world.counter_profile.get("late_swarm_danger")
            or world.counter_profile.get("comet_bait_chaser")
        ):
            value *= COUNTER_LATE_SWARM_COMET_MULT

    if mission == "snipe":
        value *= SNIPE_VALUE_MULT
    elif mission == "swarm":
        value *= SWARM_VALUE_MULT
    elif mission == "reinforce":
        value *= REINFORCE_VALUE_MULT
    elif mission == "crash_exploit":
        value *= CRASH_EXPLOIT_VALUE_MULT

    if target.id in world.exposed_planet_ids:
        value *= EXPOSED_PLANET_VALUE_MULT
        if world.counter_profile.get("late_swarmer"):
            value *= COUNTER_LATE_SWARM_EXPOSED_ENEMY_MULT

    if (
        recovery_pressure_active(world)
        and target.owner not in (-1, world.player)
        and target.production >= RECOVERY_MIN_TARGET_PROD
        and arrival_turns <= RECOVERY_MAX_ETA
    ):
        value *= RECOVERY_HIGH_PROD_MULT

    if recently_contested_high_prod(target, world):
        value *= CONTESTED_HIGH_PROD_VALUE_MULT

    if (
        onevone_tempo_active(world)
        and target.owner != world.player
        and target.id not in world.comet_ids
        and int(target.production) >= ONEVONE_HOLD_CAPTURE_MIN_PROD
    ):
        value *= ONEVONE_TEMPO_HIGH_PROD_VALUE_MULT
        if target.owner not in (-1, world.player) and int(target.ships) <= int(target.production) * 28:
            value *= ONEVONE_TEMPO_WEAK_GARRISON_MULT

    fresh_capture = fresh_enemy_capture_event(target, world)
    if fresh_capture and arrival_turns <= FRESH_RECAPTURE_MAX_ETA + 4:
        age = max(0, int(world.step) - int(fresh_capture.get("step", world.step)))
        value *= FRESH_RECAPTURE_GENERIC_VALUE_MULT
        value *= 1.0 + max(0, FRESH_RECAPTURE_LOOKBACK - age) * 0.02

    if (
        world.counter_profile.get("late_swarmer")
        and target.owner not in (-1, world.player)
        and target.ships <= max(20, target.production * 8)
    ):
        value *= 1.12

    if world.is_late:
        value += max(0, target.ships) * LATE_IMMEDIATE_SHIP_VALUE

    # Elimination bonus — applies whenever enemy is weak (not just late game)
    if target.owner not in (-1, world.player):
        enemy_strength = world.owner_strength.get(target.owner, 0)
        weak_threshold = WEAK_ENEMY_THRESHOLD_4P if world.is_four_player else WEAK_ENEMY_THRESHOLD
        elimination_bonus = ELIMINATION_BONUS_4P if world.is_four_player else ELIMINATION_BONUS
        if enemy_strength <= weak_threshold:
            value += elimination_bonus

    # Weakest enemy targeting bonus (key for 4P strategy)
    if target.owner not in (-1, world.player) and world._weakest_enemy is not None:
        if target.owner == world._weakest_enemy:
            mult = WEAKEST_ENEMY_VALUE_MULT_4P if world.is_four_player else WEAKEST_ENEMY_VALUE_MULT_2P
            value *= mult

    if modes["is_finishing"] and target.owner not in (-1, world.player):
        value *= FINISHING_HOSTILE_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and not world.is_static(target.id):
        value *= BEHIND_ROTATING_NEUTRAL_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and is_safe_neutral(target, policy):
        value *= 1.08
    if modes["is_dominating"] and target.owner == -1 and is_contested_neutral(target, policy):
        value *= 0.92

    return value


def reinforce_value(target, hold_until, world, policy):
    saved_turns = max(1, world.remaining_steps - hold_until)
    value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
    if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
        value *= DEFENSE_FRONTIER_SCORE_MULT
    value += policy["indirect_wealth_map"][target.id] * saved_turns * INDIRECT_VALUE_SCALE * 0.35
    return value * REINFORCE_VALUE_MULT


def preferred_send(target, base_needed, arrival_turns, src_available, world, modes, policy):
    send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    margin = 0
    if target.owner == -1:
        margin += min(
            NEUTRAL_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT,
        )
    else:
        margin += min(
            HOSTILE_MARGIN_CAP,
            HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT,
        )
        reinforce_est = 0
        for enemy_source in world.opp_planets.get(target.owner, []):
            if enemy_source.id == target.id:
                continue
            aim = world.plan_shot(enemy_source.id, target.id, max(1, int(enemy_source.ships)))
            if aim is None:
                continue
            if aim[1] <= arrival_turns + HOSTILE_REINFORCE_HORIZON:
                reinforce_est += max(0, int(enemy_source.ships) - 3)
        margin += min(HOSTILE_REINFORCE_CAP, int(reinforce_est * HOSTILE_REINFORCE_RATIO))
    if world.is_static(target.id):
        margin += STATIC_TARGET_MARGIN
    if is_contested_neutral(target, policy):
        margin += CONTESTED_TARGET_MARGIN
    if world.is_four_player:
        margin += FOUR_PLAYER_TARGET_MARGIN
    if arrival_turns > LONG_TRAVEL_MARGIN_START:
        margin += min(LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR)
    if target.id in world.comet_ids:
        margin = max(0, margin - COMET_MARGIN_RELIEF)
    if modes["is_finishing"] and target.owner not in (-1, world.player):
        margin += FINISHING_HOSTILE_SEND_BONUS
    if target.id in world.exposed_planet_ids:
        margin = max(0, margin - 2)
    if fresh_enemy_capture_event(target, world):
        margin += FRESH_RECAPTURE_HOLD_MARGIN + min(3, int(target.production))
    if onevone_hold_capture_target(target, arrival_turns, world):
        margin += min(
            ONEVONE_HOLD_CAPTURE_MARGIN_CAP,
            2 + int(target.production),
        )
    if is_cluster_target(target, world):
        cluster_margin = 1 + int(target.production) // 2 + int(world.cluster_target_score_map.get(target.id, 0.0))
        margin += min(CLUSTER_TARGET_SEND_MARGIN_CAP, cluster_margin)
    # In 4P, be more efficient when targeting weakest enemy (finishing blow)
    if world.is_four_player and world._weakest_enemy and target.owner == world._weakest_enemy:
        margin = max(0, margin - 2)
    if (
        world.counter_profile.get("rotator_racer")
        and target.owner not in (-1, world.player)
        and target.production >= 2
        and not world.is_static(target.id)
    ):
        margin += COUNTER_ROTATOR_HOLD_MARGIN
    return min(src_available, send + margin)


def apply_score_modifiers(base_score, target, mission, world):
    score = base_score
    if world.is_static(target.id):
        score *= STATIC_TARGET_SCORE_MULT
    if world.is_early and target.owner == -1 and world.is_static(target.id):
        score *= EARLY_STATIC_NEUTRAL_SCORE_MULT
    if world.is_four_player and target.owner == -1 and not world.is_static(target.id):
        score *= FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT
    if (
        len(world.static_neutral_planets) >= DENSE_STATIC_NEUTRAL_COUNT
        and target.owner == -1
        and not world.is_static(target.id)
    ):
        score *= DENSE_ROTATING_NEUTRAL_SCORE_MULT
    if mission == "snipe":
        score *= SNIPE_SCORE_MULT
    elif mission == "swarm":
        score *= SWARM_SCORE_MULT
    elif mission == "crash_exploit":
        score *= CRASH_EXPLOIT_SCORE_MULT
    if is_cluster_target(target, world):
        score *= CLUSTER_TARGET_SCORE_MULT
    if (
        onevone_tempo_active(world)
        and target.owner != world.player
        and target.id not in world.comet_ids
        and int(target.production) >= ONEVONE_HOLD_CAPTURE_MIN_PROD
    ):
        score *= ONEVONE_TEMPO_HIGH_PROD_SCORE_MULT
    if target.id in world.exposed_planet_ids:
        score *= 1.25
    # Bonus score for targeting weakest enemy planet
    if target.owner not in (-1, world.player) and world._weakest_enemy == target.owner:
        score *= 1.2
    if fresh_enemy_capture_event(target, world):
        score *= FRESH_RECAPTURE_GENERIC_SCORE_MULT
    return score


def settle_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    modes,
    policy,
    mission="capture",
    eval_turn_fn=None,
    anchor_turn=None,
    anchor_tolerance=None,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda turns: turns)
    anchor_tolerance = (
        anchor_tolerance
        if anchor_tolerance is not None
        else (1 if mission == "snipe" else None)
    )
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if mission == "crash_exploit" and anchor_turn is not None and turns < anchor_turn:
            tested[send] = None
            return None
        raw_eval_turn = int(math.ceil(eval_turn_fn(turns)))
        if raw_eval_turn < turns:
            tested[send] = None
            return None
        eval_turn = raw_eval_turn
        need = world.min_ships_to_own_by(
            target.id,
            eval_turn,
            world.player,
            arrival_turn=turns,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        if mission in ("snipe", "crash_exploit"):
            desired = need
        elif mission == "rescue":
            desired = min(
                src_cap,
                max(
                    need,
                    need + DEFENSE_SEND_MARGIN_BASE + target.production * DEFENSE_SEND_MARGIN_PROD_WEIGHT,
                ),
            )
        else:
            desired = min(
                src_cap,
                max(need, preferred_send(target, need, turns, src_cap, world, modes, policy)),
            )

        result = (angle, turns, eval_turn, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(src.id, target.id, src_cap, hints=(seed_hint,)),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(result[1] - anchor_turn) > anchor_tolerance
        ):
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            if (
                anchor_turn is not None
                and anchor_tolerance is not None
                and abs(turns - anchor_turn) > anchor_tolerance
            ):
                return None
            if mission == "rescue" and turns > eval_turn:
                return None
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (
            0 if mission != "snipe" or anchor_turn is None else abs(tested[send][1] - anchor_turn),
            abs(send - seed_hint),
            tested[send][1],
            send,
        ),
    )

    seen = set()
    for send in candidate_sends:
        if send in seen:
            continue
        seen.add(send)
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(turns - anchor_turn) > anchor_tolerance
        ):
            continue
        if mission == "rescue" and turns > eval_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def settle_reinforce_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    hold_until,
    max_arrival_turn,
    max_iter=4,
):
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        cached = tested.get(send)
        if cached is not None or send in tested:
            return cached

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if turns > max_arrival_turn:
            tested[send] = None
            return None

        need = world.reinforcement_needed_to_hold_until(
            target.id,
            turns,
            hold_until,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        desired = min(src_cap, need + REINFORCE_SAFETY_MARGIN)
        result = (angle, turns, hold_until, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        world.probe_ship_candidates(src.id, target.id, src_cap, hints=(seed_hint,)),
        key=lambda send: (abs(send - seed_hint), send),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            return angle, turns, eval_turn, need, actual_send

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (abs(send - seed_hint), tested[send][1], send),
    )
    for send in candidate_sends:
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = result
        if actual_send < need or turns > max_arrival_turn:
            continue
        return angle, turns, eval_turn, need, actual_send

    return None


def build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy):
    if target.owner != -1:
        return None

    enemy_etas = sorted({
        int(math.ceil(eta))
        for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
        if owner not in (-1, world.player) and ships > 0
    })
    if not enemy_etas:
        return None

    best = None
    for enemy_eta in enemy_etas[:3]:
        seeded = world.best_probe_aim(
            src.id,
            target.id,
            src_available,
            hints=(int(target.ships) + 1, int(target.ships) + 8),
            anchor_turn=enemy_eta,
            max_anchor_diff=1,
        )
        if seeded is None:
            continue

        probe, rough = seeded
        sync_turn = max(rough[1], enemy_eta)
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue

        plan = settle_plan(
            src,
            target,
            src_available,
            probe,
            world,
            planned_commitments,
            modes,
            policy,
            mission="snipe",
            eval_turn_fn=lambda turns, enemy_eta=enemy_eta: max(turns, enemy_eta),
            anchor_turn=enemy_eta,
        )
        if plan is None:
            continue

        angle, turns, sync_turn, need, send_pref = plan
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue

        value = target_value(target, sync_turn, "snipe", world, modes, policy)
        if value <= 0:
            continue

        score = apply_score_modifiers(
            value / (send_pref + sync_turn * SNIPE_COST_TURN_WEIGHT + 1.0),
            target,
            "snipe",
            world,
        )
        option = ShotOption(
            score=score,
            src_id=src.id,
            target_id=target.id,
            angle=angle,
            turns=turns,
            needed=need,
            send_cap=send_pref,
            mission="snipe",
            anchor_turn=enemy_eta,
        )
        mission_obj = Mission(
            kind="snipe",
            score=score,
            target_id=target.id,
            turns=sync_turn,
            options=[option],
        )
        if best is None or mission_obj.score > best.score:
            best = mission_obj

    return best


def build_rescue_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                max_turn=fall_turn,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="rescue",
                eval_turn_fn=lambda _turns, fall_turn=fall_turn: fall_turn,
                anchor_turn=fall_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            saved_turns = max(1, world.remaining_steps - fall_turn)
            value = target.production * saved_turns + max(0, target.ships) * DEFENSE_SHIP_VALUE
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= DEFENSE_FRONTIER_SCORE_MULT
            score = value / (send_pref + turns * DEFENSE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="rescue",
                anchor_turn=fall_turn,
            )
            missions.append(Mission(
                kind="rescue",
                score=score,
                target_id=target.id,
                turns=fall_turn,
                options=[option],
            ))

    return missions


def build_recapture_missions(world, policy, planned_commitments, modes):
    missions = []

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue

        for src in world.my_planets:
            if src.id == target.id:
                continue

            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                min_turn=fall_turn + 1,
                max_turn=fall_turn + RECAPTURE_LOOKAHEAD_TURNS,
            )
            if seeded is None:
                continue
            probe, probe_aim = seeded
            probe_turns = probe_aim[1]

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
                eval_turn_fn=onevone_capture_eval_fn(target, world),
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if turns <= fall_turn or turns - fall_turn > RECAPTURE_LOOKAHEAD_TURNS:
                continue

            saved_turns = max(1, world.remaining_steps - turns)
            value = (
                RECAPTURE_PRODUCTION_WEIGHT * target.production * saved_turns
                + RECAPTURE_IMMEDIATE_WEIGHT * max(0, target.ships)
            )
            if world.enemy_planets and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22:
                value *= RECAPTURE_FRONTIER_MULT
            value *= RECAPTURE_VALUE_MULT
            score = value / (send_pref + turns * RECAPTURE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="recapture",
                anchor_turn=fall_turn,
            )
            missions.append(Mission(
                kind="recapture",
                score=score,
                target_id=target.id,
                turns=turns,
                options=[option],
            ))

    return missions


def build_fresh_frontier_recapture_missions(world, policy, planned_commitments, modes):
    """Exploit freshly captured weak enemy prod-2+ planets with hold-positive shots.

    This is intentionally narrower than generic capture scoring:
    - 4P only
    - target must have changed owner to an enemy within a short stateful lookback
    - prod-2+ and low garrison
    - short ETA from a nearby source
    - send must own through a small hold window, not just flip momentarily
    """
    missions = []
    if not FRESH_RECAPTURE_ENABLED or not world.is_four_player:
        return missions
    if world.step < FRESH_RECAPTURE_START_STEP:
        return missions

    candidates = []
    for target in world.enemy_planets:
        event = fresh_enemy_capture_event(target, world)
        if event is None:
            continue
        age = max(0, int(world.step) - int(event.get("step", world.step)))
        candidates.append((age, target))

    for age, target in sorted(
        candidates,
        key=lambda item: (item[0], -item[1].production, item[1].ships),
    ):
        for src in nearest_sources_to_target(target, world.my_planets, FRESH_RECAPTURE_SOURCE_TOP_K):
            src_available = policy["attack_budget"].get(src.id, 0)
            src_available = min(
                src_available,
                int(src.ships * FRESH_RECAPTURE_MAX_SEND_FRACTION),
            )
            if src_available < FRESH_RECAPTURE_MIN_SOURCE_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + target.production + 3, int(target.ships) + 6),
                max_turn=FRESH_RECAPTURE_MAX_ETA,
            )
            if seeded is None:
                continue
            probe, rough_aim = seeded
            rough_turns = rough_aim[1]
            if rough_turns > FRESH_RECAPTURE_MAX_ETA:
                continue

            hold_until = min(HORIZON, rough_turns + FRESH_RECAPTURE_HOLD_TURNS)
            needed = world.min_ships_to_own_by(
                target.id,
                hold_until,
                world.player,
                arrival_turn=rough_turns,
                planned_commitments=planned_commitments,
                upper_bound=src_available,
            )
            if needed <= 0 or needed > src_available:
                continue

            send_guess = min(
                src_available,
                max(needed, int(target.ships) + target.production + 4),
            )
            plan = settle_plan(
                src,
                target,
                src_available,
                max(probe, send_guess),
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
                eval_turn_fn=lambda turns: min(
                    HORIZON,
                    int(math.ceil(turns)) + FRESH_RECAPTURE_HOLD_TURNS,
                ),
                max_iter=5,
            )
            if plan is None:
                continue

            angle, turns, eval_turn, need, send_pref = plan
            if turns > FRESH_RECAPTURE_MAX_ETA:
                continue
            if send_pref < need or need > src_available:
                continue

            value = target_value(target, turns, "capture", world, modes, policy)
            if value <= 0:
                continue
            age_bonus = 1.0 + max(0, FRESH_RECAPTURE_LOOKBACK - age) * 0.025
            low_garrison_bonus = 1.0 + max(
                0,
                FRESH_RECAPTURE_MAX_GARRISON - int(target.ships),
            ) * 0.015
            value *= FRESH_RECAPTURE_VALUE_MULT
            score = apply_score_modifiers(
                value / (send_pref + turns * RECAPTURE_COST_TURN_WEIGHT + 1.0),
                target,
                "capture",
                world,
            )
            score *= FRESH_RECAPTURE_SCORE_MULT * age_bonus * low_garrison_bonus

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="recapture",
                anchor_turn=eval_turn,
            )
            missions.append(Mission(
                kind="recapture",
                score=score,
                target_id=target.id,
                turns=turns,
                options=[option],
            ))

    return missions


def build_reinforce_missions(world, policy, planned_commitments, modes, inventory_left_fn):
    if not REINFORCE_ENABLED:
        return []

    missions = []
    if world.remaining_steps < REINFORCE_MIN_FUTURE_TURNS:
        return missions

    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None:
            continue
        if target.production < REINFORCE_MIN_PRODUCTION:
            continue

        hold_until = min(HORIZON, fall_turn + REINFORCE_HOLD_LOOKAHEAD)
        max_arrival_turn = min(fall_turn, REINFORCE_MAX_TRAVEL_TURNS)

        for src in world.my_planets:
            if src.id == target.id:
                continue

            budget = inventory_left_fn(src.id)
            source_cap = min(budget, int(src.ships * REINFORCE_MAX_SOURCE_FRACTION))
            if source_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                source_cap,
                hints=(target.production + REINFORCE_SAFETY_MARGIN + 2,),
                max_turn=max_arrival_turn,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_reinforce_plan(
                src,
                target,
                source_cap,
                probe,
                world,
                planned_commitments,
                hold_until,
                max_arrival_turn,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            value = reinforce_value(target, hold_until, world, policy)
            score = value / (send_pref + turns * REINFORCE_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="reinforce",
                anchor_turn=hold_until,
            )
            missions.append(Mission(
                kind="reinforce",
                score=score,
                target_id=target.id,
                turns=fall_turn,
                options=[option],
            ))

    return missions


def build_crash_exploit_missions(world, policy, planned_commitments, modes):
    if not CRASH_EXPLOIT_ENABLED or not world.is_four_player:
        return []

    missions = []
    for crash in detect_enemy_crashes(world):
        target = world.planet_by_id[crash["target_id"]]
        if target.owner == world.player:
            continue
        desired_arrival = crash["crash_turn"] + CRASH_EXPLOIT_POST_CRASH_DELAY

        for src in world.my_planets:
            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(12, int(target.ships) + 1),
                anchor_turn=desired_arrival,
                max_anchor_diff=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="crash_exploit",
                eval_turn_fn=lambda turns, desired_arrival=desired_arrival: max(turns, desired_arrival),
                anchor_turn=desired_arrival,
                anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            value = target_value(target, turns, "crash_exploit", world, modes, policy)
            if value <= 0:
                continue

            score = apply_score_modifiers(
                value / (send_pref + turns * SNIPE_COST_TURN_WEIGHT + 1.0),
                target,
                "crash_exploit",
                world,
            )
            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="crash_exploit",
                anchor_turn=desired_arrival,
            )
            missions.append(Mission(
                kind="crash_exploit",
                score=score,
                target_id=target.id,
                turns=turns,
                options=[option],
            ))

    return missions


def build_gang_up_missions(world, policy, planned_commitments, modes):
    """
    Attack enemy planets that will be weakened by inter-enemy combat.
    Schedule our fleet to arrive shortly after the battle resolves.
    This is the key 4P tactic: use enemies fighting each other to our advantage.
    """
    missions = []

    for battle in detect_enemy_planet_battles(world):
        target = world.planet_by_id[battle["target_id"]]
        if target.owner == world.player:
            continue

        # Arrive shortly after the battle to clean up; 4P windows are shorter and more valuable.
        gang_delay = GANG_UP_POST_BATTLE_DELAY_4P if world.is_four_player else GANG_UP_POST_BATTLE_DELAY
        gang_window = GANG_UP_ETA_WINDOW_4P if world.is_four_player else GANG_UP_ETA_WINDOW
        gang_mult = GANG_UP_VALUE_MULT_4P if world.is_four_player else GANG_UP_VALUE_MULT
        desired_arrival = battle["battle_turn"] + gang_delay

        for src in world.my_planets:
            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            # Hint: need just slightly more than estimated post-battle ships
            post_hint = max(3, int(battle["post_ships"]) + 3)
            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(post_hint, int(target.ships) + 1),
                anchor_turn=desired_arrival,
                max_anchor_diff=gang_window,
            )
            if seeded is None:
                continue
            probe, _ = seeded

            plan = settle_plan(
                src,
                target,
                src_available,
                probe,
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
                eval_turn_fn=lambda turns, da=desired_arrival: max(turns, da),
                anchor_turn=desired_arrival,
                anchor_tolerance=gang_window,
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue

            value = target_value(target, turns, "capture", world, modes, policy)
            value *= gang_mult  # extra bonus for exploiting enemy fight
            if value <= 0:
                continue

            score = apply_score_modifiers(
                value / (send_pref + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                target,
                "capture",
                world,
            )

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send_pref,
                mission="capture",
                anchor_turn=desired_arrival,
            )
            missions.append(Mission(
                kind="single",
                score=score,
                target_id=target.id,
                turns=turns,
                options=[option],
            ))

    return missions


def build_elimination_missions(world, policy, planned_commitments, modes):
    """
    Dedicated missions to eliminate the weakest enemy entirely.
    High priority in 4P — removing a player is a massive advantage.
    Only activates when we have a clear strength advantage over the target.
    """
    if world._weakest_enemy is None:
        return []

    weakest = world._weakest_enemy
    weakest_total = world._weakest_enemy_strength

    # Only pursue if we're meaningfully stronger than the target
    if weakest_total > world.my_total * 0.9:
        return []

    # Must be clearly the weakest (not just slightly weaker than someone else)
    other_enemies = [
        s for owner, s in world.owner_strength.items()
        if owner not in (world.player, weakest)
    ]
    weakest_gap_limit = 1.3 if world.is_four_player else 0.95
    if other_enemies and weakest_total > min(other_enemies) * weakest_gap_limit:
        return []

    weakest_planets = [p for p in world.enemy_planets if p.owner == weakest]
    if not weakest_planets:
        return []

    missions = []
    elimination_bonus_mult = 1.5 if world.is_four_player else 1.25

    for target in weakest_planets:
        for src in world.my_planets:
            src_available = policy["attack_budget"].get(src.id, 0)
            if src_available < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + 1, int(target.ships) + 5),
            )
            if seeded is None:
                continue
            probe, rough_aim = seeded
            rough_turns = rough_aim[1]

            if not candidate_time_valid(target, rough_turns, world, LATE_CAPTURE_BUFFER):
                continue

            global_needed = world.min_ships_to_own_at(
                target.id, rough_turns, world.player,
                planned_commitments=planned_commitments,
            )
            if global_needed <= 0 or global_needed > src_available:
                continue

            send_guess = preferred_send(target, global_needed, rough_turns, src_available, world, modes, policy)
            plan = settle_plan(
                src, target, src_available, send_guess,
                world, planned_commitments, modes, policy,
                mission="capture",
            )
            if plan is None:
                continue

            angle, turns, _, need, send_pref = plan
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            if send_pref < need:
                continue

            value = target_value(target, turns, "capture", world, modes, policy)
            if value <= 0:
                continue

            score = apply_score_modifiers(
                value * elimination_bonus_mult / (send_pref + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                target, "capture", world,
            )

            option = ShotOption(
                score=score, src_id=src.id, target_id=target.id,
                angle=angle, turns=turns, needed=need, send_cap=send_pref,
                mission="capture",
            )
            missions.append(Mission(
                kind="single", score=score, target_id=target.id,
                turns=turns, options=[option],
            ))

    return missions


def build_recovery_pressure_missions(world, policy, planned_commitments, modes):
    """
    When a single opponent pulls ahead on production/planets, add high-priority
    short capture missions.  This specifically prevents the seed-102 pattern
    where the bot stopped launching after turn 100 while old_base_v2 kept
    taking neutrals and border planets.
    """
    if not recovery_pressure_active(world):
        return []

    missions = []
    targets = [
        target
        for target in world.planets
        if target.owner != world.player
        and target.production >= RECOVERY_MIN_TARGET_PROD
        and target.id not in world.comet_ids
    ]
    if not targets:
        return missions

    for src in world.my_planets:
        src_available = policy["attack_budget"].get(src.id, 0)
        if src_available < RECOVERY_SOURCE_MIN_SHIPS:
            continue

        best_for_source = []
        for target in targets:
            if target.id == src.id:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + 1, int(target.ships) + 4),
                max_turn=RECOVERY_MAX_ETA,
            )
            if seeded is None:
                continue
            probe, rough_aim = seeded
            rough_turns = rough_aim[1]
            if bad_early_trickle_target(target, rough_turns, src_available, world):
                continue
            if opening_filter(target, rough_turns, probe, src_available, world, policy):
                continue

            global_needed = world.min_ships_to_own_at(
                target.id,
                rough_turns,
                world.player,
                planned_commitments=planned_commitments,
                upper_bound=src_available,
            )
            if global_needed <= 0 or global_needed > src_available:
                continue

            send_guess = preferred_send(
                target, global_needed, rough_turns, src_available, world, modes, policy,
            )
            plan = settle_plan(
                src,
                target,
                src_available,
                send_guess,
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
                eval_turn_fn=onevone_capture_eval_fn(target, world),
            )
            if plan is None:
                continue

            angle, turns, _, need, send_cap = plan
            if turns > RECOVERY_MAX_ETA or send_cap < need:
                continue
            value = target_value(target, turns, "capture", world, modes, policy)
            if value <= 0:
                continue
            if target.owner == -1:
                value *= 1.08
            if target.production >= CONTESTED_HIGH_PROD_MIN:
                value *= 1.08
            score = apply_score_modifiers(
                value * RECOVERY_SCORE_MULT / (send_cap + turns * (ATTACK_COST_TURN_WEIGHT * 0.8) + 1.0),
                target,
                "capture",
                world,
            )
            best_for_source.append((
                score,
                Mission(
                    kind="single",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[
                        ShotOption(
                            score=score,
                            src_id=src.id,
                            target_id=target.id,
                            angle=angle,
                            turns=turns,
                            needed=need,
                            send_cap=send_cap,
                            mission="capture",
                        )
                    ],
                ),
            ))

        best_for_source.sort(key=lambda item: -item[0])
        missions.extend(mission for _, mission in best_for_source[:2])

    return missions


def plan_moves(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    def time_left():
        if deadline is None:
            return 10**9
        return deadline - time.perf_counter()

    def allow_heavy_phase():
        return time_left() > HEAVY_PHASE_MIN_TIME and len(world.planets) <= HEAVY_ROUTE_PLANET_LIMIT

    def allow_optional_phase():
        return time_left() > OPTIONAL_PHASE_MIN_TIME

    modes = build_modes(world)
    policy = build_policy_state(world, deadline=deadline)
    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions = []
    moves = []
    spent_total = defaultdict(int)

    def source_inventory_left(source_id):
        return world.source_inventory_left(source_id, spent_total)

    def source_attack_left(source_id):
        budget = policy["attack_budget"].get(source_id, 0)
        return max(0, budget - spent_total[source_id])

    def append_move(src_id, angle, ships):
        send = min(int(ships), source_inventory_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    def finalize_moves():
        final_moves = []
        used_final = defaultdict(int)
        for src_id, angle, ships in moves:
            source = world.planet_by_id[src_id]
            max_allowed = int(source.ships) - used_final[src_id]
            send = min(int(ships), max_allowed)
            if send >= 1:
                final_moves.append([src_id, float(angle), int(send)])
                used_final[src_id] += send
        return final_moves

    def compute_live_doomed():
        doomed = set()
        for planet in world.my_planets:
            status = world.hold_status(
                planet.id,
                planned_commitments=planned_commitments,
                horizon=DOOMED_EVAC_TURN_LIMIT,
            )
            if (
                not status["holds_full"]
                and status["fall_turn"] is not None
                and status["fall_turn"] <= DOOMED_EVAC_TURN_LIMIT
                and source_inventory_left(planet.id) >= DOOMED_MIN_SHIPS
            ):
                doomed.add(planet.id)
        return doomed

    def time_filters_pass(target, turns, needed, src_cap):
        if not candidate_time_valid(target, turns, world, VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER):
            return False
        if opening_filter(target, turns, needed, src_cap, world, policy):
            return False
        return True

    # === MISSION BUILDING PHASE ===

    if allow_heavy_phase():
        missions.extend(
            build_reinforce_missions(world, policy, planned_commitments, modes, source_inventory_left)
        )

    missions.extend(build_rescue_missions(world, policy, planned_commitments, modes))
    missions.extend(build_recapture_missions(world, policy, planned_commitments, modes))
    missions.extend(build_fresh_frontier_recapture_missions(world, policy, planned_commitments, modes))
    missions.extend(build_recovery_pressure_missions(world, policy, planned_commitments, modes))

    # Elimination missions — high priority in 4P
    if allow_heavy_phase():
        missions.extend(build_elimination_missions(world, policy, planned_commitments, modes))

    # Gang-up missions — exploit inter-enemy fights (very useful in 4P)
    if allow_heavy_phase():
        missions.extend(build_gang_up_missions(world, policy, planned_commitments, modes))

    for src in world.my_planets:
        if expired():
            return finalize_moves()
        src_available = source_attack_left(src.id)
        if src_available <= 0:
            continue

        for target in world.planets:
            if expired():
                return finalize_moves()
            if target.id == src.id or target.owner == world.player:
                continue

            seeded = world.best_probe_aim(
                src.id,
                target.id,
                src_available,
                hints=(int(target.ships) + 1,),
            )
            if seeded is None:
                continue
            _, rough_aim = seeded

            rough_turns = rough_aim[1]
            if not candidate_time_valid(
                target,
                rough_turns,
                world,
                VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER,
            ):
                continue

            global_needed = onevone_ships_to_own_capture(
                target,
                rough_turns,
                world,
                planned_commitments,
            )
            if global_needed <= 0:
                continue
            if opening_filter(target, rough_turns, global_needed, src_available, world, policy):
                continue

            partial_send_cap = min(
                src_available,
                preferred_send(target, global_needed, rough_turns, src_available, world, modes, policy),
            )
            if partial_send_cap >= PARTIAL_SOURCE_MIN_SHIPS:
                partial_seed = world.best_probe_aim(
                    src.id,
                    target.id,
                    partial_send_cap,
                    hints=(partial_send_cap, global_needed, int(target.ships) + 1),
                )
                if partial_seed is not None:
                    _, partial_aim = partial_seed
                    p_angle, p_turns, _, _ = partial_aim
                    if time_filters_pass(target, p_turns, global_needed, src_available):
                        partial_value = target_value(target, p_turns, "swarm", world, modes, policy)
                        if partial_value > 0:
                            partial_score = apply_score_modifiers(
                                partial_value / (partial_send_cap + p_turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                target,
                                "swarm",
                                world,
                            )
                            source_options_by_target[target.id].append(
                                ShotOption(
                                    score=partial_score,
                                    src_id=src.id,
                                    target_id=target.id,
                                    angle=p_angle,
                                    turns=p_turns,
                                    needed=global_needed,
                                    send_cap=partial_send_cap,
                                    mission="swarm",
                                )
                            )

            if global_needed <= src_available:
                send_guess = preferred_send(
                    target, global_needed, rough_turns, src_available, world, modes, policy,
                )
                plan = settle_plan(
                    src,
                    target,
                    src_available,
                    send_guess,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                    eval_turn_fn=onevone_capture_eval_fn(target, world),
                )
                if plan is None:
                    continue

                angle, turns, _, needed, send_cap = plan
                if not time_filters_pass(target, turns, needed, src_available):
                    continue
                if send_cap < 1:
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (send_cap + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )

                option = ShotOption(
                    score=score,
                    src_id=src.id,
                    target_id=target.id,
                    angle=angle,
                    turns=turns,
                    needed=needed,
                    send_cap=send_cap,
                    mission="capture",
                )

                if send_cap >= needed:
                    missions.append(Mission(
                        kind="single",
                        score=score,
                        target_id=target.id,
                        turns=turns,
                        options=[option],
                    ))

            snipe = build_snipe_mission(src, target, src_available, world, planned_commitments, modes, policy)
            if snipe is not None:
                missions.append(snipe)

    for target_id, options in source_options_by_target.items():
        if expired():
            return finalize_moves()
        if len(options) < 2:
            continue

        target = world.planet_by_id[target_id]
        top_options = sorted(options, key=lambda item: -item.score)[:MULTI_SOURCE_TOP_K]
        for i in range(len(top_options)):
            for j in range(i + 1, len(top_options)):
                first = top_options[i]
                second = top_options[j]
                if first.src_id == second.src_id:
                    continue
                pair_tol = swarm_eta_tolerance((first, second), target, world)
                if abs(first.turns - second.turns) > pair_tol:
                    continue

                joint_turn = max(first.turns, second.turns)
                total_cap = first.send_cap + second.send_cap
                eval_turn = onevone_capture_eval_turn(target, joint_turn, world)
                need = world.min_ships_to_own_by(
                    target_id,
                    eval_turn,
                    world.player,
                    arrival_turn=joint_turn,
                    planned_commitments=planned_commitments,
                    upper_bound=total_cap,
                )
                if need <= 0:
                    continue
                if first.send_cap >= need or second.send_cap >= need:
                    continue
                if total_cap < need:
                    continue

                value = target_value(target, joint_turn, "swarm", world, modes, policy)
                if value <= 0:
                    continue

                pair_score = apply_score_modifiers(
                    value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "swarm",
                    world,
                )
                pair_score *= MULTI_SOURCE_PLAN_PENALTY
                missions.append(Mission(
                    kind="swarm",
                    score=pair_score,
                    target_id=target_id,
                    turns=joint_turn,
                    options=[first, second],
                ))

        if (
            THREE_SOURCE_SWARM_ENABLED
            and allow_heavy_phase()
            and target.owner not in (-1, world.player)
            and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS
            and len(top_options) >= 3
        ):
            for i in range(len(top_options)):
                for j in range(i + 1, len(top_options)):
                    for k in range(j + 1, len(top_options)):
                        if expired():
                            return finalize_moves()
                        trio = [top_options[i], top_options[j], top_options[k]]
                        if len({option.src_id for option in trio}) < 3:
                            continue
                        trio_tol = swarm_eta_tolerance(tuple(trio), target, world)
                        turns = [option.turns for option in trio]
                        if max(turns) - min(turns) > trio_tol:
                            continue

                        joint_turn = max(turns)
                        total_cap = sum(option.send_cap for option in trio)
                        eval_turn = onevone_capture_eval_turn(target, joint_turn, world)
                        need = world.min_ships_to_own_by(
                            target_id,
                            eval_turn,
                            world.player,
                            arrival_turn=joint_turn,
                            planned_commitments=planned_commitments,
                            upper_bound=total_cap,
                        )
                        if need <= 0 or total_cap < need:
                            continue
                        if any(
                            trio[a].send_cap + trio[b].send_cap >= need
                            for a in range(3)
                            for b in range(a + 1, 3)
                        ):
                            continue

                        value = target_value(target, joint_turn, "swarm", world, modes, policy)
                        if value <= 0:
                            continue

                        trio_score = apply_score_modifiers(
                            value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                            target,
                            "swarm",
                            world,
                        )
                        trio_score *= THREE_SOURCE_PLAN_PENALTY
                        missions.append(Mission(
                            kind="swarm",
                            score=trio_score,
                            target_id=target_id,
                            turns=joint_turn,
                            options=trio,
                        ))

    if allow_heavy_phase():
        missions.extend(build_crash_exploit_missions(world, policy, planned_commitments, modes))

    missions.sort(key=lambda item: -item.score)

    # === MISSION EXECUTION PHASE ===

    for mission in missions:
        if expired():
            return finalize_moves()
        target = world.planet_by_id[mission.target_id]

        if mission.kind in ("single", "snipe", "rescue", "recapture", "reinforce", "crash_exploit"):
            option = mission.options[0]
            src = world.planet_by_id[option.src_id]
            if mission.kind == "reinforce":
                left = min(
                    source_inventory_left(option.src_id),
                    int(src.ships * REINFORCE_MAX_SOURCE_FRACTION),
                )
            else:
                left = source_attack_left(option.src_id)
            if left <= 0:
                continue

            if mission.kind == "reinforce":
                plan = settle_reinforce_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    option.anchor_turn,
                    mission.turns,
                )
            elif mission.kind == "rescue":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="rescue",
                    eval_turn_fn=lambda _turns, hold_turn=mission.turns: hold_turn,
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "snipe":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="snipe",
                    eval_turn_fn=lambda turns, enemy_eta=option.anchor_turn: max(turns, enemy_eta),
                    anchor_turn=option.anchor_turn,
                )
            elif mission.kind == "crash_exploit":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="crash_exploit",
                    eval_turn_fn=lambda turns, desired_arrival=option.anchor_turn: max(turns, desired_arrival),
                    anchor_turn=option.anchor_turn,
                    anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
                )
            else:
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, option.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                    eval_turn_fn=onevone_capture_eval_fn(target, world),
                )
            if plan is None:
                continue

            angle, turns, _, need, send = plan
            if send < need or need > left:
                continue

            sent = append_move(option.src_id, angle, send)
            if sent < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(sent)))
            continue

        limits = []
        for option in mission.options:
            left = source_attack_left(option.src_id)
            limits.append(min(left, option.send_cap))
        if min(limits) <= 0:
            continue

        mission_eval_turn = onevone_capture_eval_turn(target, mission.turns, world)
        missing = world.min_ships_to_own_by(
            target.id,
            mission_eval_turn,
            world.player,
            arrival_turn=mission.turns,
            planned_commitments=planned_commitments,
            upper_bound=sum(limits),
        )
        if missing <= 0 or sum(limits) < missing:
            continue

        ordered = sorted(
            zip(mission.options, limits),
            key=lambda item: (item[0].turns, -item[1], item[0].src_id),
        )
        remaining = missing
        sends = {}
        for idx, (option, limit) in enumerate(ordered):
            remaining_other = sum(other_limit for _, other_limit in ordered[idx + 1:])
            send = min(limit, max(0, remaining - remaining_other))
            sends[option.src_id] = send
            remaining -= send
        if remaining > 0:
            continue

        reaimed = []
        for option, _ in ordered:
            send = sends.get(option.src_id, 0)
            if send <= 0:
                continue
            src = world.planet_by_id[option.src_id]
            fixed_aim = world.plan_shot(src.id, target.id, send)
            if fixed_aim is None:
                reaimed = []
                break
            angle, turns, _, _ = fixed_aim
            reaimed.append((option.src_id, angle, turns, send))
        if not reaimed:
            continue

        turns_only = [item[2] for item in reaimed]
        eta_tol = swarm_eta_tolerance(mission.options, target, world)
        if max(turns_only) - min(turns_only) > eta_tol:
            continue

        actual_joint_turn = max(turns_only)
        actual_eval_turn = onevone_capture_eval_turn(target, actual_joint_turn, world)
        owner_after, _ = world.projected_state(
            target.id,
            actual_eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=[(turns, world.player, send) for _, _, turns, send in reaimed],
        )
        if owner_after != world.player:
            continue

        committed = []
        for src_id, angle, turns, send in reaimed:
            actual = append_move(src_id, angle, send)
            if actual <= 0:
                continue
            committed.append((turns, world.player, int(actual)))
        if sum(item[2] for item in committed) < missing:
            continue
        planned_commitments[target.id].extend(committed)

    # === FOLLOWUP PHASE ===

    if not world.is_very_late and allow_optional_phase():
        for src in world.my_planets:
            if expired():
                return finalize_moves()
            src_left = source_attack_left(src.id)
            if src_left < FOLLOWUP_MIN_SHIPS:
                continue

            best = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == src.id or target.owner == world.player:
                    continue
                if target.id in world.comet_ids and target.production <= LOW_VALUE_COMET_PRODUCTION:
                    continue

                seeded = world.best_probe_aim(
                    src.id,
                    target.id,
                    src_left,
                    hints=(int(target.ships) + 1,),
                )
                if seeded is None:
                    continue
                rough_ships, rough_aim = seeded

                est_turns = rough_aim[1]
                if world.is_late and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue

                rough_needed = onevone_ships_to_own_capture(
                    target,
                    est_turns,
                    world,
                    planned_commitments,
                    upper_bound=src_left,
                )
                if rough_needed <= 0 or rough_needed > src_left:
                    continue
                if opening_filter(target, est_turns, rough_needed, src_left, world, policy):
                    continue

                send = preferred_send(target, rough_needed, est_turns, src_left, world, modes, policy)
                if send < rough_needed:
                    continue

                plan = settle_plan(
                    src,
                    target,
                    src_left,
                    send,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                    eval_turn_fn=onevone_capture_eval_fn(target, world),
                )
                if plan is None:
                    continue

                _, turns, _, need, final_send = plan
                if world.is_late and turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue
                if final_send < need:
                    continue

                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (final_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                if best is None or score > best[0]:
                    best = (score, target, plan)

            if best is None:
                continue

            _, target, plan = best
            angle, turns, _, need, send = plan
            src_left = source_attack_left(src.id)
            if need > src_left:
                continue

            plan = settle_plan(
                src,
                target,
                src_left,
                min(src_left, send),
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
                eval_turn_fn=onevone_capture_eval_fn(target, world),
            )
            if plan is None:
                continue

            angle, turns, _, need, send = plan
            if send < need:
                continue

            actual = append_move(src.id, angle, send)
            if actual < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(actual)))

    # === DOOMED PLANET EVACUATION ===

    if expired():
        return finalize_moves()
    live_doomed = compute_live_doomed()
    if live_doomed:
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        if frontier_targets:
            frontier_distance = {
                planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
                for planet in world.my_planets
            }
        else:
            frontier_distance = {planet.id: 10**9 for planet in world.my_planets}

        for planet in world.my_planets:
            if expired():
                return finalize_moves()
            if planet.id not in live_doomed:
                continue

            available_now = source_inventory_left(planet.id)
            if available_now < policy["reserve"].get(planet.id, 0):
                continue

            best_capture = None
            for target in world.planets:
                if expired():
                    return finalize_moves()
                if target.id == planet.id or target.owner == world.player:
                    continue
                seeded = world.best_probe_aim(
                    planet.id,
                    target.id,
                    available_now,
                    hints=(available_now, int(target.ships) + 1),
                )
                if seeded is None:
                    continue
                _, probe_aim = seeded
                probe_turns = probe_aim[1]
                if probe_turns > world.remaining_steps - 2:
                    continue

                need = world.min_ships_to_own_at(
                    target.id,
                    probe_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=available_now,
                )
                if need <= 0 or need > available_now:
                    continue

                plan = settle_plan(
                    planet,
                    target,
                    available_now,
                    min(available_now, max(need, int(target.ships) + 1)),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )
                if plan is None:
                    continue
                angle, turns, _, plan_need, send = plan
                if send < plan_need:
                    continue
                score = target_value(target, turns, "capture", world, modes, policy) / (send + turns + 1.0)
                if target.owner not in (-1, world.player):
                    score *= 1.05
                if best_capture is None or score > best_capture[0]:
                    best_capture = (score, target.id, angle, turns, send)

            if best_capture is not None:
                _, target_id, angle, turns, need = best_capture
                actual = append_move(planet.id, angle, need)
                if actual >= 1:
                    planned_commitments[target_id].append((turns, world.player, int(actual)))
                continue

            safe_allies = [
                ally
                for ally in world.my_planets
                if ally.id != planet.id and ally.id not in live_doomed
            ]
            if not safe_allies:
                continue

            retreat_target = min(
                safe_allies,
                key=lambda ally: (
                    frontier_distance.get(ally.id, 10**9),
                    planet_distance(planet, ally),
                ),
            )
            aim = world.plan_shot(planet.id, retreat_target.id, available_now)
            if aim is None:
                continue
            angle, _, _, _ = aim
            append_move(planet.id, angle, available_now)

    # === REAR FORWARDING ===

    if (
        (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
        and allow_optional_phase()
    ):
        live_doomed = compute_live_doomed()
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        frontier_distance = {
            planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
            for planet in world.my_planets
        }
        safe_fronts = [planet for planet in world.my_planets if planet.id not in live_doomed]
        if safe_fronts:
            front_anchor = min(safe_fronts, key=lambda planet: frontier_distance[planet.id])
            send_ratio = REAR_SEND_RATIO_FOUR_PLAYER if world.is_four_player else REAR_SEND_RATIO_TWO_PLAYER
            if modes["is_finishing"]:
                send_ratio = max(send_ratio, REAR_SEND_RATIO_FOUR_PLAYER)

            for rear in sorted(world.my_planets, key=lambda planet: -frontier_distance[planet.id]):
                if expired():
                    return finalize_moves()
                if rear.id == front_anchor.id or rear.id in live_doomed:
                    continue
                rear_min_ships = REAR_SOURCE_MIN_SHIPS_4P if world.is_four_player else REAR_SOURCE_MIN_SHIPS
                if source_attack_left(rear.id) < rear_min_ships:
                    continue
                if frontier_distance[rear.id] < frontier_distance[front_anchor.id] * REAR_DISTANCE_RATIO:
                    continue

                stage_candidates = [
                    planet
                    for planet in safe_fronts
                    if planet.id != rear.id
                    and frontier_distance[planet.id] < frontier_distance[rear.id] * REAR_STAGE_PROGRESS
                ]
                if stage_candidates:
                    front = min(stage_candidates, key=lambda planet: planet_distance(rear, planet))
                else:
                    objective = min(frontier_targets, key=lambda target: planet_distance(rear, target))
                    remaining_fronts = [planet for planet in safe_fronts if planet.id != rear.id]
                    if not remaining_fronts:
                        continue
                    front = min(remaining_fronts, key=lambda planet: planet_distance(planet, objective))

                if front.id == rear.id:
                    continue

                send = int(source_attack_left(rear.id) * send_ratio)
                if send < REAR_SEND_MIN_SHIPS:
                    continue

                aim = world.plan_shot(rear.id, front.id, send)
                if aim is None:
                    continue

                angle, turns, _, _ = aim
                if turns > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(rear.id, angle, send)

    # === TOTAL WAR ENDGAME ===

    if world.is_total_war and world.enemy_planets and allow_optional_phase():
        # Prioritize weakest enemy in total war
        if world._weakest_enemy is not None:
            weakest_planets = [p for p in world.enemy_planets if p.owner == world._weakest_enemy]
            if weakest_planets:
                primary_targets = weakest_planets
            else:
                primary_targets = world.enemy_planets
        else:
            primary_targets = world.enemy_planets

        # Attack weakest enemy first, then others
        for target_set in [primary_targets, world.enemy_planets]:
            if not target_set:
                continue
            for src in world.my_planets:
                if expired():
                    return finalize_moves()
                atk_left = source_attack_left(src.id)
                if atk_left < 5:
                    continue

                # Find best target from current set
                best_target = None
                best_dist = float('inf')
                for ep in target_set:
                    d = planet_distance(src, ep)
                    if d < best_dist:
                        aim_test = world.plan_shot(src.id, ep.id, atk_left)
                        if aim_test is not None:
                            best_dist = d
                            best_target = ep

                if best_target is None:
                    continue
                aim = world.plan_shot(src.id, best_target.id, atk_left)
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                if turns >= world.remaining_steps:
                    continue
                append_move(src.id, angle, atk_left)
            break  # Only run primary targets set; secondary is backup

    return finalize_moves()

# ============================================================
# Agent Entry Point
# ============================================================

STATE = {"games": {}}


def _new_counter_tracker():
    return {
        "seen_enemy_fleet_ids": set(),
        "enemy_launch_history": [],
        "enemy_last_action_step": None,
        "enemy_observed_action": False,
        "planet_owner_map": {},
        "recent_enemy_captures": {},
    }


def _update_recent_enemy_captures(tracker, world):
    """Track fresh owner changes into enemy hands for W1-B recapture logic."""
    previous = tracker.setdefault("planet_owner_map", {})
    recent = tracker.setdefault("recent_enemy_captures", {})
    live_ids = set()

    for planet in world.planets:
        pid = int(planet.id)
        live_ids.add(pid)
        owner = int(planet.owner)
        prev_owner = previous.get(pid)
        if prev_owner is not None and int(prev_owner) != owner:
            if (
                owner not in (-1, world.player)
                and int(planet.production) >= FRESH_RECAPTURE_MIN_PRODUCTION
            ):
                recent[pid] = {
                    "step": int(world.step),
                    "owner": owner,
                    "prev_owner": int(prev_owner),
                    "ships": int(planet.ships),
                    "production": int(planet.production),
                }
            else:
                recent.pop(pid, None)
        previous[pid] = owner

    for pid in list(previous):
        if pid not in live_ids:
            previous.pop(pid, None)
            recent.pop(pid, None)

    for pid, event in list(recent.items()):
        planet = world.planet_by_id.get(pid)
        if planet is None:
            recent.pop(pid, None)
            continue
        if int(planet.owner) != int(event.get("owner", -99)):
            recent.pop(pid, None)
            continue
        if int(world.step) - int(event.get("step", world.step)) > FRESH_RECAPTURE_LOOKBACK:
            recent.pop(pid, None)
            continue
        if int(planet.production) < FRESH_RECAPTURE_MIN_PRODUCTION:
            recent.pop(pid, None)
            continue

    tracker["planet_owner_map"] = previous
    tracker["recent_enemy_captures"] = recent
    return dict(recent)


def _near_counter_comet_window(step):
    return any(abs(step - window) <= COUNTER_COMET_WINDOW_RADIUS for window in COUNTER_COMET_WINDOWS)


def _rolling_enemy_launch(history, step, window):
    return sum(ships for hist_step, ships, _ in history if step - hist_step <= window)


def _update_counter_profile(tracker, world):
    """Classify opponent behavior for lightweight counter-planning.

    The replay-derived signatures are deliberately coarse. The goal is to nudge
    an already-strong planner when the opponent is passive, comet-baiting, or
    preparing a final swarm -- not to make brittle opponent-specific scripts.
    """
    seen_ids = tracker.setdefault("seen_enemy_fleet_ids", set())
    launched_ships = 0
    launched_actions = 0
    current_enemy_fleet_ships = 0

    for fleet in world.fleets:
        if fleet.owner in (-1, world.player):
            continue
        current_enemy_fleet_ships += int(fleet.ships)
        fleet_id = int(fleet.id)
        if fleet_id not in seen_ids:
            seen_ids.add(fleet_id)
            launched_ships += int(fleet.ships)
            launched_actions += 1

    history = tracker.setdefault("enemy_launch_history", [])
    if launched_ships > 0:
        history.append((int(world.step), launched_ships, launched_actions))
        tracker["enemy_last_action_step"] = int(world.step)
        tracker["enemy_observed_action"] = True
    tracker["enemy_launch_history"] = [
        item for item in history if int(world.step) - item[0] <= 90
    ]

    last_action = tracker.get("enemy_last_action_step")
    quiet_turns = (
        int(world.step) - int(last_action)
        if last_action is not None
        else int(world.step)
    )
    recent_14 = _rolling_enemy_launch(tracker["enemy_launch_history"], int(world.step), 14)
    recent_late = _rolling_enemy_launch(
        tracker["enemy_launch_history"],
        int(world.step),
        COUNTER_LATE_SWARM_WINDOW,
    )

    rotator_pressure = 0
    for target in world.planets:
        if target.owner == world.player or target.production < 2 or world.is_static(target.id):
            continue
        for eta, owner, ships in world.arrivals_by_planet.get(target.id, []):
            if owner in (-1, world.player) or eta > 25:
                continue
            rotator_pressure += int(ships)

    late_swarm_danger = (
        world.step >= COUNTER_LATE_SWARM_START
        and world.enemy_total >= 800
    )
    late_swarmer = late_swarm_danger and (
        recent_late >= COUNTER_LATE_SWARM_SHIPS
        or current_enemy_fleet_ships >= COUNTER_LATE_SWARM_SHIPS
    )
    passive_mirror = (
        tracker.get("enemy_observed_action")
        and world.step >= COUNTER_QUIET_AFTER_STEP
        and quiet_turns >= COUNTER_QUIET_TURNS
        and world.enemy_total >= 300
        and not late_swarmer
    )
    comet_bait_chaser = (
        _near_counter_comet_window(int(world.step))
        and recent_14 >= COUNTER_LATE_SWARM_SHIPS
    )

    return {
        "passive_mirror": passive_mirror,
        "late_swarm_danger": late_swarm_danger,
        "late_swarmer": late_swarmer,
        "comet_bait_chaser": comet_bait_chaser,
        "rotator_racer": rotator_pressure >= 40,
        "quiet_turns": quiet_turns,
        "recent_enemy_launch_14": recent_14,
        "recent_enemy_launch_late": recent_late,
        "current_enemy_fleet_ships": current_enemy_fleet_ships,
        "rotator_pressure": rotator_pressure,
    }


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def build_world(obs, inferred_step=None):
    player = _read(obs, "player", 0)
    obs_step = _read(obs, "step", 0) or 0
    step = max(obs_step, inferred_step or 0)
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]
    initial_planets = [Planet(*planet) for planet in raw_init]
    initial_by_id = {planet.id: planet for planet in initial_planets}

    return WorldModel(
        player=player,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
    )


def _obs_signature(obs):
    raw = _read(obs, "initial_planets", None) or _read(obs, "planets", []) or []
    sig = []
    for planet in raw:
        try:
            sig.append((int(planet[0]), round(float(planet[2]), 2), round(float(planet[3]), 2), round(float(planet[4]), 2)))
        except (TypeError, ValueError, IndexError):
            sig.append((getattr(planet, "id", -1), round(float(getattr(planet, "x", 0.0)), 2), round(float(getattr(planet, "y", 0.0)), 2), round(float(getattr(planet, "radius", 0.0)), 2)))
    return tuple(sig)


def agent(obs, config=None):
    start_time = time.perf_counter()
    player = int(_read(obs, "player", 0) or 0)
    signature = _obs_signature(obs)
    game_state = STATE.setdefault("games", {}).setdefault(player, {"signature": None, "turn": 0})
    if game_state.get("signature") != signature:
        game_state["signature"] = signature
        game_state["turn"] = 0
        game_state["counter_tracker"] = _new_counter_tracker()

    observed_step = _read(obs, "step", None)
    inferred_step = int(observed_step) if observed_step is not None else int(game_state.get("turn", 0))
    game_state["turn"] = inferred_step + 1

    world = build_world(obs, inferred_step=inferred_step)
    if not world.my_planets:
        return []
    tracker = game_state.setdefault("counter_tracker", _new_counter_tracker())
    world.recent_enemy_captures = _update_recent_enemy_captures(tracker, world)
    world.counter_profile = _update_counter_profile(tracker, world)
    game_state["last_counter_profile"] = world.counter_profile
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = min(SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
    deadline = start_time + soft_budget
    return plan_moves(world, deadline=deadline)


__all__ = ["agent", "build_world"]
