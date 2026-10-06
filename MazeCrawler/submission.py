"""Maze Crawler Dijkstra planner baseline."""

from heapq import heappop, heappush


FACTORY = 0
SCOUT = 1
WORKER = 2

NORTH = 1
EAST = 2
SOUTH = 4
WEST = 8

OFFSETS = {
    "NORTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
    "SOUTH": (0, -1),
}
WALL_BITS = {
    "NORTH": NORTH,
    "EAST": EAST,
    "SOUTH": SOUTH,
    "WEST": WEST,
}

MEMORY = {}


def agent(obs, config):
    width = config.width
    height = getattr(config, "height", 20)
    player = getattr(obs, "player", 0)
    actions = {}
    step_no = getattr(obs, "step", None)

    my = {
        uid: data
        for uid, data in obs.robots.items()
        if len(data) > 4 and data[4] == player
    }
    occupied = {(data[1], data[2]) for data in my.values()}
    reserved = set()
    factory_data = next((data for data in my.values() if data[0] == FACTORY), None)

    memory = MEMORY.setdefault(
        player,
        {
            "last_step": -1,
            "positions": {},
            "actions": {},
            "idle_counts": {},
            "progress_waits": {},
            "escape": {},
            "max_rows": {},
            "avoid_until": {},
        },
    )
    if step_no is not None and step_no < memory.get("last_step", -1):
        memory = {
            "last_step": -1,
            "positions": {},
            "actions": {},
            "idle_counts": {},
            "progress_waits": {},
            "escape": {},
            "max_rows": {},
            "avoid_until": {},
        }
        MEMORY[player] = memory
    memory.setdefault("idle_counts", {})
    memory.setdefault("progress_waits", {})
    memory.setdefault("escape", {})
    memory.setdefault("max_rows", {})
    memory.setdefault("avoid_until", {})
    positions_memory = memory["positions"]
    now_step = step_no if step_no is not None else memory.get("last_step", -1) + 1

    south = getattr(obs, "southBound", None)
    if south is None:
        rows = [data[2] for data in my.values() if len(data) > 2]
        south = max(0, min(rows) - 2) if rows else 0
    north = getattr(obs, "northBound", south + height - 1)
    scroll_start_interval = max(1.0, float(getattr(config, "scrollStartInterval", 4)))
    scroll_end_interval = max(1.0, float(getattr(config, "scrollEndInterval", 1)))
    scroll_ramp_steps = max(1.0, float(getattr(config, "scrollRampSteps", 400)))

    crystals = {}
    for key, value in getattr(obs, "crystals", {}).items():
        if "," in key:
            crystals[tuple(map(int, key.split(",")))] = value

    def wall_at(col, row):
        walls = getattr(obs, "walls", [])
        idx = (row - south) * width + col
        if 0 <= col < width and 0 <= idx < len(walls) and walls[idx] != -1:
            return walls[idx]
        return 0

    def in_bounds(col, row):
        return 0 <= col < width and south <= row <= north

    def can_move_basic(col, row, direction):
        dc, dr = OFFSETS[direction]
        ncol, nrow = col + dc, row + dr
        return (
            in_bounds(ncol, nrow)
            and not (wall_at(col, row) & WALL_BITS[direction])
        )

    def can_move(col, row, direction):
        if not can_move_basic(col, row, direction):
            return False
        dc, dr = OFFSETS[direction]
        target = (col + dc, row + dr)
        return target not in occupied and target not in reserved

    def adjacent_direction(from_col, from_row, to_col, to_row):
        for direction, (dc, dr) in OFFSETS.items():
            if (from_col + dc, from_row + dr) == (to_col, to_row):
                return direction
        return None

    def has_exit(col, row):
        return any(can_move_basic(col, row, direction) for direction in OFFSETS)

    def jump_landing_stats(col, row, max_depth=7):
        start = (col, row)
        queue = [(col, row, 0)]
        seen = {start}
        best_row = row
        found_crystal = crystals.get(start, 0) > 0
        forward_exits = sum(1 for direction in ("NORTH", "EAST", "WEST") if can_move_basic(col, row, direction))

        for cur_col, cur_row, dist in queue:
            best_row = max(best_row, cur_row)
            found_crystal = found_crystal or crystals.get((cur_col, cur_row), 0) > 0
            if dist >= max_depth:
                continue
            for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
                if direction == "SOUTH" and cur_row <= south + 6:
                    continue
                if not can_move_basic(cur_col, cur_row, direction):
                    continue
                dc, dr = OFFSETS[direction]
                nxt = (cur_col + dc, cur_row + dr)
                if nxt in seen or nxt in occupied or nxt in reserved:
                    continue
                seen.add(nxt)
                queue.append((nxt[0], nxt[1], dist + 1))

        return best_row, len(seen), found_crystal, forward_exits

    def jump_landing_is_safe(col, row, max_depth=7):
        best_row, seen_count, found_crystal, forward_exits = jump_landing_stats(col, row, max_depth)
        if forward_exits <= 0:
            return False
        # A good jump landing must let us make normal northward progress soon.
        # A crystal landing is acceptable even with less progress because it
        # buys energy, but still requires more than a one-cell pocket.
        return best_row >= row + 2 or (found_crystal and best_row >= row + 1 and seen_count >= 5)

    def can_jump_north(col, row):
        target = (col, row + 2)
        if not in_bounds(*target) or target in occupied or target in reserved:
            return False
        if wall_at(*target) == 15:
            return False
        return has_exit(*target) and jump_landing_is_safe(*target)

    def can_emergency_jump_north(col, row):
        target = (col, row + 2)
        if not in_bounds(*target) or target in occupied or target in reserved:
            return False
        if wall_at(*target) == 15:
            return False
        best_row, seen_count, found_crystal, forward_exits = jump_landing_stats(*target, max_depth=7)
        if forward_exits <= 0:
            return False
        if best_row >= target[1] + 1:
            return True
        if projected_lava_gap(row, 3) <= 4 and seen_count >= 5:
            return True
        return found_crystal and seen_count >= 6

    def energy_mode(robot_type, energy):
        if robot_type == FACTORY:
            if energy < 350:
                return "REFUEL"
            if energy < 650:
                return "BALANCED"
            return "PROGRESS"
        if energy < getattr(config, "wallRemoveCost", 100) + 60:
            return "REFUEL"
        if energy < 220:
            return "BALANCED"
        return "PROGRESS"

    def recent_positions(uid, limit=8):
        return positions_memory.get(uid, [])[-limit:]

    def position_counts(uid, limit=12):
        counts = {}
        for pos in recent_positions(uid, limit):
            counts[pos] = counts.get(pos, 0) + 1
        return counts

    def action_target(col, row, action):
        if action in OFFSETS:
            dc, dr = OFFSETS[action]
            return col + dc, row + dr
        if action == "JUMP_NORTH":
            return col, row + 2
        return col, row

    def memory_penalty(uid, target):
        recent = recent_positions(uid, 8)
        penalty = 0.0
        if is_avoided(uid, target):
            penalty += 260.0
        if target not in recent:
            return penalty
        age = next(i for i, pos in enumerate(reversed(recent)) if pos == target)
        penalty += max(12.0, 90.0 - age * 12.0)
        penalty += max(0, position_counts(uid, 12).get(target, 0) - 1) * 45.0
        if len(recent) >= 2 and target == recent[-2]:
            penalty += 180.0
        return penalty

    def is_avoided(uid, target):
        return memory["avoid_until"].get(uid, {}).get(target, -1) > now_step

    def mark_avoided(uid, cells, ttl=18):
        avoid = memory["avoid_until"].setdefault(uid, {})
        expiry = now_step + ttl
        for cell in cells:
            if crystals.get(cell, 0) <= 0:
                avoid[cell] = max(avoid.get(cell, -1), expiry)
        for cell, until in list(avoid.items()):
            if until <= now_step:
                avoid.pop(cell, None)

    def loop_pressure(uid, col, row):
        recent = recent_positions(uid, 12)
        if len(recent) < 5:
            return False
        current = (col, row)
        if recent.count(current) >= 2:
            return True
        if len(recent) >= 6 and recent[-1] == recent[-3] and recent[-2] == recent[-4]:
            return True
        max_row = memory["max_rows"].get(uid, row)
        if row <= max_row and sum(1 for _, r in recent[-8:] if r <= max_row) >= 6:
            return True
        return side_cycle_loop(uid)

    def side_cycle_loop(uid):
        recent = recent_positions(uid, 8)
        if len(recent) < 5:
            return False
        rows = [r for _, r in recent[-5:]]
        if max(rows) != min(rows):
            return False
        if len(set(recent[-4:])) <= 2:
            return True
        # Fallback for the current memory model, which stores only the last
        # action: position alternation is enough to detect side ping-pong.
        return recent[-1] == recent[-3] and recent[-2] == recent[-4]

    def plateau_loop(uid):
        recent = recent_positions(uid, 16)
        if len(recent) < 10:
            return False
        rows = [r for _, r in recent[-10:]]
        cols = [c for c, _ in recent[-10:]]
        max_row = memory["max_rows"].get(uid, max(rows))
        if max(rows) - min(rows) <= 2 and max(rows) >= max_row - 1:
            return True
        if len(set(recent[-10:])) <= 6 and max(rows) >= max_row - 2:
            return True
        return len(set(cols)) <= 3 and max(rows) - min(rows) <= 3 and max(rows) >= max_row - 1

    def has_non_south_escape(col, row):
        for direction in ("NORTH", "EAST", "WEST"):
            if not can_move(col, row, direction):
                continue
            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            if direction == "NORTH" or has_exit(ncol, nrow) or crystals.get((ncol, nrow), 0) > 0:
                return True
        return False

    def has_fresh_non_south_escape(uid, col, row):
        if can_move(col, row, "NORTH"):
            return True
        recent = set(recent_positions(uid, 8))
        for direction in ("EAST", "WEST"):
            if not can_move(col, row, direction):
                continue
            target = action_target(col, row, direction)
            if target in recent and crystals.get(target, 0) <= 0:
                continue
            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            if has_exit(ncol, nrow) or crystals.get(target, 0) > 0:
                return True
        return False

    def has_any_safe_move(col, row):
        for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
            if direction == "SOUTH" and retreat_budget(row) <= 0:
                continue
            if can_move(col, row, direction):
                return True
        return False

    def lava_phase(step_offset=0):
        step = max(0.0, float(now_step + step_offset))
        return min(1.0, step / scroll_ramp_steps)

    def scroll_interval(step_offset=0):
        phase = lava_phase(step_offset)
        interval = scroll_start_interval + (scroll_end_interval - scroll_start_interval) * phase
        return max(scroll_end_interval, interval)

    def projected_scrolls(turns):
        turns = max(0, int(turns))
        expected = 0.0
        for offset in range(turns):
            expected += 1.0 / scroll_interval(offset)
        # Ceiling the expectation keeps late-game backtracking conservative.
        return int(expected + 0.999)

    def lava_gap(row):
        return row - south

    def projected_lava_gap(row, turns=0):
        return lava_gap(row) - projected_scrolls(turns)

    def lava_margin(extra=0):
        phase = lava_phase(extra)
        if phase >= 0.75:
            return 7
        if phase >= 0.45:
            return 6
        return 5

    def retreat_budget(row):
        # Keep enough buffer for the accelerating scroll. Early game can
        # backtrack a little; late game treats every detour as near-immediate
        # lava pressure.
        return max(0, min(5, projected_lava_gap(row, 2) - lava_margin(2)))

    def retreat_limit_row(row):
        return max(south + 4, row - retreat_budget(row))

    def enough_lava_budget(row, planned_steps):
        return projected_lava_gap(row, planned_steps + 1) >= max(4, lava_margin(planned_steps) - 1)

    def is_bad_memory_step(uid, col, row, action):
        if action not in OFFSETS:
            return False
        target = action_target(col, row, action)
        if crystals.get(target, 0) > 0:
            return False
        if is_avoided(uid, target):
            return True
        max_row = memory["max_rows"].get(uid, row)
        if action == "SOUTH":
            waits = memory["progress_waits"].get(uid, 0)
            if memory["actions"].get(uid) == "SOUTH" and crystals.get(target, 0) <= 0:
                return True
            if target[1] < max_row - 2 and crystals.get(target, 0) <= 0:
                return True
            if waits >= 2 and row <= max_row and crystals.get(target, 0) <= 0:
                return True
            if plateau_loop(uid) or side_cycle_loop(uid):
                if has_fresh_non_south_escape(uid, col, row):
                    return True
                return not enough_lava_budget(row, 2)
            # SOUTH without a crystal is only for real dead-end escape. If
            # north or a useful side move exists, taking SOUTH just creates
            # the loops visible in replay logs.
            if can_move(col, row, "NORTH"):
                return True
            for side in ("EAST", "WEST"):
                if not can_move(col, row, side):
                    continue
                dc, dr = OFFSETS[side]
                ncol, nrow = col + dc, row + dr
                side_target = (ncol, nrow)
                if (
                    side_target not in recent_positions(uid, 4)
                    and (has_exit(ncol, nrow) or crystals.get(side_target, 0) > 0)
                ):
                    return True
            if not has_non_south_escape(col, row):
                return False
            if row <= max_row and target in recent_positions(uid, 10):
                return True
            return target in recent_positions(uid, 6)
        recent = recent_positions(uid, 6)
        if target in recent[-2:]:
            return True
        return False

    def should_reject_step(uid, col, row, action, ignore_memory):
        if not action:
            return False
        # Never let ignore-memory override SOUTH loop protection. It is only
        # meant to free harmless side moves after IDLE stalls.
        if action == "SOUTH":
            return is_bad_memory_step(uid, col, row, action)
        return (not ignore_memory) and is_bad_memory_step(uid, col, row, action)

    def edge_cost(direction, row, next_col, next_row, mode, start_row):
        cost = 10.0
        crystal = crystals.get((next_col, next_row), 0)
        progress = next_row - row
        lava_gap = next_row - south

        if mode == "REFUEL":
            cost -= crystal * 2.7
            cost -= max(0, progress) * 4.0
        elif mode == "BALANCED":
            cost -= crystal * 1.3
            cost -= max(0, progress) * 9.0
        else:
            cost -= crystal * 0.45
            cost -= max(0, progress) * 17.0

        if direction == "NORTH":
            cost -= 4.0
        elif direction in ("EAST", "WEST"):
            cost += 2.5
        else:
            cost += 28.0

        # Avoid low rows. This is stronger near lava and still useful later.
        danger_margin = lava_margin(1) - 2
        caution_margin = lava_margin(2) + 1
        if lava_gap <= danger_margin:
            cost += 80.0
        elif lava_gap <= caution_margin:
            cost += 20.0

        # SOUTH is only acceptable for direct high-value energy while safe.
        if direction == "SOUTH" and (crystal <= 0 or start_row <= south + 7):
            cost += 300.0

        # Edges are often long pockets. Strongly prefer staying one or two
        # columns away unless the route proves northward progress/energy.
        edge_dist = min(next_col, width - 1 - next_col)
        if edge_dist == 0:
            cost += 45.0
            if crystal <= 0 and progress <= 0:
                cost += 150.0
        elif edge_dist == 1:
            cost += 22.0
            if crystal <= 0 and progress <= 0:
                cost += 45.0
        elif edge_dist == 2:
            cost += 8.0

        return max(0.5, cost)

    def target_value(col, row, dist_cost, mode, start_col, start_row):
        crystal = crystals.get((col, row), 0)
        progress = row - start_row
        lava_gap = row - south
        value = 0.0

        if mode == "REFUEL":
            value += crystal * 28.0
            value += max(0, progress) * 18.0
        elif mode == "BALANCED":
            value += crystal * 16.0
            value += max(0, progress) * 35.0
        else:
            value += crystal * 6.0
            value += max(0, progress) * 55.0

        value += projected_lava_gap(row, 3) * 5.0
        value -= abs(col - width // 2) * 1.5
        edge_dist = min(col, width - 1 - col)
        if edge_dist == 0:
            value -= 80.0
            if crystal <= 0 and progress <= 0:
                value -= 220.0
        elif edge_dist == 1:
            value -= 38.0
            if crystal <= 0 and progress <= 0:
                value -= 70.0
        elif edge_dist == 2:
            value -= 12.0
        value -= dist_cost * 1.8

        # A side move that cannot lead to higher row or energy should never win.
        if progress <= 0 and crystal <= 0:
            value -= 500.0

        return value

    def dijkstra_step(uid, robot_type, start_col, start_row, energy, max_depth):
        mode = energy_mode(robot_type, energy)
        start = (start_col, start_row)
        heap = [(0.0, 0, start_col, start_row, None)]
        best_cost = {start: 0.0}
        first_stats = {}
        best_candidate = None

        while heap:
            cost, depth, col, row, first = heappop(heap)
            if cost != best_cost.get((col, row)):
                continue

            if depth > 0:
                stats = first_stats.setdefault(first, {"max_row": start_row, "crystal": 0})
                stats["max_row"] = max(stats["max_row"], row)
                stats["crystal"] = max(stats["crystal"], crystals.get((col, row), 0))
                value = target_value(col, row, cost, mode, start_col, start_row)
                candidate = (value, row, crystals.get((col, row), 0), -depth, first)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate

            if depth >= max_depth:
                continue

            dirs = (
                ("NORTH", "WEST", "EAST", "SOUTH")
                if col < width // 2
                else ("NORTH", "EAST", "WEST", "SOUTH")
            )
            for direction in dirs:
                if not can_move_basic(col, row, direction):
                    continue
                dc, dr = OFFSETS[direction]
                ncol, nrow = col + dc, row + dr
                target = (ncol, nrow)
                if target != start and (target in occupied or target in reserved):
                    continue
                if is_avoided(uid, target) and crystals.get(target, 0) <= 0:
                    continue

                next_first = first or direction
                # Hard gate: never choose SOUTH as first action unless it is a
                # direct safe crystal pickup. This avoids lava-suicide routes.
                if next_first == "SOUTH":
                    direct = (start_col, start_row - 1)
                    if crystals.get(direct, 0) <= 0 or start_row <= south + 7:
                        continue

                next_cost = cost + edge_cost(direction, row, ncol, nrow, mode, start_row)
                if next_cost + 1e-9 < best_cost.get(target, 10**9):
                    best_cost[target] = next_cost
                    heappush(heap, (next_cost, depth + 1, ncol, nrow, next_first))

        if not best_candidate:
            return None

        step = best_candidate[-1]
        stats = first_stats.get(step, {"max_row": start_row, "crystal": 0})

        # Stateless anti-shuffle gate. A side step must prove it reaches
        # northward progress or visible energy; otherwise it is just ping-pong.
        if step in ("EAST", "WEST") and stats["max_row"] <= start_row and stats["crystal"] <= 0:
            return None

        return step

    def beam_step(uid, robot_type, start_col, start_row, energy, max_depth=18, beam_width=28):
        mode = energy_mode(robot_type, energy)
        start = (start_col, start_row)
        start_state = (0.0, start_col, start_row, None, (start,))
        beam = [start_state]
        best_candidate = None
        first_stats = {}

        for depth in range(1, max_depth + 1):
            candidates = []
            for score, col, row, first, path in beam:
                dirs = (
                    ("NORTH", "WEST", "EAST", "SOUTH")
                    if col < width // 2
                    else ("NORTH", "EAST", "WEST", "SOUTH")
                )
                for direction in dirs:
                    if direction == "SOUTH" and row <= south + 7:
                        continue
                    if not can_move_basic(col, row, direction):
                        continue

                    dc, dr = OFFSETS[direction]
                    ncol, nrow = col + dc, row + dr
                    target = (ncol, nrow)
                    if target in path:
                        continue
                    if target != start and (target in occupied or target in reserved):
                        continue
                    if is_avoided(uid, target) and crystals.get(target, 0) <= 0:
                        continue

                    next_first = first or direction
                    move_score = -edge_cost(direction, row, ncol, nrow, mode, start_row)
                    move_score += target_value(ncol, nrow, 0, mode, start_col, start_row) * 0.08
                    move_score -= memory_penalty(uid, target)
                    if direction == "NORTH":
                        move_score += 10.0
                    elif direction == "SOUTH":
                        move_score -= 80.0

                    new_score = score + move_score
                    new_path = path + (target,)
                    candidates.append((new_score, ncol, nrow, next_first, new_path))

                    stats = first_stats.setdefault(next_first, {"max_row": start_row, "crystal": 0})
                    stats["max_row"] = max(stats["max_row"], nrow)
                    stats["crystal"] = max(stats["crystal"], crystals.get(target, 0))

                    progress = nrow - start_row
                    crystal = crystals.get(target, 0)
                    terminal_value = new_score + progress * 35.0 + crystal * 12.0
                    candidate = (terminal_value, nrow, crystal, -depth, next_first)
                    if best_candidate is None or candidate > best_candidate:
                        best_candidate = candidate

            if not candidates:
                break
            candidates.sort(reverse=True, key=lambda item: item[0])
            beam = candidates[:beam_width]

        if not best_candidate:
            return None

        step = best_candidate[-1]
        stats = first_stats.get(step, {"max_row": start_row, "crystal": 0})
        if step in ("EAST", "WEST") and stats["max_row"] <= start_row and stats["crystal"] <= 0:
            return None
        return step

    def reserve(start, action):
        if action in OFFSETS:
            dc, dr = OFFSETS[action]
            target = (start[0] + dc, start[1] + dr)
        elif action == "JUMP_NORTH":
            target = (start[0], start[1] + 2)
        else:
            target = start
        occupied.discard(start)
        occupied.add(target)
        reserved.add(target)

    def step_toward_worker_target(col, row):
        if not factory_data:
            return None

        target_col = factory_data[1]
        target_row = min(north, factory_data[2] + 3)

        # Worker is mainly a snowplow. Move north when possible; otherwise
        # align with factory lane. Do not wander for crystals yet.
        prefs = []
        if row < target_row:
            prefs.append("NORTH")
        if col < target_col:
            prefs.append("EAST")
        elif col > target_col:
            prefs.append("WEST")
        prefs.extend(["NORTH", "EAST", "WEST"])

        seen = set()
        for direction in prefs:
            if direction in seen:
                continue
            seen.add(direction)
            if can_move(col, row, direction):
                return direction
        return None

    def direction_to(src_col, src_row, dst_col, dst_row):
        for direction, (dc, dr) in OFFSETS.items():
            if src_col + dc == dst_col and src_row + dr == dst_row:
                return direction
        return None

    def step_toward_target(col, row, target_col, target_row, allow_north_fallback=True):
        prefs = []
        if row < target_row:
            prefs.append("NORTH")
        elif row > target_row:
            prefs.append("SOUTH")
        if col < target_col:
            prefs.append("EAST")
        elif col > target_col:
            prefs.append("WEST")
        prefs.extend(["NORTH", "EAST", "WEST"])

        current_dist = abs(col - target_col) + abs(row - target_row)
        seen = set()
        for direction in prefs:
            if direction in seen:
                continue
            seen.add(direction)
            if direction == "SOUTH" and row <= south + 6:
                continue
            if not can_move(col, row, direction):
                continue
            dc, dr = OFFSETS[direction]
            next_dist = abs(col + dc - target_col) + abs(row + dr - target_row)
            if next_dist < current_dist or (allow_north_fallback and direction == "NORTH"):
                return direction
        return None

    def nearest_reachable_crystal_step(col, row, max_depth=8):
        if not crystals:
            return None

        queue = [(col, row, 0, None)]
        seen = {(col, row)}
        best = None

        for cur_col, cur_row, dist, first in queue:
            value = crystals.get((cur_col, cur_row), 0)
            if dist > 0 and value > 0:
                cand = (value * 10 - dist * 4 + (cur_row - row) * 3, first)
                if best is None or cand > best:
                    best = cand
            if dist >= max_depth:
                continue
            for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
                if direction == "SOUTH" and cur_row <= south + 6:
                    continue
                if not can_move_basic(cur_col, cur_row, direction):
                    continue
                dc, dr = OFFSETS[direction]
                nxt = (cur_col + dc, cur_row + dr)
                if nxt in seen or nxt in occupied or nxt in reserved:
                    continue
                seen.add(nxt)
                queue.append((nxt[0], nxt[1], dist + 1, first or direction))

        return best[-1] if best else None

    def scout_collect_opportunity(factory_col, factory_row):
        if not crystals:
            return False
        if projected_lava_gap(factory_row, 6) < 8:
            return False

        spawn_col, spawn_row = factory_col, factory_row + 1
        best_score = None
        for (c_col, c_row), value in crystals.items():
            if value <= 0:
                continue
            # Scout is the cheap collector for crystals the factory should not
            # reverse for: same row, behind, or low-side detours while safe.
            if c_row > factory_row + 2:
                continue
            if c_row <= south + 4:
                continue
            drop = max(0, factory_row - c_row)
            if drop > retreat_budget(factory_row) + 6:
                continue
            dist = abs(c_col - spawn_col) + abs(c_row - spawn_row)
            if dist > 10:
                continue
            score = value * 4 - dist * 7 - drop * 5
            if c_row <= factory_row:
                score += 45
            if abs(c_col - factory_col) <= 2:
                score += 15
            best_score = score if best_score is None else max(best_score, score)

        return best_score is not None and best_score >= 45

    def worker_drill_opportunity(uid, factory_col, factory_row, jump_cd, idle_count):
        # A worker can only spawn north of the factory. Use it when the next
        # cell is open but immediately capped by a north wall, which is the
        # common "wait for jump" pocket a drill can actually fix.
        if projected_lava_gap(factory_row, 5) < 7:
            return False
        spawn_col, spawn_row = factory_col, factory_row + 1
        if not in_bounds(spawn_col, spawn_row):
            return False
        if wall_at(spawn_col, spawn_row) & NORTH == 0:
            return False
        stuck = (
            idle_count >= 2
            or memory["progress_waits"].get(uid, 0) >= 4
            or loop_pressure(uid, factory_col, factory_row)
            or plateau_loop(uid)
            or side_cycle_loop(uid)
            or (jump_cd <= 1 and (can_jump_north(factory_col, factory_row) or can_emergency_jump_north(factory_col, factory_row)))
        )
        return stuck

    def factory_worker_transfer_step(factory_energy, factory_col, factory_row):
        wall_cost = getattr(config, "wallRemoveCost", 100)
        if factory_energy < 850:
            return None
        if projected_lava_gap(factory_row, 3) <= 4:
            return None

        best = None
        for data in my.values():
            if data[0] != WORKER or data[3] <= 0:
                continue
            worker_col, worker_row, worker_energy = data[1], data[2], data[3]
            direction = adjacent_direction(factory_col, factory_row, worker_col, worker_row)
            if direction is None or not can_move_basic(factory_col, factory_row, direction):
                continue
            if projected_lava_gap(worker_row, 3) <= 2:
                continue

            worker_wall = wall_at(worker_col, worker_row)
            drill_job = bool(worker_wall & NORTH) or worker_row >= factory_row
            if not drill_job:
                continue
            if worker_energy >= wall_cost + 60:
                continue

            need = max(0, wall_cost + 60 - worker_energy)
            score = need + max(0, worker_row - factory_row) * 12
            if worker_wall & NORTH:
                score += 80
            candidate = (score, f"TRANSFER_{direction}")
            if best is None or candidate > best:
                best = candidate

        return best[1] if best else None

    def astar_path_first_step(uid, start_col, start_row, goal_col, goal_row, max_dist=10):
        goal = (goal_col, goal_row)
        heap = [(abs(goal_col - start_col) + abs(goal_row - start_row), 0, start_col, start_row, None)]
        best_dist = {(start_col, start_row): 0}

        while heap:
            _, dist, col, row, first = heappop(heap)
            if dist != best_dist.get((col, row)):
                continue
            if (col, row) == goal:
                return first
            if dist >= max_dist:
                continue

            for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
                if direction == "SOUTH" and row <= south + 6:
                    continue
                if not can_move_basic(col, row, direction):
                    continue
                dc, dr = OFFSETS[direction]
                ncol, nrow = col + dc, row + dr
                target = (ncol, nrow)
                if target != goal and (target in occupied or target in reserved):
                    continue
                if target != goal and is_avoided(uid, target) and crystals.get(target, 0) <= 0:
                    continue
                ndist = dist + 1
                if ndist >= best_dist.get(target, 10**9):
                    continue
                next_first = first or direction
                if crystals.get(target, 0) <= 0 and target in recent_positions(uid, 3):
                    continue
                best_dist[target] = ndist
                priority = ndist + abs(goal_col - ncol) + abs(goal_row - nrow)
                heappush(heap, (priority, ndist, ncol, nrow, next_first))

        return None

    def future_reach_score(uid, start_col, start_row, origin_row, max_depth=8):
        queue = [(start_col, start_row, 0)]
        seen = {(start_col, start_row)}
        best_row = start_row
        crystal_gain = crystals.get((start_col, start_row), 0)
        no_regress_floor = max(south + 4, origin_row - 2)

        for cur_col, cur_row, dist in queue:
            best_row = max(best_row, cur_row)
            if dist >= max_depth:
                continue
            for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
                if direction == "SOUTH" and cur_row - 1 < no_regress_floor:
                    continue
                if not can_move_basic(cur_col, cur_row, direction):
                    continue
                dc, dr = OFFSETS[direction]
                ncol, nrow = cur_col + dc, cur_row + dr
                target = (ncol, nrow)
                if target in seen:
                    continue
                if target != (start_col, start_row) and (target in occupied or target in reserved):
                    continue
                crystal = crystals.get(target, 0)
                if is_avoided(uid, target) and crystal <= 0:
                    continue
                if direction == "SOUTH" and crystal <= 0 and not enough_lava_budget(origin_row, dist + 2):
                    continue
                seen.add(target)
                crystal_gain += crystal
                queue.append((ncol, nrow, dist + 1))

        future_gain = best_row - origin_row
        pocket_penalty = 0
        if best_row <= origin_row and len(seen) <= 4:
            pocket_penalty += 90
        elif best_row <= start_row and len(seen) <= 6:
            pocket_penalty += 45
        if start_row < origin_row and best_row <= origin_row:
            pocket_penalty += 40
        return future_gain * 28.0 + crystal_gain * 0.8 + len(seen) * 2.0 - pocket_penalty

    def crystal_astar_step(uid, col, row, energy):
        if not crystals:
            return None

        best = None
        for (c_col, c_row), value in crystals.items():
            manhattan = abs(c_col - col) + abs(c_row - row)
            if manhattan > 8:
                continue
            if c_row < row - 2 and row <= south + 8:
                continue
            first = astar_path_first_step(uid, col, row, c_col, c_row, max_dist=10)
            if not first:
                continue

            south_drop = max(0, row - c_row)
            progress = c_row - row
            future_score = future_reach_score(uid, c_col, c_row, row, max_depth=8)
            score = (
                value * 2.4
                + max(0, progress) * 8.0
                + future_score
                - manhattan * 7.0
                - south_drop * 22.0
            )
            if energy < 700:
                score += value * 1.2
            if first == "SOUTH":
                score -= 45.0
            if future_score < 0 and value < 35:
                continue
            candidate = (score, value, -manhattan, first)
            if score > 12.0 and (best is None or candidate > best):
                best = candidate

        return best[-1] if best else None

    def scout_crystal_step(uid, col, row):
        if not crystals:
            return None

        best = None
        for (c_col, c_row), value in crystals.items():
            if value <= 0:
                continue
            if c_row <= south + 3:
                continue
            manhattan = abs(c_col - col) + abs(c_row - row)
            if manhattan > 12:
                continue
            drop = max(0, row - c_row)
            if drop > retreat_budget(row) + 5:
                continue
            first = astar_path_first_step(uid, col, row, c_col, c_row, max_dist=12)
            if not first:
                continue
            future_score = future_reach_score(uid, c_col, c_row, row, max_depth=7)
            score = value * 3.0 + future_score - manhattan * 5.0 - drop * 5.0
            if c_row <= row:
                score += 35.0
            if first == "SOUTH":
                score += 12.0
            candidate = (score, value, -manhattan, first)
            if score > 20.0 and (best is None or candidate > best):
                best = candidate

        return best[-1] if best else None

    def scout_unblock_factory_step(uid, col, row):
        if not factory_data:
            return None
        f_col, f_row = factory_data[1], factory_data[2]
        if (col, row) != (f_col, f_row + 1):
            return None
        side_dirs = ("WEST", "EAST") if col >= width // 2 else ("EAST", "WEST")
        for direction in side_dirs:
            if can_move(col, row, direction):
                target = action_target(col, row, direction)
                if target not in recent_positions(uid, 4):
                    return direction
        return None

    def scout_explore_step(uid, col, row, energy):
        if projected_lava_gap(row, 3) <= 5:
            step = simple_progress_step(col, row)
            if step:
                return step
        step = scout_unblock_factory_step(uid, col, row)
        if step:
            return step
        step = beam_step(uid, SCOUT, col, row, energy, max_depth=10, beam_width=16)
        if step:
            return step
        return dijkstra_step(uid, SCOUT, col, row, energy, max_depth=10)

    def progress_route_step(uid, col, row, max_depth=10):
        max_row = memory["max_rows"].get(uid, row)
        target_row = max(row + 2, max_row + 1)
        no_regress_floor = max(south + 4, max_row - 2)
        recent = set(recent_positions(uid, 8))
        queue = [(col, row, 0, None, 0)]
        seen = {(col, row)}
        best = None

        for cur_col, cur_row, dist, first, crystal_gain in queue:
            if dist > 0 and cur_row >= target_row:
                repeat_penalty = 35 if (cur_col, cur_row) in recent else 0
                cand = (
                    cur_row * 100
                    + crystal_gain * 3
                    - dist * 10
                    - repeat_penalty,
                    -dist,
                    first,
                )
                if best is None or cand > best:
                    best = cand
                continue

            if dist >= max_depth:
                continue

            dirs = (
                ("NORTH", "EAST", "WEST", "SOUTH")
                if cur_col < width // 2
                else ("NORTH", "WEST", "EAST", "SOUTH")
            )
            for direction in dirs:
                if not can_move_basic(cur_col, cur_row, direction):
                    continue
                dc, dr = OFFSETS[direction]
                ncol, nrow = cur_col + dc, cur_row + dr
                target = (ncol, nrow)
                if target in seen:
                    continue
                if target != (col, row) and (target in occupied or target in reserved):
                    continue
                crystal = crystals.get(target, 0)
                if direction == "SOUTH":
                    if crystal <= 0 and nrow < no_regress_floor:
                        continue
                    if not enough_lava_budget(row, dist + 2):
                        continue
                if is_avoided(uid, target) and crystal <= 0:
                    continue
                if crystal <= 0 and target in recent and nrow <= max_row:
                    continue

                seen.add(target)
                queue.append((ncol, nrow, dist + 1, first or direction, crystal_gain + crystal))

        if not best:
            return None
        step = best[-1]
        if step == "SOUTH" and crystals.get(action_target(col, row, step), 0) <= 0:
            return None
        return step

    def simple_progress_step(col, row):
        if col <= 1:
            side_dirs = ("EAST", "WEST")
        elif col >= width - 2:
            side_dirs = ("WEST", "EAST")
        else:
            side_dirs = ("WEST", "EAST") if col >= width // 2 else ("EAST", "WEST")
        for direction in ("NORTH",) + side_dirs:
            if not can_move(col, row, direction):
                continue
            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            if direction == "NORTH" or has_exit(ncol, nrow) or crystals.get((ncol, nrow), 0) > 0:
                return direction
        return None

    def wall_bypass_step(col, row):
        if can_move(col, row, "NORTH"):
            return "NORTH"

        side_dirs = ("WEST", "EAST") if col >= width // 2 else ("EAST", "WEST")
        if col <= 1:
            side_dirs = ("EAST", "WEST")
        elif col >= width - 2:
            side_dirs = ("WEST", "EAST")

        for direction in side_dirs:
            if not can_move(col, row, direction):
                continue
            dc, dr = OFFSETS[direction]
            one_col, one_row = col + dc, row + dr
            if can_move_basic(one_col, one_row, "NORTH"):
                return direction

            two_col, two_row = one_col + dc, one_row + dr
            if (
                in_bounds(two_col, two_row)
                and not (wall_at(one_col, one_row) & WALL_BITS[direction])
                and (two_col, two_row) not in occupied
                and (two_col, two_row) not in reserved
                and can_move_basic(two_col, two_row, "NORTH")
            ):
                return direction

        return None

    def adjacent_crystal_step(col, row):
        choices = []
        for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
            if not can_move(col, row, direction):
                continue
            target = action_target(col, row, direction)
            value = crystals.get(target, 0)
            if value <= 0:
                continue
            if direction == "SOUTH" and (
                retreat_budget(row) <= 0
                or not enough_lava_budget(row, 2)
                or row <= south + 6
            ):
                continue

            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            score = value * 10 + (nrow - row) * 18
            if direction == "SOUTH":
                score -= 25
                # Only take a one-step-down crystal if the landing cell still
                # has a way out. Otherwise the energy is not worth a lava trap.
                if not any(
                    can_move_basic(ncol, nrow, escape_dir)
                    for escape_dir in ("NORTH", "EAST", "WEST")
                ):
                    continue
            choices.append((score, direction))

        if not choices:
            return None
        choices.sort(reverse=True)
        return choices[0][1]

    def deadend_escape_step(col, row):
        if has_non_south_escape(col, row):
            return None
        if retreat_budget(row) <= 0:
            return None
        if can_move(col, row, "SOUTH"):
            return "SOUTH"
        return None

    def after_south_escape_step(uid, col, row):
        if memory["actions"].get(uid) != "SOUTH":
            return None

        side_dirs = ("WEST", "EAST") if col >= width // 2 else ("EAST", "WEST")
        if col <= 1:
            side_dirs = ("EAST", "WEST")
        elif col >= width - 2:
            side_dirs = ("WEST", "EAST")

        for direction in side_dirs:
            if not can_move(col, row, direction):
                continue
            target = action_target(col, row, direction)
            if target in recent_positions(uid, 4) and crystals.get(target, 0) <= 0:
                continue
            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            if has_exit(ncol, nrow) or crystals.get((ncol, nrow), 0) > 0:
                return direction

        if can_move(col, row, "NORTH"):
            target = action_target(col, row, "NORTH")
            if target not in recent_positions(uid, 3) or crystals.get(target, 0) > 0:
                return "NORTH"
        return None

    def trap_escape_step(uid, col, row):
        if not loop_pressure(uid, col, row):
            return None

        recent = recent_positions(uid, 12)
        counts = position_counts(uid, 12)
        max_row = memory["max_rows"].get(uid, row)
        trap_cells = [cell for cell in recent[-8:] if cell[1] <= max_row and counts.get(cell, 0) >= 2]
        if plateau_loop(uid) or side_cycle_loop(uid):
            trap_cells.extend(cell for cell in recent[-10:] if cell[1] >= max_row - 3)
        mark_avoided(uid, trap_cells, ttl=20)

        last_action = memory["actions"].get(uid)
        reverse = {
            "NORTH": "SOUTH",
            "SOUTH": "NORTH",
            "EAST": "WEST",
            "WEST": "EAST",
        }.get(last_action)

        choices = []
        for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
            if direction == "SOUTH" and (retreat_budget(row) <= 0 or not enough_lava_budget(row, 2)):
                continue
            if direction == "SOUTH" and (plateau_loop(uid) or side_cycle_loop(uid)):
                continue
            if not can_move(col, row, direction):
                continue

            target = action_target(col, row, direction)
            crystal = crystals.get(target, 0)
            if is_avoided(uid, target) and crystal <= 0:
                continue

            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            visits = counts.get(target, 0)
            edge_dist = min(ncol, width - 1 - ncol)
            score = crystal * 5 + (nrow - row) * 28 - visits * 55
            if target not in recent:
                score += 70
            if has_exit(ncol, nrow):
                score += 16
            if direction == reverse:
                score -= 80
            if direction == "SOUTH":
                score -= 95
                if target in recent or row <= memory["max_rows"].get(uid, row):
                    score -= 75
            if edge_dist == 0:
                score -= 85
            elif edge_dist == 1:
                score -= 36
            choices.append((score, direction))

        if not choices:
            return None
        choices.sort(reverse=True)
        step = choices[0][1]
        if step in ("SOUTH", "EAST", "WEST"):
            start_escape_commit(uid, col, row, step)
        return step

    def safe_backtrack_step(uid, col, row, max_depth=18):
        min_safe_row = retreat_limit_row(row)
        if retreat_budget(row) <= 0:
            return None

        queue = [(col, row, 0, None)]
        seen = {(col, row)}
        best = None
        recent = set(recent_positions(uid, 8))

        for cur_col, cur_row, dist, first in queue:
            if dist > 0 and cur_row > row:
                repeat_penalty = 35 if (cur_col, cur_row) in recent else 0
                cand = (
                    (cur_row - row) * 120
                    + crystals.get((cur_col, cur_row), 0) * 8
                    - dist * 7
                    - repeat_penalty,
                    -dist,
                    first,
                )
                if best is None or cand > best:
                    best = cand

            if dist >= max_depth or not enough_lava_budget(row, dist + 1):
                continue

            dirs = ("NORTH", "EAST", "WEST", "SOUTH")
            for direction in dirs:
                if direction == "SOUTH" and cur_row - 1 <= min_safe_row:
                    continue
                if not can_move_basic(cur_col, cur_row, direction):
                    continue
                dc, dr = OFFSETS[direction]
                nxt = (cur_col + dc, cur_row + dr)
                if nxt in seen or nxt in occupied or nxt in reserved:
                    continue
                seen.add(nxt)
                queue.append((nxt[0], nxt[1], dist + 1, first or direction))

        if best:
            return best[-1]
        south_target = action_target(col, row, "SOUTH")
        if (
            can_move(col, row, "SOUTH")
            and row - 1 > min_safe_row
            and (south_target not in recent_positions(uid, 4) or crystals.get(south_target, 0) > 0)
        ):
            return "SOUTH"
        return None

    def escape_commit_step(uid, col, row):
        state = memory["escape"].get(uid)
        if not state:
            return None
        origin_col = state.get("origin_col", col)
        origin_row = state.get("origin_row", row)
        if row > origin_row:
            memory["escape"].pop(uid, None)
            return None
        if state.get("ttl", 0) <= 0:
            memory["escape"].pop(uid, None)
            return None

        direction = state.get("direction")
        state["ttl"] -= 1

        side_options = [direction]
        if direction == "EAST":
            side_options.append("WEST")
        elif direction == "WEST":
            side_options.append("EAST")

        for side in side_options:
            if side in ("EAST", "WEST") and can_move(col, row, side):
                target = action_target(col, row, side)
                if target not in recent_positions(uid, 6) or crystals.get(target, 0) > 0:
                    state["direction"] = side
                    return side

        side_distance = abs(col - origin_col)
        if can_move(col, row, "NORTH"):
            target = action_target(col, row, "NORTH")
            if (
                side_distance >= 2
                and (target not in recent_positions(uid, 5) or row + 1 > origin_row)
            ):
                return "NORTH"

        if retreat_budget(row) > 0 and can_move(col, row, "SOUTH"):
            target = action_target(col, row, "SOUTH")
            if target not in recent_positions(uid, 5) or crystals.get(target, 0) > 0:
                return "SOUTH"

        return None

    def start_escape_commit(uid, col, row, first_step):
        if first_step not in OFFSETS:
            return
        if first_step in ("EAST", "WEST"):
            direction = first_step
        else:
            side_dirs = ("WEST", "EAST") if col >= width // 2 else ("EAST", "WEST")
            direction = next((d for d in side_dirs if can_move(col, row, d)), None)
            if not direction:
                return
        memory["escape"][uid] = {
            "direction": direction,
            "origin_col": col,
            "origin_row": row,
            "ttl": 8,
        }

    def edge_escape_step(uid, col, row):
        # Do not peel away from a valid north path just because we are near an
        # edge. Edge escape is only for actual edge traps or detected loops.
        if can_move(col, row, "NORTH") and not loop_pressure(uid, col, row):
            return None

        if col <= 1 and can_move(col, row, "EAST"):
            target = action_target(col, row, "EAST")
            if is_avoided(uid, target) and crystals.get(target, 0) <= 0:
                return None
            return "EAST"
        if col >= width - 2 and can_move(col, row, "WEST"):
            target = action_target(col, row, "WEST")
            if is_avoided(uid, target) and crystals.get(target, 0) <= 0:
                return None
            return "WEST"
        return None

    def least_recent_legal_step(uid, col, row):
        if projected_lava_gap(row, 2) <= 4:
            return None

        recent = recent_positions(uid, 12)
        side_dirs = ("WEST", "EAST") if col >= width // 2 else ("EAST", "WEST")
        if col <= 1:
            side_dirs = ("EAST", "WEST")
        elif col >= width - 2:
            side_dirs = ("WEST", "EAST")

        choices = []
        directions = (
            ("NORTH", "SOUTH") + side_dirs
            if plateau_loop(uid) or side_cycle_loop(uid)
            else side_dirs + ("SOUTH", "NORTH")
        )

        for direction in directions:
            if direction == "SOUTH" and retreat_budget(row) <= 0:
                continue
            if not can_move(col, row, direction):
                continue
            target = action_target(col, row, direction)
            try:
                age = next(i for i, pos in enumerate(reversed(recent)) if pos == target)
            except StopIteration:
                age = 99
            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            score = age * 10 + crystals.get(target, 0) * 3 + (nrow - row) * 8
            edge_dist = min(ncol, width - 1 - ncol)
            if edge_dist == 0:
                score -= 120
            elif edge_dist == 1:
                score -= 70
            if direction in side_dirs and has_exit(ncol, nrow):
                score += 18
            if direction == "SOUTH":
                score -= 20
            if direction == "NORTH" and target in recent[-4:]:
                score -= 35
            choices.append((score, direction))

        if not choices:
            return None
        choices.sort(reverse=True)
        step = choices[0][1]
        if step in ("SOUTH", "EAST", "WEST"):
            start_escape_commit(uid, col, row, step)
        return step

    def forced_unstick_step(uid, col, row):
        if projected_lava_gap(row, 2) <= 4:
            return None

        recent = recent_positions(uid, 12)
        counts = position_counts(uid, 12)
        mark_avoided(uid, recent[-6:], ttl=10)
        last_action = memory["actions"].get(uid)
        reverse = {
            "NORTH": "SOUTH",
            "SOUTH": "NORTH",
            "EAST": "WEST",
            "WEST": "EAST",
        }.get(last_action)

        choices = []
        for direction in ("NORTH", "EAST", "WEST", "SOUTH"):
            if direction == "SOUTH" and retreat_budget(row) <= 0:
                continue
            if (
                direction == "SOUTH"
                and (plateau_loop(uid) or side_cycle_loop(uid))
                and crystals.get(action_target(col, row, direction), 0) <= 0
                and has_fresh_non_south_escape(uid, col, row)
            ):
                continue
            if not can_move(col, row, direction):
                continue

            target = action_target(col, row, direction)
            crystal = crystals.get(target, 0)
            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            edge_dist = min(ncol, width - 1 - ncol)
            visits = counts.get(target, 0)

            score = crystal * 8 + (nrow - row) * 45
            if direction == "NORTH":
                score += 65
            elif direction in ("EAST", "WEST"):
                score += 12
                if has_exit(ncol, nrow):
                    score += 18
            else:
                score -= 65
                if not enough_lava_budget(row, 2):
                    continue

            if target not in recent:
                score += 45
            else:
                score -= visits * 22
            if is_avoided(uid, target) and crystal <= 0:
                score -= 60
            if direction == reverse:
                score -= 45
            if edge_dist == 0:
                score -= 110
            elif edge_dist == 1:
                score -= 45

            choices.append((score, direction))

        if not choices:
            return None
        choices.sort(reverse=True)
        step = choices[0][1]
        if step in ("SOUTH", "EAST", "WEST"):
            start_escape_commit(uid, col, row, step)
        return step

    def progress_deadline_jump(uid, col, row, step):
        if jump_cd > 1:
            return False
        if can_move(col, row, "NORTH"):
            return False

        target = action_target(col, row, step) if step else (col, row)
        if step in OFFSETS and crystals.get(target, 0) > 0:
            return False

        waits = memory["progress_waits"].get(uid, 0)
        gap = projected_lava_gap(row, 4)
        looped = loop_pressure(uid, col, row) or plateau_loop(uid) or side_cycle_loop(uid)
        if gap <= 6:
            return can_emergency_jump_north(col, row)
        if gap <= 8 and waits >= 2:
            return can_jump_north(col, row) or can_emergency_jump_north(col, row)
        if gap <= 11 and waits >= 4 and looped:
            return can_jump_north(col, row) or can_emergency_jump_north(col, row)
        if lava_phase(4) >= 0.75 and waits >= 2 and looped:
            return can_jump_north(col, row) or can_emergency_jump_north(col, row)
        if waits >= 8 and looped:
            return can_jump_north(col, row)
        return False

    def worker_blocks_factory_north(col, row):
        for data in my.values():
            if data[0] == WORKER and data[1] == col and data[2] == row + 1 and data[3] > 0:
                return True
        return False

    def side_wait_for_worker_step(uid, col, row):
        side_dirs = ("WEST", "EAST") if col >= width // 2 else ("EAST", "WEST")
        if col <= 1:
            side_dirs = ("EAST", "WEST")
        elif col >= width - 2:
            side_dirs = ("WEST", "EAST")
        recent = recent_positions(uid, 8)

        choices = []
        for direction in side_dirs:
            if not can_move(col, row, direction):
                continue
            target = action_target(col, row, direction)
            if is_avoided(uid, target) and crystals.get(target, 0) <= 0:
                continue
            dc, dr = OFFSETS[direction]
            ncol, nrow = col + dc, row + dr
            score = crystals.get(target, 0) * 6
            if target not in recent:
                score += 35
            if has_exit(ncol, nrow):
                score += 12
            choices.append((score, direction))

        if not choices:
            return None
        choices.sort(reverse=True)
        step = choices[0][1]
        start_escape_commit(uid, col, row, step)
        return step

    for uid, data in sorted(my.items(), key=lambda item: (0 if item[1][0] == FACTORY else 1, str(item[0]))):
        robot_type, col, row = data[0], data[1], data[2]
        energy = data[3]
        move_cd = data[5] if len(data) > 5 else 0
        jump_cd = data[6] if len(data) > 6 else 0
        build_cd = data[7] if len(data) > 7 else 0
        start = (col, row)

        if robot_type == FACTORY:
            worker_count = sum(1 for d in my.values() if d[0] == WORKER and d[3] > 0)
            scout_count = sum(1 for d in my.values() if d[0] == SCOUT and d[3] > 0)
            spawn = (col, row + 1)
            north_blocked = bool(wall_at(col, row) & NORTH)
            idle_count = memory["idle_counts"].get(uid, 0)
            worker_ahead = worker_blocks_factory_north(col, row)

            # Jump is an escape tool, not the default wall response. If lava is
            # not close, try to route around first and keep jump cooldown ready.
            if jump_cd <= 1 and projected_lava_gap(row, 4) <= 6 and not can_move(col, row, "NORTH"):
                if can_jump_north(col, row) or can_emergency_jump_north(col, row):
                    actions[uid] = "JUMP_NORTH"
                    reserve(start, "JUMP_NORTH")
                    continue

            transfer_step = factory_worker_transfer_step(energy, col, row)
            if transfer_step:
                actions[uid] = transfer_step
                reserve(start, transfer_step)
                continue

            # Build a worker only as a stall breaker/drill. If the factory is
            # trapped in a pocket where jump is the only progress tool, a worker
            # can spawn north and remove the next wall instead.
            if (
                (idle_count >= 3 or worker_drill_opportunity(uid, col, row, jump_cd, idle_count))
                and build_cd <= 1
                and worker_count < 1
                and energy >= getattr(config, "workerCost", 200) + 450
                and projected_lava_gap(row, 6) >= 7
                and can_move_basic(col, row, "NORTH")
                and spawn not in occupied
                and spawn not in reserved
            ):
                actions[uid] = "BUILD_WORKER"
                reserved.add(spawn)
                continue

            if (
                scout_collect_opportunity(col, row)
                and build_cd <= 1
                and scout_count < 1
                and worker_count == 0
                and idle_count == 0
                and not loop_pressure(uid, col, row)
                and energy >= getattr(config, "scoutCost", 50) + 1050
                and projected_lava_gap(row, 8) >= 12
                and jump_cd > 6
                and can_move_basic(col, row, "NORTH")
                and spawn not in occupied
                and spawn not in reserved
            ):
                actions[uid] = "BUILD_SCOUT"
                reserved.add(spawn)
                continue

            if move_cd <= 1:
                step = progress_route_step(uid, col, row, max_depth=10)
                if step and step in ("EAST", "WEST", "SOUTH"):
                    start_escape_commit(uid, col, row, step)
                if should_reject_step(uid, col, row, step, False):
                    step = None
                if not step:
                    step = adjacent_crystal_step(col, row)
                if not step:
                    step = edge_escape_step(uid, col, row)
                if step:
                    start_escape_commit(uid, col, row, step)
                if not step:
                    step = escape_commit_step(uid, col, row)
                if not step:
                    step = trap_escape_step(uid, col, row)
                ignore_memory = idle_count >= 2
                if not step and ignore_memory:
                    step = wall_bypass_step(col, row) or simple_progress_step(col, row)
                if not step:
                    step = after_south_escape_step(uid, col, row)
                if not step and north_blocked:
                    step = wall_bypass_step(col, row)
                if not step:
                    step = crystal_astar_step(uid, col, row, energy)
                if not step:
                    step = beam_step(uid, robot_type, col, row, energy, max_depth=18, beam_width=28)
                if not step:
                    step = dijkstra_step(uid, robot_type, col, row, energy, max_depth=16)
                if should_reject_step(uid, col, row, step, ignore_memory):
                    step = None
                if not step and north_blocked:
                    step = wall_bypass_step(col, row)
                if should_reject_step(uid, col, row, step, ignore_memory):
                    step = None
                if not step:
                    step = simple_progress_step(col, row)
                if should_reject_step(uid, col, row, step, ignore_memory):
                    step = None
                if not step:
                    step = safe_backtrack_step(uid, col, row, max_depth=18)
                    if step in ("SOUTH", "EAST", "WEST"):
                        start_escape_commit(uid, col, row, step)
                if should_reject_step(uid, col, row, step, ignore_memory):
                    step = None
                if not step:
                    step = deadend_escape_step(col, row)
                    if step == "SOUTH":
                        start_escape_commit(uid, col, row, step)
                if should_reject_step(uid, col, row, step, ignore_memory):
                    step = None
                if not step:
                    step = least_recent_legal_step(uid, col, row)
                    if step in ("SOUTH", "EAST", "WEST"):
                        start_escape_commit(uid, col, row, step)
                if not step and idle_count >= 2 and jump_cd > 1:
                    step = forced_unstick_step(uid, col, row)
                if step == "SOUTH" and worker_ahead and crystals.get(action_target(col, row, step), 0) <= 0:
                    step = side_wait_for_worker_step(uid, col, row)
                    if not step:
                        actions[uid] = "IDLE"
                        continue
                if progress_deadline_jump(uid, col, row, step):
                    memory["escape"].pop(uid, None)
                    mark_avoided(uid, recent_positions(uid, 8), ttl=12)
                    actions[uid] = "JUMP_NORTH"
                    reserve(start, "JUMP_NORTH")
                    continue
                if step and can_move(col, row, step):
                    actions[uid] = step
                    reserve(start, step)
                    continue

            # Late escape when lava budget is too small for another search.
            if (
                jump_cd <= 1
                and (can_jump_north(col, row) or (idle_count >= 4 and can_emergency_jump_north(col, row)))
                and (projected_lava_gap(row, 4) <= 6 or idle_count >= 3 or not has_any_safe_move(col, row))
            ):
                actions[uid] = "JUMP_NORTH"
                reserve(start, "JUMP_NORTH")
                continue

            actions[uid] = "IDLE"
            continue

        if robot_type == SCOUT:
            if energy <= 0:
                actions[uid] = "IDLE"
                continue
            if move_cd <= 1:
                step = scout_crystal_step(uid, col, row)
                if should_reject_step(uid, col, row, step, False):
                    step = None
                if not step:
                    step = scout_explore_step(uid, col, row, energy)
                if should_reject_step(uid, col, row, step, False):
                    step = None
                if step and can_move(col, row, step):
                    actions[uid] = step
                    reserve(start, step)
                    continue
            actions[uid] = "IDLE"
            continue

        if robot_type == WORKER:
            if energy <= 0:
                actions[uid] = "IDLE"
                continue

            wall = wall_at(col, row)
            if factory_data and row <= factory_data[2]:
                step = step_toward_target(col, row, factory_data[1], factory_data[2] + 2)
                if step and can_move(col, row, step):
                    actions[uid] = step
                    reserve(start, step)
                    continue

            if (wall & NORTH) and energy >= getattr(config, "wallRemoveCost", 100) and in_bounds(col, row + 1):
                actions[uid] = "REMOVE_NORTH"
                continue

            wall_cost = getattr(config, "wallRemoveCost", 100)
            if factory_data:
                direction_to_worker = adjacent_direction(factory_data[1], factory_data[2], col, row)
                can_receive_transfer = (
                    direction_to_worker is not None
                    and can_move_basic(factory_data[1], factory_data[2], direction_to_worker)
                    and factory_data[3] >= 850
                    and projected_lava_gap(factory_data[2], 3) > 4
                )
                if can_receive_transfer and energy < wall_cost + 60 and ((wall & NORTH) or row >= factory_data[2]):
                    actions[uid] = "IDLE"
                    continue

            if move_cd <= 1:
                step = nearest_reachable_crystal_step(col, row, max_depth=10)
                if step and can_move(col, row, step):
                    actions[uid] = step
                    reserve(start, step)
                    continue

            if energy < wall_cost + 30:
                step = nearest_reachable_crystal_step(col, row, max_depth=8)
                if not step and factory_data:
                    step = step_toward_target(
                        col,
                        row,
                        factory_data[1],
                        factory_data[2],
                        allow_north_fallback=False,
                    )
                if step and can_move(col, row, step):
                    actions[uid] = step
                    reserve(start, step)
                    continue
                actions[uid] = "IDLE"
                continue

            if move_cd <= 1:
                step = step_toward_worker_target(col, row)
                if step and can_move(col, row, step):
                    actions[uid] = step
                    reserve(start, step)
                    continue
            actions[uid] = "IDLE"
            continue

        actions[uid] = "IDLE"

    for uid, data in my.items():
        history = positions_memory.setdefault(uid, [])
        history.append((data[1], data[2]))
        if len(history) > 12:
            del history[:-12]
        previous_max = memory["max_rows"].get(uid, data[2])
        if data[2] > previous_max:
            memory["progress_waits"][uid] = 0
        elif data[0] == FACTORY:
            memory["progress_waits"][uid] = memory["progress_waits"].get(uid, 0) + 1
        else:
            memory["progress_waits"][uid] = 0
        memory["max_rows"][uid] = max(previous_max, data[2])
        action = actions.get(uid, "IDLE")
        memory["actions"][uid] = action
        if action == "IDLE":
            memory["idle_counts"][uid] = memory["idle_counts"].get(uid, 0) + 1
        else:
            memory["idle_counts"][uid] = 0
    if step_no is not None:
        memory["last_step"] = step_no
    else:
        memory["last_step"] = memory.get("last_step", -1) + 1

    return actions
