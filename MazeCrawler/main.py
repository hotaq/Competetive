"""Maze Crawler v1b: safe northward starter with collision-aware movement."""

NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
DEFAULT_HEIGHT = 20
DEFAULT_WIDTH = 20

DIRECTIONS = {
    "NORTH": (0, 1, NORTH),
    "EAST": (1, 0, EAST),
    "SOUTH": (0, -1, SOUTH),
    "WEST": (-1, 0, WEST),
}

TYPE_FACTORY = 0
TYPE_SCOUT = 1
TYPE_WORKER = 2
TYPE_MINER = 3


def agent(obs, config):
    config = ConfigView(config)
    obs = ObsView(obs, config)
    width = config.width
    actions = {}

    my_robots = {
        uid: data for uid, data in obs.robots.items()
        if data[4] == obs.player
    }
    visible_robots = {
        (data[1], data[2]): data
        for data in obs.robots.values()
    }
    blocked = set(visible_robots)
    reserved = set()

    # Move northern robots first so trailing robots can safely follow vacated cells.
    turn_order = sorted(
        my_robots.items(),
        key=lambda item: (-item[1][2], item[1][1], str(item[0])),
    )

    for uid, data in turn_order:
        rtype, col, row, energy = data[0], data[1], data[2], data[3]
        move_cd = data[5] if len(data) > 5 else 0
        jump_cd = data[6] if len(data) > 6 else 0
        build_cd = data[7] if len(data) > 7 else 0
        current = (col, row)
        wall = wall_at(obs, width, col, row)

        if rtype == TYPE_FACTORY:
            action = factory_action(
                obs, config, my_robots, blocked, reserved,
                col, row, energy, wall, move_cd, jump_cd, build_cd,
            )
        elif rtype == TYPE_WORKER and should_remove_north(obs, config, width, col, row, energy, wall):
            action = "REMOVE_NORTH"
        elif rtype == TYPE_MINER and can_transform(obs, config, col, row, energy):
            action = "TRANSFORM"
        elif move_cd == 0:
            action = best_move(obs, width, uid, rtype, col, row, wall, blocked, reserved)
        else:
            action = "IDLE"

        actions[uid] = action
        update_blockers(action, current, blocked, reserved)

    return actions


def factory_action(obs, config, my_robots, blocked, reserved, col, row, energy, wall, move_cd, jump_cd, build_cd):
    worker_count = count_type(my_robots, TYPE_WORKER)
    scout_count = count_type(my_robots, TYPE_SCOUT)

    if build_cd == 0 and row > obs.southBound + 2 and can_spawn(obs, config.width, col, row, wall, blocked):
        if worker_count == 0 and energy >= config.workerCost:
            return "BUILD_WORKER"
        if scout_count < 2 and energy >= config.scoutCost:
            return "BUILD_SCOUT"
        if worker_count < 2 and energy >= config.workerCost and energy > getattr(config, "factoryEnergy", 1000) // 2:
            return "BUILD_WORKER"

    if jump_cd == 0 and (wall & NORTH or row <= obs.southBound + 2):
        jump_row = row + 2
        if safe_jump_target(obs, config.width, col, jump_row, blocked, reserved):
            return "JUMP_NORTH"

    if move_cd == 0:
        move = best_move(obs, config.width, "factory", TYPE_FACTORY, col, row, wall, blocked, reserved)
        if move != "IDLE":
            return move

    return "IDLE"


def best_move(obs, width, uid, rtype, col, row, wall, blocked, reserved):
    options = []
    for direction in direction_order(obs.player, uid, col, width):
        target = step(col, row, direction)
        if not can_move_to(obs, width, col, row, direction, wall, blocked, reserved):
            continue
        options.append((move_score(obs, width, rtype, direction, target), direction, target))

    if not options:
        return "IDLE"

    options.sort(reverse=True)
    return options[0][1]


def move_score(obs, width, rtype, direction, target):
    col, row = target
    score = 0

    if direction == "NORTH":
        score += 100
    elif direction in ("EAST", "WEST"):
        score += 20
    else:
        score -= 30

    if rtype == TYPE_SCOUT and not cell_known(obs, width, col, row):
        score += 35

    crystal_energy = crystal_at(obs, col, row)
    if crystal_energy:
        score += 60 + crystal_energy

    if rtype == TYPE_MINER and mining_node_at(obs, col, row):
        score += 120

    if row <= obs.southBound + 1:
        score -= 100

    return score


def direction_order(player, uid, col, width):
    center_first = ["EAST", "WEST"] if col < width // 2 else ["WEST", "EAST"]
    edge_first = list(reversed(center_first))
    if stable_bit(uid) ^ player:
        lateral = edge_first
    else:
        lateral = center_first
    return ["NORTH", lateral[0], lateral[1], "SOUTH"]


def can_spawn(obs, width, col, row, wall, blocked):
    target = (col, row + 1)
    return (
        not (wall & NORTH)
        and in_bounds(obs, width, target[0], target[1])
        and target not in blocked
    )


def can_move_to(obs, width, col, row, direction, wall, blocked, reserved):
    target = step(col, row, direction)
    return (
        not (wall & DIRECTIONS[direction][2])
        and in_bounds(obs, width, target[0], target[1])
        and target not in blocked
        and target not in reserved
    )


def should_remove_north(obs, config, width, col, row, energy, wall):
    target_row = row + 1
    return (
        energy >= config.wallRemoveCost
        and (wall & NORTH)
        and in_bounds(obs, width, col, target_row)
    )


def can_transform(obs, config, col, row, energy):
    return energy > config.transformCost and mining_node_at(obs, col, row)


def safe_jump_target(obs, width, col, row, blocked, reserved):
    target = (col, row)
    return (
        in_bounds(obs, width, col, row)
        and target not in blocked
        and target not in reserved
    )


def update_blockers(action, current, blocked, reserved):
    if action in DIRECTIONS:
        target = (current[0] + DIRECTIONS[action][0], current[1] + DIRECTIONS[action][1])
        blocked.discard(current)
        blocked.add(target)
        reserved.add(target)
    elif action.startswith("JUMP_"):
        direction = action[5:]
        if direction in DIRECTIONS:
            target = (
                current[0] + DIRECTIONS[direction][0] * 2,
                current[1] + DIRECTIONS[direction][1] * 2,
            )
            blocked.discard(current)
            blocked.add(target)
            reserved.add(target)


def wall_at(obs, width, col, row):
    idx = (row - obs.southBound) * width + col
    if 0 <= idx < len(obs.walls) and obs.walls[idx] != -1:
        return obs.walls[idx]
    return 0


def cell_known(obs, width, col, row):
    idx = (row - obs.southBound) * width + col
    return 0 <= idx < len(obs.walls) and obs.walls[idx] != -1


def crystal_at(obs, col, row):
    return obs.crystals.get(f"{col},{row}", 0)


def mining_node_at(obs, col, row):
    return f"{col},{row}" in obs.miningNodes


def in_bounds(obs, width, col, row):
    return 0 <= col < width and obs.southBound <= row <= obs.northBound


def step(col, row, direction):
    dx, dy, _ = DIRECTIONS[direction]
    return col + dx, row + dy


def count_type(robots, rtype):
    return sum(1 for data in robots.values() if data[0] == rtype)


def stable_bit(value):
    total = 0
    for char in str(value):
        total ^= ord(char)
    return total & 1


class ConfigView:
    def __init__(self, config):
        self.width = read_field(config, "width", DEFAULT_WIDTH)
        self.height = read_field(config, "height", DEFAULT_HEIGHT)
        self.scoutCost = read_field(config, "scoutCost", 50)
        self.workerCost = read_field(config, "workerCost", 200)
        self.minerCost = read_field(config, "minerCost", 300)
        self.wallRemoveCost = read_field(config, "wallRemoveCost", 100)
        self.transformCost = read_field(config, "transformCost", 100)
        self.factoryEnergy = read_field(config, "factoryEnergy", 1000)


class ObsView:
    def __init__(self, obs, config):
        self.player = read_field(obs, "player", 0)
        self.walls = read_field(obs, "walls", []) or []
        self.crystals = as_mapping(read_field(obs, "crystals", {}))
        self.robots = as_mapping(read_field(obs, "robots", {}))
        self.mines = as_mapping(read_field(obs, "mines", {}))
        self.miningNodes = as_mapping(read_field(obs, "miningNodes", {}))
        self.southBound = read_field(obs, "southBound", None)
        if self.southBound is None:
            self.southBound = infer_south_bound(self.robots, config.height)
        self.northBound = read_field(obs, "northBound", self.southBound + config.height - 1)


def read_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def as_mapping(value):
    if value is None:
        return {}
    if hasattr(value, "items"):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def infer_south_bound(robots, height):
    rows = [
        data[2] for data in robots.values()
        if isinstance(data, (list, tuple)) and len(data) > 2
    ]
    if not rows:
        return 0
    return max(0, min(rows) - 2)
