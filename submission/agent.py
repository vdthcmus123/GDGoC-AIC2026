from pathlib import Path
from collections import deque
import numpy as np
import torch

MOVES = {0:(0,0),1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
HORIZON = 7
ADVERSARIAL_BOMB_DISTANCE = 4
MIN_BOMB_EXITS = 2
HIDDEN_SIZE = 128
N_CHANNELS = 19
N_SCALARS = 8
PLANNER_DEPTH = 6
PLANNER_WIDTH = 12

def inside(grid, r, c):
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

def passable(grid, r, c):
    return inside(grid, r, c) and int(grid[r, c]) in (0, 3, 4)

def blast(grid, r, c, radius):
    tiles = {(int(r), int(c))}
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        for d in range(1, int(radius) + 1):
            nr, nc = int(r) + dr*d, int(c) + dc*d
            if not inside(grid, nr, nc):
                break
            cell = int(grid[nr, nc])
            if cell == 1:
                break
            tiles.add((nr, nc))
            if cell == 2:
                break
    return tiles

class BombTracker:
    def __init__(self):
        self.radius = {}
    def update(self, obs):
        current = set()
        players = obs["players"]
        for b in obs["bombs"]:
            r, c, _, owner = [int(v) for v in b]
            key = (r, c, owner)
            current.add(key)
            if key not in self.radius:
                self.radius[key] = 1 + int(players[owner][4])
        self.radius = {key: value for key, value in self.radius.items() if key in current}
        return self.radius

def bombs(obs, radius_lookup, extra=None, extras=None):
    players = obs["players"]
    result = []
    for b in obs["bombs"]:
        r, c, timer, owner = [int(v) for v in b]
        fallback = 1 + int(players[owner][4])
        result.append((r, c, timer, owner, int(radius_lookup.get((r,c,owner), fallback))))
    if extra is not None:
        result.append(tuple(int(v) for v in extra))
    for b in extras or ():
        result.append(tuple(int(v) for v in b))
    return result

def danger(obs, horizon=HORIZON, radius_lookup=None, extra=None, extras=None):
    grid = obs["map"]
    entries = bombs(obs, radius_lookup or {}, extra, extras)
    planes = np.zeros((horizon, 13, 13), dtype=np.float32)
    if not entries:
        return planes
    timers = [max(1, int(entry[2])) for entry in entries]
    blasts = [blast(grid, entry[0], entry[1], entry[4]) for entry in entries]
    for _ in range(len(entries)):
        changed = False
        for i, source in enumerate(blasts):
            for j, target in enumerate(entries):
                if i != j and (target[0], target[1]) in source and timers[j] > timers[i]:
                    timers[j] = timers[i]
                    changed = True
        if not changed:
            break
    for timer, tiles in zip(timers, blasts):
        if 1 <= timer <= horizon:
            for r, c in tiles:
                planes[timer - 1, r, c] = 1.0
    return planes

def enemy_bomb_threats(obs, player_id, max_distance=ADVERSARIAL_BOMB_DISTANCE):
    players = obs["players"]
    own_row, own_col = int(players[player_id][0]), int(players[player_id][1])
    occupied = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
    result = []
    for enemy_id, enemy in enumerate(players):
        if enemy_id == player_id or int(enemy[2]) != 1 or int(enemy[3]) <= 0:
            continue
        row, col = int(enemy[0]), int(enemy[1])
        if (row, col) in occupied:
            continue
        if abs(row - own_row) + abs(col - own_col) <= max_distance:
            result.append((row, col, 6, enemy_id, 1 + int(enemy[4])))
    return result

def survival_depth(obs, player_id, first_action, radius_lookup, stress=None):
    grid = obs["map"]
    row, col, alive, _, bonus = [int(v) for v in obs["players"][player_id]]
    if not alive:
        return HORIZON if int(first_action) == 0 else 0
    blocked = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
    extra = None
    if int(first_action) == 5:
        blocked.add((row, col))
        extra = (row, col, 6, player_id, 1 + bonus)
        nr, nc = row, col
    else:
        dr, dc = MOVES[int(first_action)]
        nr, nc = row + dr, col + dc
        if int(first_action) != 0 and (not passable(grid, nr, nc) or (nr, nc) in blocked):
            return 0
    extras = []
    if extra is not None:
        extras.append(extra)
    if stress is not None:
        blocked.add((int(stress[0]), int(stress[1])))
        extras.append(stress)
    future = danger(obs, HORIZON, radius_lookup, extras=extras)
    if future[0, nr, nc] > 0.5:
        return 0
    positions = {(nr, nc)}
    for tick in range(1, HORIZON):
        following = set()
        for r, c in positions:
            for action in range(5):
                dr, dc = MOVES[action]
                nr, nc = r + dr, c + dc
                if action != 0 and (not passable(grid, nr, nc) or (nr, nc) in blocked):
                    continue
                if future[tick, nr, nc] < 0.5:
                    following.add((nr, nc))
        positions = following
        if not positions:
            return tick
    return HORIZON

def robust_survival_depth(obs, player_id, first_action, radius_lookup):
    depth = survival_depth(obs, player_id, first_action, radius_lookup)
    if depth < HORIZON:
        return depth
    for stress in enemy_bomb_threats(obs, player_id):
        depth = min(depth, survival_depth(obs, player_id, first_action, radius_lookup, stress))
    return depth

def bomb_exit_count(obs, player_id, radius_lookup):
    grid = obs["map"]
    row, col, alive, _, bonus = [int(v) for v in obs["players"][player_id]]
    if not alive:
        return 0
    blocked = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
    blocked.add((row, col))
    future = danger(obs, HORIZON, radius_lookup, (row, col, 6, player_id, 1 + bonus))
    exits = 0
    for action in range(1, 5):
        dr, dc = MOVES[action]
        nr, nc = row + dr, col + dc
        if passable(grid, nr, nc) and (nr, nc) not in blocked and future[0, nr, nc] < 0.5:
            exits += 1
    return exits

def mask_actions(obs, player_id, radius_lookup):
    grid = obs["map"]
    row, col, alive, left, _ = [int(v) for v in obs["players"][player_id]]
    result = np.zeros(6, dtype=np.bool_)
    if not alive:
        result[0] = True
        return result
    blocked = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
    result[0] = True
    for action in range(1, 5):
        dr, dc = MOVES[action]
        nr, nc = row + dr, col + dc
        result[action] = passable(grid, nr, nc) and (nr, nc) not in blocked
    result[5] = left > 0 and (row, col) not in blocked
    depths = np.full(6, -1, dtype=np.int8)
    for action in np.flatnonzero(result):
        depths[action] = survival_depth(obs, player_id, int(action), radius_lookup)
    safe = result & (depths >= HORIZON)
    if safe[5]:
        if bomb_exit_count(obs, player_id, radius_lookup) < MIN_BOMB_EXITS:
            safe[5] = False
            depths[5] = -1
    if safe.any() and enemy_bomb_threats(obs, player_id):
        robust = np.full(6, -1, dtype=np.int8)
        for action in np.flatnonzero(safe):
            robust[action] = robust_survival_depth(obs, player_id, int(action), radius_lookup)
        robust_safe = safe & (robust >= HORIZON)
        if robust_safe.any():
            safe = robust_safe
    if safe.any():
        return safe
    non_bomb = result.copy()
    non_bomb[5] = False
    if non_bomb.any():
        return non_bomb & (depths == depths[non_bomb].max())
    result[5] = False
    if result.any():
        return result
    result[0] = True
    return result

def encode(obs, player_id, step_count, radius_lookup):
    grid, players = obs["map"], obs["players"]
    channels = [(grid == tile).astype(np.float32) for tile in range(5)]
    own = np.zeros((13, 13), dtype=np.float32)
    if int(players[player_id][2]) == 1:
        own[int(players[player_id][0]), int(players[player_id][1])] = 1.0
    channels.append(own)
    enemies = [index for index in range(4) if index != player_id]
    for enemy_id in enemies:
        plane = np.zeros((13, 13), dtype=np.float32)
        if int(players[enemy_id][2]) == 1:
            plane[int(players[enemy_id][0]), int(players[enemy_id][1])] = 1.0
        channels.append(plane)
    bomb_plane = np.zeros((13, 13), dtype=np.float32)
    timer_plane = np.zeros((13, 13), dtype=np.float32)
    radius_plane = np.zeros((13, 13), dtype=np.float32)
    for r, c, timer, _, radius in bombs(obs, radius_lookup):
        bomb_plane[r, c] = 1.0
        timer_plane[r, c] = max(timer_plane[r, c], float(timer) / 7.0)
        radius_plane[r, c] = max(radius_plane[r, c], float(radius) / 5.0)
    channels.extend((bomb_plane, timer_plane, radius_plane))
    future = danger(obs, HORIZON, radius_lookup)
    channels.extend(future)
    own_row, own_col = int(players[player_id][0]), int(players[player_id][1])
    enemy_distances = [
        abs(int(players[index][0]) - own_row) + abs(int(players[index][1]) - own_col)
        for index in enemies if int(players[index][2]) == 1
    ]
    nearest_enemy = min(enemy_distances) if enemy_distances else 24
    blocked = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
    safe_moves = 0
    for action in range(5):
        dr, dc = MOVES[action]
        nr, nc = own_row + dr, own_col + dc
        if action != 0 and (not passable(grid, nr, nc) or (nr, nc) in blocked):
            continue
        if future[0, nr, nc] < 0.5:
            safe_moves += 1
    boxes_in_range = sum(
        int(grid[r, c]) == 2 for r, c in blast(grid, own_row, own_col, 1 + int(players[player_id][4]))
    )
    scalar = np.asarray([
        float(players[player_id][3]) / 5.0,
        float(players[player_id][4]) / 4.0,
        float(sum(int(players[index][2]) for index in enemies)) / 3.0,
        float(player_id) / 3.0,
        min(float(step_count), 500.0) / 500.0,
        min(float(nearest_enemy), 24.0) / 24.0,
        float(safe_moves) / 5.0,
        min(float(boxes_in_range), 4.0) / 4.0,
    ], dtype=np.float32)
    return np.stack(channels).astype(np.float32), scalar

def objectives(grid):
    result = {}
    for r, c in np.argwhere((grid == 3) | (grid == 4)):
        result[(int(r), int(c))] = 1.25
    for br, bc in np.argwhere(grid == 2):
        for dr, dc in MOVES.values():
            r, c = int(br) + dr, int(bc) + dc
            if passable(grid, r, c):
                result[(r, c)] = max(result.get((r, c), 0.0), 0.60)
    return tuple((r, c, value) for (r, c), value in result.items())

def escape_region_size(grid, start_row, start_col, blocked, future, ticks=6):
    positions = {(int(start_row), int(start_col))}
    for tick in range(min(int(ticks), future.shape[0])):
        next_positions = set()
        for row, col in positions:
            for action in range(5):
                drow, dcol = MOVES[action]
                next_row, next_col = row + drow, col + dcol
                if action != 0 and (
                    not passable(grid, next_row, next_col)
                    or (next_row, next_col) in blocked
                ):
                    continue
                if future[tick, next_row, next_col] < 0.5:
                    next_positions.add((next_row, next_col))
        positions = next_positions
        if not positions:
            return 0
    return len(positions)

def plan_action(obs, player_id, radius_lookup, policy_logits, recent_positions, action_mask=None, step_count=0):
    grid, players = obs["map"], obs["players"]
    row, col, alive, _, bonus = [int(v) for v in players[player_id]]
    if not alive:
        return 0
    mask = mask_actions(obs, player_id, radius_lookup) if action_mask is None else action_mask
    logits = np.asarray(policy_logits, dtype=np.float32)
    priors = np.zeros(6, dtype=np.float32)
    priors[mask] = np.exp(logits[mask] - np.max(logits[mask]))
    priors /= max(float(priors.sum()), 1e-8)
    bomb_positions = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
    enemy_records = [
        (
            index,
            int(players[index][0]),
            int(players[index][1]),
            int(players[index][3]),
            int(players[index][4]),
        )
        for index in range(len(players)) if index != player_id and int(players[index][2]) == 1
    ]
    enemies = [(er, ec) for _, er, ec, _, _ in enemy_records]
    targets = objectives(grid)
    recent_sequence = tuple(recent_positions)
    recent_positions = set(recent_sequence)
    current_future = danger(obs, HORIZON, radius_lookup)
    threatened = bool(np.any(current_future[:, row, col] > 0.5))
    boxes_left = int(np.count_nonzero(grid == 2))
    opening = int(step_count) <= 80
    endgame = int(step_count) >= 260 or len(enemy_records) <= 1 or boxes_left <= 4
    closing_phase = int(step_count) >= 400
    aggression = 1.0 + 0.45 * endgame + 0.35 * closing_phase
    own_recent_bombs = [
        (int(b[0]), int(b[1]), int(b[2])) for b in obs['bombs']
        if int(b[3]) == player_id and int(b[2]) >= 4
    ]
    baseline_regions = {
        enemy_id: escape_region_size(grid, er, ec, bomb_positions, current_future)
        for enemy_id, er, ec, _, _ in enemy_records
    }

    def adversarial_exposure(r, c):
        return sum(
            bombs_left > 0 and (r, c) in blast(grid, er, ec, 1 + radius)
            for _, er, ec, bombs_left, radius in enemy_records
        )

    def position_score(r, c, future, blocked, tick):
        mobility = 0
        for action in range(5):
            dr, dc = MOVES[action]
            nr, nc = r + dr, c + dc
            if action != 0 and (not passable(grid, nr, nc) or (nr, nc) in blocked):
                continue
            if future[min(tick, HORIZON - 1), nr, nc] < 0.5:
                mobility += 1
        margin = sum(future[index, r, c] < 0.5 for index in range(tick, HORIZON))
        distance = min((abs(er - r) + abs(ec - c) for er, ec in enemies), default=24)
        item = 0.35 if int(grid[r, c]) in (3, 4) else 0.0
        objective = max(
            (value - 0.04 * (abs(target_row - r) + abs(target_col - c))
             for target_row, target_col, value in targets),
            default=0.0,
        )
        corridor = 0.12 if mobility <= 1 else 0.0
        pressure = 0.12 * adversarial_exposure(r, c)
        objective_scale = 1.35 if opening else (0.72 if endgame else 1.0)
        distance_weight = 0.018 if endgame else 0.01
        return (
            0.08 * mobility
            + 0.04 * margin
            + item
            + objective_scale * objective
            - distance_weight * distance
            - corridor
            - pressure
        )

    beams = []
    for first in np.flatnonzero(mask):
        first = int(first)
        dr, dc = MOVES.get(first, (0, 0))
        nr, nc = row + dr, col + dc
        blocked, extra, tactical = set(bomb_positions), None, 0.0
        if first == 0 and np.any(mask[1:5]):
            near_own_bomb = any(abs(row - br) + abs(col - bc) <= 2 for br, bc, _ in own_recent_bombs)
            if near_own_bomb:
                tactical -= 1.25
            elif threatened or len(obs['bombs']):
                tactical += 0.05
            else:
                tactical -= 0.60 if opening else (0.50 if endgame else 0.30)
        if first != 5 and (nr, nc) in recent_positions:
            tactical -= 0.12
        if first != 5 and len(recent_sequence) >= 2 and (nr, nc) == recent_sequence[-2]:
            tactical -= 0.30
        if first in (1, 2, 3, 4) and own_recent_bombs:
            br, bc, _ = min(own_recent_bombs, key=lambda b: abs(row - b[0]) + abs(col - b[1]))
            before_d = abs(row - br) + abs(col - bc)
            after_d = abs(nr - br) + abs(nc - bc)
            if after_d > before_d:
                tactical += 0.70
            elif after_d < before_d:
                tactical -= 0.55
            if row == br or col == bc:
                tactical += 0.25 if (nr != br and nc != bc) else -0.20
        if first == 5:
            nr, nc = row, col
            blocked.add((row, col))
            extra = (row, col, 6, player_id, 1 + bonus)
            tiles = blast(grid, row, col, 1 + bonus)
            box_hits = sum(int(grid[r, c]) == 2 for r, c in tiles)
            enemy_hits = sum((er, ec) in tiles for er, ec in enemies)
            pressure = 0.0
            for _, er, ec, _, _ in enemy_records:
                escapes = sum(
                    passable(grid, er + dr, ec + dc) and (er + dr, ec + dc) not in tiles
                    for dr, dc in MOVES.values()
                )
                covered = sum((er + dr, ec + dc) in tiles for dr, dc in MOVES.values())
                if (er, ec) in tiles:
                    pressure += 0.45 + 0.22 * max(0, 2 - escapes)
                elif abs(er - row) + abs(ec - col) <= 4:
                    pressure += 0.06 * covered
            tactical += (1.65 if opening else 1.00) * box_hits + aggression * (1.50 * enemy_hits + 1.18 * pressure)
            exits = bomb_exit_count(obs, player_id, radius_lookup)
            if exits < MIN_BOMB_EXITS:
                tactical -= 4.0
            elif exits == MIN_BOMB_EXITS and len(obs["bombs"]) >= 2:
                tactical -= 0.30
            if box_hits == 0 and enemy_hits == 0:
                tactical -= 1.05 if opening else 0.58
            elif opening and box_hits > 0:
                tactical += 0.55
        future = danger(obs, HORIZON, radius_lookup, extra)
        if first == 5:
            trap_gain = 0
            forced_traps = 0
            for enemy_id, er, ec, _, _ in enemy_records:
                remaining = escape_region_size(grid, er, ec, blocked, future)
                trap_gain += max(0, baseline_regions.get(enemy_id, remaining) - remaining)
                forced_traps += int(remaining == 0)
            tactical += aggression * (0.11 * min(trap_gain, 14) + 1.10 * forced_traps)
        score = 1.20 * float(priors[first]) + tactical + position_score(nr, nc, future, blocked, 0)
        beams.append((score, nr, nc, first, blocked, future))

    for tick in range(1, PLANNER_DEPTH):
        expanded = []
        for score, r, c, first, blocked, future in beams:
            for action in range(5):
                dr, dc = MOVES[action]
                nr, nc = r + dr, c + dc
                if action != 0 and (not passable(grid, nr, nc) or (nr, nc) in blocked):
                    continue
                if future[min(tick, HORIZON - 1), nr, nc] > 0.5:
                    continue
                expanded.append((score + position_score(nr, nc, future, blocked, tick), nr, nc, first, blocked, future))
        if not expanded:
            break
        beams = sorted(expanded, key=lambda item: item[0], reverse=True)[:PLANNER_WIDTH]
    if beams:
        return int(max(beams, key=lambda item: item[0])[3])
    return int(np.flatnonzero(mask)[np.argmax(priors[mask])])

class Agent:
    def __init__(self, agent_id: int):
        torch.set_num_threads(1)
        self.agent_id = int(agent_id)
        self.step_count = 0
        self.tracker = BombTracker()
        self.model = torch.jit.load(str(Path(__file__).parent / "policy.pt"), map_location="cpu")
        self.model.eval()
        self.hidden = torch.zeros(1, HIDDEN_SIZE, dtype=torch.float32)
        self.recent_positions = deque(maxlen=6)
        # Move TorchScript cold-start work into the 20-second startup
        # budget. The evaluator gives the first act() call only 100ms.
        with torch.no_grad():
            warm_maps = torch.zeros(1, N_CHANNELS, 13, 13, dtype=torch.float32)
            warm_scalars = torch.zeros(1, N_SCALARS, dtype=torch.float32)
            warm_masks = torch.ones(1, 6, dtype=torch.bool)
            warm_hidden = torch.zeros(1, HIDDEN_SIZE, dtype=torch.float32)
            self.model(warm_maps, warm_scalars, warm_masks, warm_hidden)

    def act(self, obs: dict) -> int:
        try:
            self.step_count += 1
            radius_lookup = self.tracker.update(obs)
            features, scalars = encode(obs, self.agent_id, self.step_count, radius_lookup)
            masks = mask_actions(obs, self.agent_id, radius_lookup)
            with torch.no_grad():
                logits, _, self.hidden = self.model(
                    torch.from_numpy(features).unsqueeze(0),
                    torch.from_numpy(scalars).unsqueeze(0),
                    torch.from_numpy(masks).unsqueeze(0),
                    self.hidden,
                )
            action = plan_action(
                obs,
                self.agent_id,
                radius_lookup,
                logits[0].numpy(),
                self.recent_positions,
                masks,
                step_count=self.step_count,
            )
            self.recent_positions.append(
                (int(obs["players"][self.agent_id][0]), int(obs["players"][self.agent_id][1]))
            )
            return action
        except Exception:
            return 0
