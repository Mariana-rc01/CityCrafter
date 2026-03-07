import time
from coordinates import Coordinates

def greedy2(city):
    """
    Constructive Heuristic O(1) extracted from Tabu Search.
    Builds the city through a pattern of blocks in 3x3 jumps.
    """
    t0 = time.time()
    H, W, D = city.H, city.W, city.D

    print("========== GUIDED BLOCKS greedy2 ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")

    # --- 1. STATE PREPARATION O(1) ---
    occupied = [[False] * W for _ in range(H)]
    influence_grid = [[{} for _ in range(W)] for _ in range(H)]
    residential_at = [[None] * W for _ in range(H)]

    res_coverage = {}
    res_capacity = {} # Stores the capacity of each placed house

    final_placements = []
    current_score = 0
    next_uid = 0
    cell_cache = {}

    # --- 2. PIECE SELECTION ---
    residential = [p for p in city.projects if p.build_type == "R"]
    utilities = [p for p in city.projects if p.build_type == "U"]

    # Sort houses by best ratio (Capacity / Size)
    residential_sorted = sorted(residential, key=lambda p: p.capacity / max(1, p.h * p.w), reverse=True)
    top_res = residential_sorted[:40]

    # Choose 1 service (utility) of each type
    best_utils = []
    seen = set()
    for u in utilities:
        if u.service_type not in seen:
            seen.add(u.service_type)
            best_utils.append(u.project_id)

    # --- 3. HELPER FUNCTIONS ---
    def get_cells(b, r, c):
        key = (b, r, c)
        if key not in cell_cache:
            cell_cache[key] = city.get_project(b).absolute_hash_cells(Coordinates(r, c))
        return cell_cache[key]

    def get_influence_diamond(r, c, dist):
        for dr in range(-dist, dist + 1):
            rem = dist - abs(dr)
            for dc in range(-rem, rem + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    yield nr, nc

    def place(b, r, c):
        nonlocal current_score, next_uid
        proj = city.get_project(b)

        # Check city boundaries
        if r < 0 or r + proj.h > H or c < 0 or c + proj.w > W:
            return False

        cells = get_cells(b, r, c)

        # Check collisions
        for cell in cells:
            if occupied[cell.r][cell.c]:
                return False

        uid = next_uid
        next_uid += 1
        gain = 0

        # Calculate instant score and fill grid
        if proj.build_type == "R":
            cap = proj.capacity
            res_capacity[uid] = cap
            cov = {}
            for cell in cells:
                occupied[cell.r][cell.c] = True
                residential_at[cell.r][cell.c] = uid
                for s_type, count in influence_grid[cell.r][cell.c].items():
                    cov[s_type] = cov.get(s_type, 0) + count

            for s_type, count in cov.items():
                if count > 0:
                    gain += cap
            res_coverage[uid] = cov

        else: # Build Type == "U"
            s_type = proj.service_type
            affected = set()
            for cell in cells:
                occupied[cell.r][cell.c] = True
                for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                    affected.add((nr, nc))

            for nr, nc in affected:
                influence_grid[nr][nc][s_type] = influence_grid[nr][nc].get(s_type, 0) + 1
                ruid = residential_at[nr][nc]
                if ruid is not None:
                    cov = res_coverage[ruid]
                    cov[s_type] = cov.get(s_type, 0) + 1
                    if cov[s_type] == 1:
                        gain += res_capacity[ruid]

        current_score += gain
        final_placements.append((b, r, c))
        return True

    # --- 4. MAIN CONSTRUCTION ENGINE (3x3 JUMPS) ---
    placed_res = 0
    placed_util = 0

    print("Start building guided blocks...")
    for r in range(0, H, 3):
        if r % 60 == 0:
            print(f"  Scan row {r}/{H} | placements: {len(final_placements)} | R={placed_res} U={placed_util}")

        for c in range(0, W, 3):
            # 1st Step: Try to place all possible service types at root (r, c)
            for uid in best_utils:
                if place(uid, r, c):
                    placed_util += 1

            # 2nd Step: Try to surround with the best houses on the diagonal (r+1, c+1)
            for res in top_res:
                if place(res.project_id, r + 1, c + 1):
                    placed_res += 1

    dt = time.time() - t0
    print(f"Guided Blocks greedy2 finished: placements={len(final_placements)} | score={current_score} | time={dt:.2f}s")
    print("======== END GUIDED BLOCKS greedy2 ========")

    return final_placements
