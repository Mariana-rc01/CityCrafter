import random
import time
import math
from coordinates import Coordinates
from algorithms.greedy import greedy

#def simulated_annealing(city, max_iterations=22000, initial_temperature=4200.0, cooling_rate=0.9982, min_temperature=0.08, reheats=3, reheat_factor=0.40, neighborhood_radius=5, top_k_res=45, max_utility_types=15, add_weight=0.16, remove_weight=0.10, move_weight=0.48, change_weight=0.26, max_runtime_s=720, seed=2, use_greedy_init=True, same_type_change_prob=0.92, similar_shape_change_prob=0.85, shape_slack=2):
#def simulated_annealing(city, max_iterations=12000, initial_temperature=2500.0, cooling_rate=0.9975, min_temperature=0.15, reheats=1, reheat_factor=0.35, neighborhood_radius=4, top_k_res=35, max_utility_types=12, add_weight=0.18, remove_weight=0.10, move_weight=0.40, change_weight=0.32, max_runtime_s=540, seed=0, use_greedy_init=True, same_type_change_prob=0.82, similar_shape_change_prob=0.70, shape_slack=2):
def simulated_annealing(city, max_iterations=12000, initial_temperature=1200.0, cooling_rate=0.997, min_temperature=0.2, reheats=1, reheat_factor=0.25, neighborhood_radius=3, top_k_res=35, max_utility_types=12, add_weight=0.15, remove_weight=0.12, move_weight=0.45, change_weight=0.28, max_runtime_s=540, seed=1, use_greedy_init=True, same_type_change_prob=0.88, similar_shape_change_prob=0.75, shape_slack=2,):
    """
    Simulated Annealing com:
    - score incremental
    - warm start por greedy
    - operador CHANGE inteligente

    Retorna:
        list[(project_id, row, col)]
    """

    random.seed(seed)
    t0 = time.time()

    H, W, D = city.H, city.W, city.D

    # =========================================================
    # STATE
    # =========================================================
    occupied = [[False] * W for _ in range(H)]
    influence_grid = [[{} for _ in range(W)] for _ in range(H)]
    residential_at = [[None] * W for _ in range(H)]

    placements = {}      # uid -> [b_id, r, c, b_type, val, cells]
    res_coverage = {}    # uid -> {service_type: count}
    active_uids = []
    uid_to_idx = {}
    next_uid = 0

    current_score = 0
    best_score = 0
    best_solution = []

    cell_cache = {}

    residential = [p for p in city.projects if p.build_type == "R"]
    utilities = [p for p in city.projects if p.build_type == "U"]

    residential_sorted = sorted(
        residential,
        key=lambda p: p.capacity / max(1, len(p.hash_offsets)),
        reverse=True,
    )
    top_res = residential_sorted[:top_k_res]

    utility_by_type = {}
    for u in utilities:
        s = u.service_type
        area = len(u.hash_offsets)
        if s not in utility_by_type or area < len(utility_by_type[s].hash_offsets):
            utility_by_type[s] = u

    selected_utils = list(utility_by_type.values())
    selected_utils.sort(key=lambda u: len(u.hash_offsets))
    selected_utils = selected_utils[:max_utility_types]

    # pools auxiliares para CHANGE inteligente
    residential_by_shape = {}
    utility_by_shape = {}

    for p in residential:
        key = (p.h, p.w)
        residential_by_shape.setdefault(key, []).append(p)
    for p in utilities:
        key = (p.h, p.w)
        utility_by_shape.setdefault(key, []).append(p)

    print("========== SIMULATED ANNEALING ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")
    print(f"Top residential candidates: {len(top_res)} (top_k_res={top_k_res})")
    print(f"Selected utility types: {len(selected_utils)} (max_utility_types={max_utility_types})")
    print(f"Warm start greedy: {use_greedy_init}")
    print("Start building...")

    # =========================================================
    # HELPERS
    # =========================================================
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

    def add_active_uid(uid):
        uid_to_idx[uid] = len(active_uids)
        active_uids.append(uid)

    def remove_active_uid(uid):
        idx = uid_to_idx.pop(uid)
        last_uid = active_uids.pop()
        if idx < len(active_uids):
            active_uids[idx] = last_uid
            uid_to_idx[last_uid] = idx

    def can_place(proj, r, c):
        if r < 0 or c < 0 or r + proj.h > H or c + proj.w > W:
            return False
        cells = get_cells(proj.project_id, r, c)
        for cell in cells:
            if occupied[cell.r][cell.c]:
                return False
        return cells

    def place_residential(uid, capacity, cells):
        gain = 0
        cov = {}

        for cell in cells:
            occupied[cell.r][cell.c] = True
            residential_at[cell.r][cell.c] = uid

            for s_type, count in influence_grid[cell.r][cell.c].items():
                cov[s_type] = cov.get(s_type, 0) + count

        for s_type, count in cov.items():
            if count > 0:
                gain += capacity

        res_coverage[uid] = cov
        return gain

    def remove_residential(uid, capacity, cells):
        loss = 0
        cov = res_coverage.pop(uid)

        for s_type, count in cov.items():
            if count > 0:
                loss += capacity

        for cell in cells:
            occupied[cell.r][cell.c] = False
            residential_at[cell.r][cell.c] = None

        return loss

    def place_utility(service_type, cells):
        gain = 0
        affected = set()

        for cell in cells:
            occupied[cell.r][cell.c] = True
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr, nc))

        for nr, nc in affected:
            influence_grid[nr][nc][service_type] = influence_grid[nr][nc].get(service_type, 0) + 1

            ruid = residential_at[nr][nc]
            if ruid is not None:
                cov = res_coverage[ruid]
                old = cov.get(service_type, 0)
                cov[service_type] = old + 1
                if old == 0:
                    gain += placements[ruid][4]   # capacity do residencial

        return gain

    def remove_utility(service_type, cells):
        loss = 0
        affected = set()

        for cell in cells:
            occupied[cell.r][cell.c] = False
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr, nc))

        for nr, nc in affected:
            grid_cell = influence_grid[nr][nc]
            grid_cell[service_type] -= 1
            if grid_cell[service_type] == 0:
                del grid_cell[service_type]

            ruid = residential_at[nr][nc]
            if ruid is not None:
                cov = res_coverage[ruid]
                cov[service_type] -= 1
                if cov[service_type] == 0:
                    loss += placements[ruid][4]
                    del cov[service_type]

        return loss

    def place(b, r, c):
        nonlocal current_score, next_uid

        proj = city.get_project(b)
        cells = can_place(proj, r, c)
        if not cells:
            return False, None

        uid = next_uid
        next_uid += 1

        if proj.build_type == "R":
            val = proj.capacity
            gain = place_residential(uid, val, cells)
        else:
            val = proj.service_type
            gain = place_utility(val, cells)

        current_score += gain
        placements[uid] = [b, r, c, proj.build_type, val, cells]
        add_active_uid(uid)
        return True, uid

    def remove(uid):
        nonlocal current_score

        b, r, c, b_type, val, cells = placements.pop(uid)

        if b_type == "R":
            loss = remove_residential(uid, val, cells)
        else:
            loss = remove_utility(val, cells)

        current_score -= loss
        remove_active_uid(uid)
        return [b, r, c, b_type, val, cells]

    def insert_with_uid(uid, b, r, c, b_type, val, cells):
        nonlocal current_score

        if b_type == "R":
            gain = place_residential(uid, val, cells)
        else:
            gain = place_utility(val, cells)

        current_score += gain
        placements[uid] = [b, r, c, b_type, val, cells]
        add_active_uid(uid)

    def capture_solution():
        return [(p[0], p[1], p[2]) for p in placements.values()]

    def acceptance_probability(delta, temperature):
        if delta >= 0:
            return 1.0
        if temperature <= 1e-12:
            return 0.0
        return math.exp(delta / temperature)

    def accept_move(delta, temperature):
        return random.random() <= acceptance_probability(delta, temperature)

    def random_position_for(proj):
        max_r = max(0, H - proj.h)
        max_c = max(0, W - proj.w)
        return random.randint(0, max_r), random.randint(0, max_c)

    def select_add_project():
        roll = random.random()
        if roll < 0.60 and top_res:
            return random.choice(top_res)
        if roll < 0.90 and selected_utils:
            return random.choice(selected_utils)
        return city.get_project(random.randint(0, city.B - 1))

    def choose_similar_project(old_proj):
        """
        Escolhe um projeto mais inteligente para CHANGE:
        1) normalmente mantém o mesmo tipo
        2) tenta shape semelhante
        3) para R prefere boa razão capacidade/área
        4) para U prefere utility compacta / tipo útil
        """
        target_type = old_proj.build_type
        if random.random() > same_type_change_prob:
            target_type = "U" if old_proj.build_type == "R" else "R"

        if target_type == "R":
            by_shape = residential_by_shape
        else:
            by_shape = utility_by_shape

        candidates = []

        if random.random() < similar_shape_change_prob:
            for dh in range(-shape_slack, shape_slack + 1):
                for dw in range(-shape_slack, shape_slack + 1):
                    key = (old_proj.h + dh, old_proj.w + dw)
                    if key in by_shape:
                        candidates.extend(by_shape[key])

        if not candidates:
            if target_type == "R":
                candidates = top_res[:] if top_res else residential[:]
                extra = random.sample(residential, min(12, len(residential))) if residential else []
                candidates.extend(extra)
            else:
                candidates = selected_utils[:] if selected_utils else utilities[:]
                extra = random.sample(utilities, min(12, len(utilities))) if utilities else []
                candidates.extend(extra)

        seen = set()
        dedup = []
        for p in candidates:
            if p.project_id not in seen:
                seen.add(p.project_id)
                dedup.append(p)
        candidates = dedup

        if len(candidates) > 1:
            candidates = [p for p in candidates if p.project_id != old_proj.project_id] or candidates

        if target_type == "R":
            candidates.sort(
                key=lambda p: (
                    p.h != old_proj.h,
                    p.w != old_proj.w,
                    -p.capacity / max(1, len(p.hash_offsets))
                )
            )
        else:
            candidates.sort(
                key=lambda p: (
                    p.h != old_proj.h,
                    p.w != old_proj.w,
                    len(p.hash_offsets)
                )
            )

        top_slice = candidates[:min(10, len(candidates))]
        return random.choice(top_slice) if top_slice else old_proj

    # =========================================================
    # INITIAL SOLUTION
    # =========================================================
    print("[MODE] Initial solution")

    if use_greedy_init:
        print("  -> Using greedy warm start")

        warm_runtime = max(10, int(max_runtime_s * 0.45))
        warm_solution = greedy(
            city,
            seed=seed,
            max_runtime_s=warm_runtime,
            top_k_res=max(top_k_res, 40),
            max_utility_types=max_utility_types,
        )

        loaded = 0
        for b, r, c in warm_solution:
            if time.time() - t0 > max_runtime_s * 0.60:
                print("  [INFO] Warm start truncated due to time budget.")
                break
            success, _ = place(b, r, c)
            if success:
                loaded += 1

        print(f"  -> Warm start loaded: placements={loaded} | score={current_score}")
    else:
        print("  -> Using lightweight constructive init")

        placed_res = 0
        placed_util = 0

        for r in range(0, H, 3):
            if r % 60 == 0:
                print(f"  Init row {r}/{H} | placements={len(placements)} | score={current_score}")

            for c in range(0, W, 3):
                for u in selected_utils:
                    if place(u.project_id, r, c)[0]:
                        placed_util += 1
                for res in top_res:
                    rr = r + 1
                    cc = c + 1
                    if rr < H and cc < W and place(res.project_id, rr, cc)[0]:
                        placed_res += 1
                        break

        print(f"  -> Constructive init finished: R={placed_res} U={placed_util} | score={current_score}")

    best_score = current_score
    best_solution = capture_solution()

    # =========================================================
    # SA LOOP
    # =========================================================
    temperature = initial_temperature
    current_reheats = 0

    ops = ["ADD", "REMOVE", "MOVE", "CHANGE"]
    weights = [add_weight, remove_weight, move_weight, change_weight]

    print("[MODE] Simulated Annealing fine-tuning")

    for it in range(max_iterations):
        if time.time() - t0 > max_runtime_s * 0.98:
            print("  [INFO] Time budget reached: stopping early.")
            break

        if temperature <= min_temperature:
            if current_reheats < reheats:
                current_reheats += 1
                temperature = max(initial_temperature * reheat_factor, min_temperature * 10)
                print(f"  [INFO] Reheat #{current_reheats} -> T={temperature:.3f}")
            else:
                print("  [INFO] Minimum temperature reached.")
                break

        op = random.choices(ops, weights=weights, k=1)[0]

        if op == "ADD":
            proj = select_add_project()
            r, c = random_position_for(proj)

            old_score = current_score
            success, uid = place(proj.project_id, r, c)

            if success:
                delta = current_score - old_score
                if not accept_move(delta, temperature):
                    remove(uid)

        elif active_uids:
            uid = random.choice(active_uids)

            if op == "REMOVE":
                old_score = current_score
                old_data = remove(uid)
                delta = current_score - old_score

                if not accept_move(delta, temperature):
                    insert_with_uid(uid, *old_data)

            elif op == "MOVE":
                old_score = current_score
                old_data = remove(uid)
                old_b, old_r, old_c, old_type, old_val, old_cells = old_data
                old_proj = city.get_project(old_b)

                nr = max(0, min(H - old_proj.h, old_r + random.randint(-neighborhood_radius, neighborhood_radius)))
                nc = max(0, min(W - old_proj.w, old_c + random.randint(-neighborhood_radius, neighborhood_radius)))

                success, new_uid = place(old_b, nr, nc)
                if success:
                    delta = current_score - old_score
                    if not accept_move(delta, temperature):
                        remove(new_uid)
                        insert_with_uid(uid, *old_data)
                else:
                    insert_with_uid(uid, *old_data)

            elif op == "CHANGE":
                old_score = current_score
                old_data = remove(uid)
                old_b, old_r, old_c, old_type, old_val, old_cells = old_data
                old_proj = city.get_project(old_b)

                new_proj = choose_similar_project(old_proj)
                success, new_uid = place(new_proj.project_id, old_r, old_c)

                if success:
                    delta = current_score - old_score
                    if not accept_move(delta, temperature):
                        remove(new_uid)
                        insert_with_uid(uid, *old_data)
                else:
                    insert_with_uid(uid, *old_data)

        if current_score > best_score:
            best_score = current_score
            best_solution = capture_solution()

        temperature *= cooling_rate

        if it % 500 == 0:
            print(
                f"  Iteration {it}/{max_iterations} | "
                f"placements={len(placements)} | "
                f"score={current_score} | best={best_score} | T={temperature:.3f}"
            )

    dt = time.time() - t0
    print(
        f"Simulated Annealing finished: "
        f"placements={len(best_solution)} | best_score={best_score} | time={dt:.2f}s"
    )
    print("======== END SIMULATED ANNEALING ========")

    return best_solution