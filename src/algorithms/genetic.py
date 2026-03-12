import random
import time
from coordinates import Coordinates
from algorithms.greedy import greedy


def genetic_algorithm(
    city,
    population_size=10,
    generations=35,
    elite_size=2,
    tournament_size=3,
    crossover_rate=0.90,
    mutation_rate=0.40,
    max_runtime_s=1000,
    seed=0,
    top_k_res=40,
    destroy_repair_rate=0.45,
    crossover_type="clustered",

    # --- adaptive population ---
    min_population_size=6,
    max_population_size=18,
    stagnation_generations=6,
    population_growth_step=4,
    population_shrink_step=2,
):
    """
    Memetic Genetic Algorithm for urban planning optimization.

    Combines genetic operators (crossover and mutation) with intensive local search
    to find high-quality solutions for the problem of placing residential and utility
    buildings on an urban grid.

    Args:
        city: City instance containing grid, projects and constraints
        population_size: Initial population size
        generations: Number of generations to run
        elite_size: Number of elite individuals preserved
        tournament_size: Tournament size for selection
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        max_runtime_s: Maximum runtime in seconds
        seed: Random seed for reproducibility
        top_k_res: Top K residential projects to consider
        destroy_repair_rate: Destroy & repair rate in mutation
        crossover_type: Crossover type ("spatial", "score_aware", "clustered")
        min_population_size: Adaptive minimum population size
        max_population_size: Adaptive maximum population size
        stagnation_generations: Stagnation generations before expanding population
        population_growth_step: Increment when expanding population
        population_shrink_step: Decrement when shrinking population

    Returns:
        List of tuples (building_id, row, col) representing the best solution
    """
    random.seed(seed)
    t0 = time.time()

    H, W, D = city.H, city.W, city.D
    projects = city.projects
    B = city.B

    # ---------------------------------------------------------
    # Automatic population limits adjustment
    # ---------------------------------------------------------
    if min_population_size is None:
        min_population_size = max(6, population_size - max(2, population_size // 4))

    if max_population_size is None:
        if B <= 20:
            max_population_size = max(population_size + 4, 12)
        elif D <= 2:
            max_population_size = max(population_size + 6, 16)
        elif D >= 15:
            max_population_size = max(population_size + 8, 18)
        else:
            max_population_size = max(population_size + 6, 16)

    if population_growth_step is None:
        population_growth_step = max(2, population_size // 3)

    if population_shrink_step is None:
        population_shrink_step = max(1, population_size // 5)

    base_population_size = population_size
    current_population_target = population_size

    residential = [p for p in projects if p.build_type == "R"]
    utilities = [p for p in projects if p.build_type == "U"]

    residential_sorted = sorted(
        residential,
        key=lambda p: (
            p.capacity / max(1, len(p.hash_offsets)),
            p.capacity,
            -len(p.hash_offsets),
        ),
        reverse=True,
    )
    top_res_projects = residential_sorted[:top_k_res]

    utility_candidates_by_type = {}
    for u in utilities:
        utility_candidates_by_type.setdefault(u.service_type, []).append(u)

    for st in utility_candidates_by_type:
        utility_candidates_by_type[st].sort(
            key=lambda u: (len(u.hash_offsets), u.h * u.w, -u.project_id)
        )
        utility_candidates_by_type[st] = utility_candidates_by_type[st][:3]

    utility_types = sorted(utility_candidates_by_type.keys())

    print("========== MEMETIC ALGORITHM (GA + Local Search) ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")
    print(f"Base population size: {base_population_size}")
    print(f"Population range: [{min_population_size}, {max_population_size}]")
    print(f"Generations: {generations}")
    print(f"Crossover type: {crossover_type}")
    print("Start evolving...")

    cell_cache = {}
    influence_cache = {}

    # =========================================================
    # Low-level geometry helpers
    # =========================================================
    def get_cells(b_id, r, c):
        """Get cells occupied by a project at position (r,c) with caching."""
        key = (b_id, r, c)
        if key not in cell_cache:
            cell_cache[key] = city.get_project(b_id).absolute_hash_cells(Coordinates(r, c))
        return cell_cache[key]

    def get_influence_cells(r, c):
        """Return cells within influence area of distance D from position (r,c)."""
        key = (r, c)
        if key not in influence_cache:
            cells = []
            for dr in range(-D, D + 1):
                rem = D - abs(dr)
                for dc in range(-rem, rem + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        cells.append((nr, nc))
            influence_cache[key] = cells
        return influence_cache[key]

    def rect_intersects_cells(r0, c0, r1, c1, cells):
        """Check if any cell intersects the rectangle defined by (r0,c0) and (r1,c1)."""
        for cell in cells:
            if r0 <= cell.r <= r1 and c0 <= cell.c <= c1:
                return True
        return False

    def random_window():
        """Generate a random rectangular window in the grid for local operations."""
        h_span = max(10, min(H // 4, max(20, H // 8)))
        w_span = max(10, min(W // 4, max(20, W // 8)))
        r0 = random.randint(0, max(0, H - h_span))
        r1 = min(H - 1, r0 + random.randint(max(5, h_span // 2), h_span))
        c0 = random.randint(0, max(0, W - w_span))
        c1 = min(W - 1, c0 + random.randint(max(5, w_span // 2), w_span))
        return r0, c0, r1, c1

    def placement_center(cells):
        """Calculate the geometric center of a set of cells."""
        sr = sum(cell.r for cell in cells)
        sc = sum(cell.c for cell in cells)
        n = max(1, len(cells))
        return sr / n, sc / n

    def placement_in_region(cells, r0, c0, r1, c1):
        """Check if the center of a set of cells is within a region."""
        cr, cc = placement_center(cells)
        return r0 <= cr <= r1 and c0 <= cc <= c1

    # =========================================================
    # Individual representation
    # =========================================================
    def make_empty_individual():
        """Create an empty individual with initialized data structures."""
        return {
            "occupied": set(),
            "influence_grid": {},
            "residential_at": {},
            "placements": {},
            "res_coverage": {},
            "active_uids": [],
            "uid_to_idx": {},
            "next_uid": 0,
            "score": 0,
        }

    def clone_individual(ind):
        """Create a deep copy of an individual."""
        return {
            "occupied": ind["occupied"].copy(),
            "influence_grid": {k: v.copy() for k, v in ind["influence_grid"].items()},
            "residential_at": ind["residential_at"].copy(),
            "placements": {k: v[:] for k, v in ind["placements"].items()},
            "res_coverage": {k: v.copy() for k, v in ind["res_coverage"].items()},
            "active_uids": ind["active_uids"][:],
            "uid_to_idx": ind["uid_to_idx"].copy(),
            "next_uid": ind["next_uid"],
            "score": ind["score"],
        }

    def add_active_uid(ind, uid):
        """Add a UID to the active list with O(1) indexing."""
        ind["uid_to_idx"][uid] = len(ind["active_uids"])
        ind["active_uids"].append(uid)

    def remove_active_uid(ind, uid):
        """Remove a UID from the active list with O(1) operation."""
        idx = ind["uid_to_idx"].pop(uid)
        last_uid = ind["active_uids"].pop()
        if idx < len(ind["active_uids"]):
            ind["active_uids"][idx] = last_uid
            ind["uid_to_idx"][last_uid] = idx

    def can_place(ind, proj, r, c):
        """Check if a project can be placed at position (r,c) without conflicts."""
        if r < 0 or r + proj.h > H or c < 0 or c + proj.w > W:
            return False
        cells = get_cells(proj.project_id, r, c)
        for cell in cells:
            if (cell.r, cell.c) in ind["occupied"]:
                return False
        return cells

    def place_residential(ind, uid, capacity, cells):
        """Place a residential building and update score based on service coverage."""
        cov = {}

        for cell in cells:
            pos = (cell.r, cell.c)
            ind["occupied"].add(pos)
            ind["residential_at"][pos] = uid
            if pos in ind["influence_grid"]:
                for s, count in ind["influence_grid"][pos].items():
                    cov[s] = cov.get(s, 0) + count

        distinct_services = sum(1 for count in cov.values() if count > 0)
        gain = capacity * distinct_services

        ind["res_coverage"][uid] = cov
        ind["score"] += gain

    def remove_residential(ind, uid, capacity, cells):
        """Remove a residential building and update the score accordingly."""
        cov = ind["res_coverage"].pop(uid, {})
        distinct_services = sum(1 for count in cov.values() if count > 0)
        loss = capacity * distinct_services

        for cell in cells:
            pos = (cell.r, cell.c)
            ind["occupied"].remove(pos)
            del ind["residential_at"][pos]

        ind["score"] -= loss

    def place_utility(ind, uid, service_type, cells):
        """Place a utility building and update residential coverage scores."""
        gain = 0
        affected = set()

        for cell in cells:
            ind["occupied"].add((cell.r, cell.c))
            for pos in get_influence_cells(cell.r, cell.c):
                affected.add(pos)

        for pos in affected:
            grid_cell = ind["influence_grid"].setdefault(pos, {})
            grid_cell[service_type] = grid_cell.get(service_type, 0) + 1

            res_uid = ind["residential_at"].get(pos)
            if res_uid is not None:
                cov = ind["res_coverage"][res_uid]
                old_count = cov.get(service_type, 0)
                cov[service_type] = old_count + 1
                if old_count == 0:
                    gain += ind["placements"][res_uid][4]

        ind["score"] += gain

    def remove_utility(ind, uid, service_type, cells):
        """Remove a utility building and update residential coverage scores."""
        loss = 0
        affected = set()

        for cell in cells:
            ind["occupied"].remove((cell.r, cell.c))
            for pos in get_influence_cells(cell.r, cell.c):
                affected.add(pos)

        for pos in affected:
            grid_cell = ind["influence_grid"][pos]
            grid_cell[service_type] -= 1

            if grid_cell[service_type] == 0:
                del grid_cell[service_type]
                if not grid_cell:
                    del ind["influence_grid"][pos]

            res_uid = ind["residential_at"].get(pos)
            if res_uid is not None:
                cov = ind["res_coverage"][res_uid]
                cov[service_type] -= 1
                if cov[service_type] == 0:
                    loss += ind["placements"][res_uid][4]

        ind["score"] -= loss

    def add_new(ind, b_id, r, c):
        """Add a new building placement to the individual if valid."""
        proj = city.get_project(b_id)
        cells = can_place(ind, proj, r, c)
        if cells is False:
            return False

        uid = ind["next_uid"]
        ind["next_uid"] += 1
        val = proj.capacity if proj.build_type == "R" else proj.service_type

        if proj.build_type == "R":
            place_residential(ind, uid, val, cells)
        else:
            place_utility(ind, uid, val, cells)

        ind["placements"][uid] = [b_id, r, c, proj.build_type, val, cells]
        add_active_uid(ind, uid)
        return uid

    def remove_uid(ind, uid):
        """Remove a building placement by UID from the individual."""
        if uid not in ind["placements"]:
            return None

        b_id, r, c, b_type, val, cells = ind["placements"].pop(uid)

        if b_type == "R":
            remove_residential(ind, uid, val, cells)
        else:
            remove_utility(ind, uid, val, cells)

        remove_active_uid(ind, uid)
        return [b_id, r, c, b_type, val, cells]

    # =========================================================
    # Build / initialize
    # =========================================================
    def build_from_solution(solution):
        """Build an individual from a list of (building_id, row, col) placements."""
        ind = make_empty_individual()
        for b_id, r, c in solution:
            add_new(ind, b_id, r, c)
        return ind

    def diversified_seed(base_solution):
        """Create a diversified individual from a base solution using destroy-repair."""
        ind = build_from_solution(base_solution)
        return destroy_and_repair(ind)

    def spawn_from_parent(parent, strong=False):
        """Spawn a new individual from a parent with mutation and local search."""
        child = clone_individual(parent)
        child = destroy_and_repair(child)
        child = memetic_local_search(child, steps=16 if strong else 10)
        return child

    # =========================================================
    # Region fill / constructive helpers
    # =========================================================
    def get_region_residential_uids(ind, r0, c0, r1, c1):
        """Get all residential UIDs that intersect with the given region."""
        out = []
        for uid in ind["active_uids"]:
            _, _, _, b_type, _, cells = ind["placements"][uid]
            if b_type == "R" and rect_intersects_cells(r0, c0, r1, c1, cells):
                out.append(uid)
        return out

    def estimate_utility_gain(ind, proj, r, c, region_res_uids_set):
        """Estimate the gain from placing a utility at (r,c) for residentials in region."""
        cells = can_place(ind, proj, r, c)
        if cells is False:
            return -1

        st = proj.service_type
        affected_res = set()

        for cell in cells:
            for pos in get_influence_cells(cell.r, cell.c):
                res_uid = ind["residential_at"].get(pos)
                if res_uid is not None and res_uid in region_res_uids_set:
                    affected_res.add(res_uid)

        gain = 0
        for res_uid in affected_res:
            if ind["res_coverage"].get(res_uid, {}).get(st, 0) == 0:
                gain += ind["placements"][res_uid][4]

        footprint_penalty = 0.10 * len(cells)
        return gain - footprint_penalty

    def guided_random_fill_region(ind, r0, c0, r1, c1, attempts_scale=1.0):
        """Fill a region with utilities and residentials using guided random placement."""
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(H - 1, r1)
        c1 = min(W - 1, c1)
        if r0 > r1 or c0 > c1:
            return

        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        util_rounds = max(6, min(28, int(area / 900) + int(6 * attempts_scale)))
        res_attempts = max(50, min(800, int(area / 28) + int(60 * attempts_scale)))

        for _ in range(util_rounds):
            region_res_uids = set(get_region_residential_uids(ind, r0, c0, r1, c1))
            if not region_res_uids:
                break

            best_choice = None
            best_gain = -1e18

            sampled_types = random.sample(utility_types, min(len(utility_types), 3))
            for st in sampled_types:
                for proj in utility_candidates_by_type[st]:
                    rr_min = max(0, r0 - D - 3)
                    rr_max = min(H - proj.h, r1 + D + 3)
                    cc_min = max(0, c0 - D - 3)
                    cc_max = min(W - proj.w, c1 + D + 3)
                    if rr_min > rr_max or cc_min > cc_max:
                        continue

                    for _ in range(8):
                        rr = random.randint(rr_min, rr_max)
                        cc = random.randint(cc_min, cc_max)
                        gain = estimate_utility_gain(ind, proj, rr, cc, region_res_uids)
                        if gain > best_gain:
                            best_gain = gain
                            best_choice = (proj.project_id, rr, cc)

            if best_choice is not None and best_gain >= 0:
                add_new(ind, *best_choice)

        for _ in range(res_attempts):
            proj = random.choice(top_res_projects)

            rr_min = max(0, r0 - D)
            rr_max = min(H - proj.h, r1 + D)
            cc_min = max(0, c0 - D)
            cc_max = min(W - proj.w, c1 + D)
            if rr_min > rr_max or cc_min > cc_max:
                continue

            if random.random() < 0.70 and ind["active_uids"]:
                anchor_uid = random.choice(ind["active_uids"])
                _, ar, ac, _, _, _ = ind["placements"][anchor_uid]
                rr = max(rr_min, min(rr_max, ar + random.randint(-D - 2, D + 2)))
                cc = max(cc_min, min(cc_max, ac + random.randint(-D - 2, D + 2)))
            else:
                rr = random.randint(rr_min, rr_max)
                cc = random.randint(cc_min, cc_max)

            add_new(ind, proj.project_id, rr, cc)

    # =========================================================
    # Heuristics for crossovers
    # =========================================================
    def placement_local_score(ind, uid):
        """Calculate a local quality score for a placement considering its context."""
        _, _, _, b_type, val, cells = ind["placements"][uid]

        if b_type == "R":
            cov = ind["res_coverage"].get(uid, {})
            distinct_services = sum(1 for cnt in cov.values() if cnt > 0)
            return 1.60 * distinct_services * val + 0.30 * val - 0.18 * len(cells)

        st = val
        helped_capacity = 0
        uniquely_helped_capacity = 0
        seen_res = set()

        for cell in cells:
            for pos in get_influence_cells(cell.r, cell.c):
                res_uid = ind["residential_at"].get(pos)
                if res_uid is None or res_uid in seen_res:
                    continue
                seen_res.add(res_uid)

                cap = ind["placements"][res_uid][4]
                helped_capacity += cap

                cov = ind["res_coverage"].get(res_uid, {})
                if cov.get(st, 0) == 1:
                    uniquely_helped_capacity += cap

        return 1.80 * uniquely_helped_capacity + 0.30 * helped_capacity - 0.20 * len(cells)

    def add_placements_in_order(child, placements_list):
        """Add placements to child in descending priority order."""
        placements_list.sort(reverse=True, key=lambda x: x[0])
        for _, b_id, r, c in placements_list:
            add_new(child, b_id, r, c)

    def estimate_utility_marginal_impact(ind, uid):
        """Estimate the marginal impact of a utility on residential coverage."""
        _, _, _, b_type, service_type, cells = ind["placements"][uid]
        if b_type != "U":
            return -1

        impacted_res = set()
        unique_support_capacity = 0
        total_reached_capacity = 0

        for cell in cells:
            for pos in get_influence_cells(cell.r, cell.c):
                res_uid = ind["residential_at"].get(pos)
                if res_uid is None or res_uid in impacted_res:
                    continue

                impacted_res.add(res_uid)
                cap = ind["placements"][res_uid][4]
                total_reached_capacity += cap

                cov = ind["res_coverage"].get(res_uid, {})
                if cov.get(service_type, 0) == 1:
                    unique_support_capacity += cap

        return 2.20 * unique_support_capacity + 0.40 * total_reached_capacity - 0.22 * len(cells)

    def select_best_anchor_utilities(parent, max_anchors=3):
        """Select the best utility placements to use as anchors for clustering."""
        utility_uids = [
            uid for uid in parent["active_uids"]
            if parent["placements"][uid][3] == "U"
        ]
        if not utility_uids:
            return []

        ranked = [
            (estimate_utility_marginal_impact(parent, uid), uid)
            for uid in utility_uids
        ]
        ranked.sort(reverse=True, key=lambda x: x[0])

        k = min(len(ranked), max_anchors)
        chosen_k = random.randint(1, k)
        return [uid for _, uid in ranked[:chosen_k]]

    def cluster_item_priority(item, parent):
        """Calculate priority of an item within a cluster for ordering."""
        uid, _, _, _, b_type, val, cells = item

        if b_type == "U":
            return 3, estimate_utility_marginal_impact(parent, uid), -len(cells)

        cov = parent["res_coverage"].get(uid, {})
        distinct_services = sum(1 for cnt in cov.values() if cnt > 0)
        return 2, 1.4 * distinct_services * val + 0.25 * val, -len(cells)

    def get_cluster_from_utility(parent, utility_uid):
        """Extract a cluster of buildings around a utility anchor."""
        cluster = []

        b_id, r, c, b_type, val, cells = parent["placements"][utility_uid]
        if b_type != "U":
            return cluster

        cluster.append((utility_uid, b_id, r, c, b_type, val, cells))

        nearby_res = set()
        nearby_utils = set()

        for cell in cells:
            for pos in get_influence_cells(cell.r, cell.c):
                res_uid = parent["residential_at"].get(pos)
                if res_uid is not None:
                    nearby_res.add(res_uid)

        cr, cc = placement_center(cells)
        radius = D + 4

        for uid in parent["active_uids"]:
            if uid == utility_uid:
                continue
            pb_id, pr, pc, pb_type, pval, pcells = parent["placements"][uid]
            if pb_type != "U":
                continue
            pcr, pcc = placement_center(pcells)
            if abs(pcr - cr) + abs(pcc - cc) <= radius:
                nearby_utils.add(uid)

        for uid in nearby_res:
            pb_id, pr, pc, pb_type, pval, pcells = parent["placements"][uid]
            cluster.append((uid, pb_id, pr, pc, pb_type, pval, pcells))

        for uid in nearby_utils:
            pb_id, pr, pc, pb_type, pval, pcells = parent["placements"][uid]
            cluster.append((uid, pb_id, pr, pc, pb_type, pval, pcells))

        return cluster

    def bounding_box_of_cluster(cluster, pad=D):
        """Calculate the bounding box of a cluster with optional padding."""
        if not cluster:
            return 0, 0, H - 1, W - 1

        min_r, min_c = H - 1, W - 1
        max_r, max_c = 0, 0

        for _, _, _, _, _, _, cells in cluster:
            for cell in cells:
                min_r = min(min_r, cell.r)
                min_c = min(min_c, cell.c)
                max_r = max(max_r, cell.r)
                max_c = max(max_c, cell.c)

        return (
            max(0, min_r - pad),
            max(0, min_c - pad),
            min(H - 1, max_r + pad),
            min(W - 1, max_c + pad),
        )

    # =========================================================
    # Evolution operators
    # =========================================================
    def destroy_and_repair(ind):
        """Destroy a random region of the individual and repair with new placements."""
        child = clone_individual(ind)
        r0, c0, r1, c1 = random_window()

        to_remove = []
        for uid in child["active_uids"]:
            cells = child["placements"][uid][5]
            if rect_intersects_cells(r0, c0, r1, c1, cells):
                to_remove.append(uid)

        for uid in to_remove:
            remove_uid(child, uid)

        guided_random_fill_region(child, r0, c0, r1, c1, attempts_scale=1.0)
        return child

    def spatial_crossover(parent1, parent2):
        """Perform spatial crossover by swapping regions between parents."""
        r0, c0, r1, c1 = random_window()
        child1, child2 = make_empty_individual(), make_empty_individual()

        p1_inside, p1_outside = [], []
        p2_inside, p2_outside = [], []

        for uid in parent1["active_uids"]:
            b_id, r, c, _, _, cells = parent1["placements"][uid]
            prio = placement_local_score(parent1, uid)
            if placement_in_region(cells, r0, c0, r1, c1):
                p1_inside.append((prio, b_id, r, c))
            else:
                p1_outside.append((prio, b_id, r, c))

        for uid in parent2["active_uids"]:
            b_id, r, c, _, _, cells = parent2["placements"][uid]
            prio = placement_local_score(parent2, uid)
            if placement_in_region(cells, r0, c0, r1, c1):
                p2_inside.append((prio, b_id, r, c))
            else:
                p2_outside.append((prio, b_id, r, c))

        add_placements_in_order(child1, p1_inside)
        add_placements_in_order(child1, p2_outside)

        add_placements_in_order(child2, p2_inside)
        add_placements_in_order(child2, p1_outside)

        guided_random_fill_region(
            child1,
            max(0, r0 - D), max(0, c0 - D),
            min(H - 1, r1 + D), min(W - 1, c1 + D),
            attempts_scale=0.45
        )
        guided_random_fill_region(
            child2,
            max(0, r0 - D), max(0, c0 - D),
            min(H - 1, r1 + D), min(W - 1, c1 + D),
            attempts_scale=0.45
        )

        return child1, child2

    def score_aware_crossover(parent1, parent2):
        """Perform crossover based on placement quality scores."""
        child1, child2 = make_empty_individual(), make_empty_individual()

        ranked1 = [
            (placement_local_score(parent1, uid), parent1["placements"][uid][0], parent1["placements"][uid][1], parent1["placements"][uid][2])
            for uid in parent1["active_uids"]
        ]
        ranked2 = [
            (placement_local_score(parent2, uid), parent2["placements"][uid][0], parent2["placements"][uid][1], parent2["placements"][uid][2])
            for uid in parent2["active_uids"]
        ]

        ranked1.sort(reverse=True, key=lambda x: x[0])
        ranked2.sort(reverse=True, key=lambda x: x[0])

        merged1 = []
        i = j = 0
        while i < len(ranked1) or j < len(ranked2):
            if i < len(ranked1):
                merged1.append(ranked1[i])
                i += 1
            if j < len(ranked2):
                merged1.append(ranked2[j])
                j += 1

        merged2 = ranked1[: len(ranked1) // 2] + ranked2[: len(ranked2) // 2]
        random.shuffle(merged2)

        add_placements_in_order(child1, merged1)
        add_placements_in_order(child2, merged2)

        if random.random() < 0.7:
            guided_random_fill_region(child1, *random_window(), attempts_scale=0.45)
        if random.random() < 0.7:
            guided_random_fill_region(child2, *random_window(), attempts_scale=0.45)

        return child1, child2

    def clustered_crossover(parent1, parent2):
        """Perform crossover by exchanging clusters of related buildings."""
        child1, child2 = make_empty_individual(), make_empty_individual()

        anchor_utils_1 = select_best_anchor_utilities(parent1, max_anchors=3)
        anchor_utils_2 = select_best_anchor_utilities(parent2, max_anchors=3)

        if not anchor_utils_1 or not anchor_utils_2:
            return spatial_crossover(parent1, parent2)

        cluster1, cluster2 = [], []
        seen1, seen2 = set(), set()

        for uid in anchor_utils_1:
            for item in get_cluster_from_utility(parent1, uid):
                if item[0] not in seen1:
                    seen1.add(item[0])
                    cluster1.append(item)

        for uid in anchor_utils_2:
            for item in get_cluster_from_utility(parent2, uid):
                if item[0] not in seen2:
                    seen2.add(item[0])
                    cluster2.append(item)

        cluster1.sort(reverse=True, key=lambda item: cluster_item_priority(item, parent1))
        cluster2.sort(reverse=True, key=lambda item: cluster_item_priority(item, parent2))

        for _, b_id, r, c, _, _, _ in cluster1:
            add_new(child1, b_id, r, c)

        parent2_ranked = [
            (placement_local_score(parent2, uid), parent2["placements"][uid][0], parent2["placements"][uid][1], parent2["placements"][uid][2])
            for uid in parent2["active_uids"]
        ]
        add_placements_in_order(child1, parent2_ranked)

        for _, b_id, r, c, _, _, _ in cluster2:
            add_new(child2, b_id, r, c)

        parent1_ranked = [
            (placement_local_score(parent1, uid), parent1["placements"][uid][0], parent1["placements"][uid][1], parent1["placements"][uid][2])
            for uid in parent1["active_uids"]
        ]
        add_placements_in_order(child2, parent1_ranked)

        guided_random_fill_region(child1, *bounding_box_of_cluster(cluster1, pad=D), attempts_scale=0.60)
        guided_random_fill_region(child2, *bounding_box_of_cluster(cluster2, pad=D), attempts_scale=0.60)

        return child1, child2

    def crossover(parent1, parent2):
        """Execute the selected crossover operator based on configuration."""
        if crossover_type == "score_aware":
            return score_aware_crossover(parent1, parent2)
        if crossover_type == "clustered":
            return clustered_crossover(parent1, parent2)
        return spatial_crossover(parent1, parent2)

    def memetic_local_search(ind, steps=12):
        """Apply local search with ADD, MOVE, and REMOVE operations."""
        for _ in range(steps):
            if not ind["active_uids"]:
                break

            op = random.choices(["ADD", "MOVE", "REMOVE"], weights=[0.35, 0.50, 0.15])[0]

            if op == "ADD":
                if random.random() < 0.72:
                    proj = random.choice(top_res_projects)
                else:
                    st = random.choice(utility_types)
                    proj = random.choice(utility_candidates_by_type[st])

                if proj.h > H or proj.w > W:
                    continue

                if random.random() < 0.75 and ind["active_uids"]:
                    uid = random.choice(ind["active_uids"])
                    _, br, bc, _, _, _ = ind["placements"][uid]
                    r = max(0, min(H - proj.h, br + random.randint(-D - 2, D + 2)))
                    c = max(0, min(W - proj.w, bc + random.randint(-D - 2, D + 2)))
                else:
                    r = random.randint(0, H - proj.h)
                    c = random.randint(0, W - proj.w)

                add_new(ind, proj.project_id, r, c)

            elif op == "MOVE":
                uid = random.choice(ind["active_uids"])
                old = remove_uid(ind, uid)
                if old is None:
                    continue

                old_b, old_r, old_c = old[0], old[1], old[2]
                proj = city.get_project(old_b)

                prev_score = ind["score"]
                best_pos = None
                best_score = prev_score

                for _ in range(3):
                    nr = max(0, min(H - proj.h, old_r + random.randint(-3, 3)))
                    nc = max(0, min(W - proj.w, old_c + random.randint(-3, 3)))
                    new_uid = add_new(ind, old_b, nr, nc)

                    if new_uid is not False:
                        if ind["score"] > best_score:
                            best_score = ind["score"]
                            best_pos = (nr, nc)
                        remove_uid(ind, new_uid)

                restored_uid = add_new(ind, old_b, old_r, old_c)

                if best_pos is not None:
                    remove_uid(ind, restored_uid)
                    add_new(ind, old_b, best_pos[0], best_pos[1])

            else:  # REMOVE
                uid = random.choice(ind["active_uids"])
                local_score = placement_local_score(ind, uid)
                if local_score < 0.25 * max(1, ind["score"] / max(1, len(ind["active_uids"]))):
                    remove_uid(ind, uid)

        return ind

    # =========================================================
    # Population adaptation helpers
    # =========================================================
    def expand_population(population, target_size):
        """Expand population to target size by spawning from elite individuals."""
        if len(population) >= target_size:
            return population

        ranked = sorted(population, key=lambda x: x["score"], reverse=True)
        expanded = [clone_individual(ind) for ind in ranked]

        elite_pool = ranked[:max(1, min(len(ranked), elite_size + 1))]
        while len(expanded) < target_size:
            if time.time() - t0 > max_runtime_s * 0.90:
                break
            parent = random.choice(elite_pool)
            expanded.append(spawn_from_parent(parent, strong=True))

        return expanded

    def shrink_population(population, target_size):
        """Shrink population to target size by keeping best individuals."""
        if len(population) <= target_size:
            return population
        ranked = sorted(population, key=lambda x: x["score"], reverse=True)
        return [clone_individual(ind) for ind in ranked[:target_size]]

    # =========================================================
    # Main loop
    # =========================================================
    base_seed = greedy(city)
    population = [build_from_solution(base_seed)]

    # Time-controlled initialization
    init_local_steps = 8 if D <= 5 else 12
    init_budget = 0.45 if H * W >= 500_000 else 0.35

    while len(population) < current_population_target:
        if time.time() - t0 > max_runtime_s * init_budget:
            break
        ind = diversified_seed(base_seed)
        ind = memetic_local_search(ind, steps=init_local_steps)
        population.append(ind)

    current_population_target = len(population)
    base_population_size = min(base_population_size, current_population_target)

    best_individual = max(population, key=lambda x: x["score"])
    best_score = best_individual["score"]
    best_generation = 0
    stagnation_counter = 0

    for gen in range(1, generations + 1):
        if time.time() - t0 > max_runtime_s * 0.98:
            break

        population = shrink_population(population, current_population_target)

        ranked = sorted(population, key=lambda x: x["score"], reverse=True)
        eff_elite_size = min(elite_size, len(ranked), current_population_target)
        next_population = [clone_individual(ind) for ind in ranked[:eff_elite_size]]

        while len(next_population) < current_population_target:
            if time.time() - t0 > max_runtime_s * 0.98:
                break

            sample_size = min(tournament_size, len(population))
            p1 = max(random.sample(population, sample_size), key=lambda x: x["score"])
            p2 = max(random.sample(population, sample_size), key=lambda x: x["score"])

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = clone_individual(p1), clone_individual(p2)

            if random.random() < mutation_rate:
                if random.random() < destroy_repair_rate:
                    c1 = destroy_and_repair(c1)
                else:
                    c1 = memetic_local_search(c1, steps=14)
            else:
                c1 = memetic_local_search(c1, steps=6)

            next_population.append(c1)

            if len(next_population) < current_population_target:
                if time.time() - t0 > max_runtime_s * 0.98:
                    break

                if random.random() < mutation_rate:
                    if random.random() < destroy_repair_rate:
                        c2 = destroy_and_repair(c2)
                    else:
                        c2 = memetic_local_search(c2, steps=14)
                else:
                    c2 = memetic_local_search(c2, steps=6)

                next_population.append(c2)

        population = next_population

        gen_best = max(population, key=lambda x: x["score"])
        improved = gen_best["score"] > best_score

        if improved:
            best_individual = clone_individual(gen_best)
            best_score = gen_best["score"]
            best_generation = gen
            stagnation_counter = 0

            if current_population_target > base_population_size:
                current_population_target = max(
                    base_population_size,
                    current_population_target - population_shrink_step
                )
        else:
            stagnation_counter += 1

            if stagnation_counter >= stagnation_generations:
                old_target = current_population_target
                current_population_target = min(
                    max_population_size,
                    current_population_target + population_growth_step
                )

                if current_population_target > old_target:
                    population = expand_population(population, current_population_target)
                    current_population_target = len(population)

                stagnation_counter = 0

        if gen % 5 == 0 or gen == 1:
            avg_score = sum(ind["score"] for ind in population) / max(1, len(population))
            print(
                f"  Generation {gen}/{generations} | "
                f"global_best={best_score} | avg={avg_score:.2f} | "
                f"pop={len(population)} | target={current_population_target}"
            )

    dt = time.time() - t0
    final_solution = [
        (b_id, r, c)
        for b_id, r, c, _, _, _ in best_individual["placements"].values()
    ]

    print(f"Final best internal score: {best_score}")
    print(f"Best found on generation: {best_generation}")
    print(f"Final population target: {current_population_target}")
    print(f"Genetic Algorithm finished | placements={len(final_solution)} | time={dt:.2f}s")
    print("======== END GENETIC ALGORITHM ========")

    return final_solution
