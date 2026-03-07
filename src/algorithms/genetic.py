import random
import time
from coordinates import Coordinates
from algorithms.greedy2 import greedy2


def genetic_algorithm(
    city,
    population_size=10,
    generations=60,
    elite_size=2,
    tournament_size=3,
    crossover_rate=0.90,
    mutation_rate=0.40,
    max_runtime_s=1000,
    seed=0,
    top_k_res=40,
    neighborhood_radius=3,
    max_mutation_tries=20,
    destroy_repair_rate=0.45,
):
    random.seed(seed)
    t0 = time.time()

    H, W, D = city.H, city.W, city.D
    projects = city.projects

    residential = [p for p in projects if p.build_type == "R"]
    utilities = [p for p in projects if p.build_type == "U"]

    residential_sorted = sorted(
        residential,
        key=lambda p: p.capacity / max(1, p.h * p.w),
        reverse=True,
    )
    top_res_projects = residential_sorted[:top_k_res]

    # top 3 menores por tipo de utility
    utility_candidates_by_type = {}
    for u in utilities:
        st = u.service_type
        utility_candidates_by_type.setdefault(st, []).append(u)

    for st in utility_candidates_by_type:
        utility_candidates_by_type[st].sort(
            key=lambda u: (len(u.hash_offsets), u.h * u.w, -u.project_id)
        )
        utility_candidates_by_type[st] = utility_candidates_by_type[st][:3]

    utility_types = sorted(utility_candidates_by_type.keys())

    print("========== GENETIC ALGORITHM ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")
    print(f"Population size: {population_size}")
    print(f"Generations: {generations}")
    print(f"Elite size: {elite_size}")
    print(f"Selected residential candidates: {len(top_res_projects)} (top_k_res={top_k_res})")
    print(f"Selected utility types (max): {len(utility_types)}")
    print("Start evolving...")

    cell_cache = {}

    # =========================================================
    # Basic geometric helpers
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

    def rect_intersects_cells(r0, c0, r1, c1, cells):
        for cell in cells:
            if r0 <= cell.r <= r1 and c0 <= cell.c <= c1:
                return True
        return False

    def placement_intersects_rect(ind, uid, r0, c0, r1, c1):
        cells = ind["placements"][uid][5]
        return rect_intersects_cells(r0, c0, r1, c1, cells)

    def random_window():
        h_span = max(10, min(H // 4, max(20, H // 8)))
        w_span = max(10, min(W // 4, max(20, W // 8)))

        if H <= h_span:
            r0, r1 = 0, H - 1
        else:
            r0 = random.randint(0, H - h_span)
            r1 = min(H - 1, r0 + random.randint(max(5, h_span // 2), h_span))

        if W <= w_span:
            c0, c1 = 0, W - 1
        else:
            c0 = random.randint(0, W - w_span)
            c1 = min(W - 1, c0 + random.randint(max(5, w_span // 2), w_span))

        return r0, c0, r1, c1

    # =========================================================
    # Individual representation
    # =========================================================
    def make_empty_individual():
        return {
            "occupied": [[False] * W for _ in range(H)],
            "influence_grid": [[{} for _ in range(W)] for _ in range(H)],
            "residential_at": [[None] * W for _ in range(H)],
            "placements": {},      # uid -> [b_id, r, c, b_type, val, cells]
            "res_coverage": {},    # uid -> {service_type: count}
            "active_uids": [],
            "uid_to_idx": {},
            "next_uid": 0,
            "score": 0,
        }

    def add_active_uid(ind, uid):
        ind["uid_to_idx"][uid] = len(ind["active_uids"])
        ind["active_uids"].append(uid)

    def remove_active_uid(ind, uid):
        idx = ind["uid_to_idx"].pop(uid)
        last_uid = ind["active_uids"].pop()
        if idx < len(ind["active_uids"]):
            ind["active_uids"][idx] = last_uid
            ind["uid_to_idx"][last_uid] = idx

    def can_place(ind, proj, r, c):
        if r < 0 or r + proj.h > H or c < 0 or c + proj.w > W:
            return False

        cells = get_cells(proj.project_id, r, c)
        for cell in cells:
            if ind["occupied"][cell.r][cell.c]:
                return False
        return cells

    # =========================================================
    # Incremental scoring primitives
    # =========================================================
    def place_residential(ind, uid, capacity, cells):
        gain = 0
        cov = {}

        for cell in cells:
            ind["occupied"][cell.r][cell.c] = True
            ind["residential_at"][cell.r][cell.c] = uid

            for s, count in ind["influence_grid"][cell.r][cell.c].items():
                cov[s] = cov.get(s, 0) + count

        for _, count in cov.items():
            if count > 0:
                gain += capacity

        ind["res_coverage"][uid] = cov
        ind["score"] += gain

    def remove_residential(ind, uid, capacity, cells):
        cov = ind["res_coverage"].pop(uid, {})
        loss = 0

        for _, count in cov.items():
            if count > 0:
                loss += capacity

        for cell in cells:
            ind["occupied"][cell.r][cell.c] = False
            ind["residential_at"][cell.r][cell.c] = None

        ind["score"] -= loss

    def place_utility(ind, uid, service_type, cells):
        for cell in cells:
            ind["occupied"][cell.r][cell.c] = True

        affected = set()
        gain = 0

        for cell in cells:
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr, nc))

        for nr, nc in affected:
            grid_cell = ind["influence_grid"][nr][nc]
            prev = grid_cell.get(service_type, 0)
            grid_cell[service_type] = prev + 1

            res_uid = ind["residential_at"][nr][nc]
            if res_uid is not None:
                cov = ind["res_coverage"][res_uid]
                prev_cov = cov.get(service_type, 0)
                cov[service_type] = prev_cov + 1
                if prev_cov == 0:
                    capacity = ind["placements"][res_uid][4]
                    gain += capacity

        ind["score"] += gain

    def remove_utility(ind, uid, service_type, cells):
        affected = set()
        loss = 0

        for cell in cells:
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr, nc))

        for nr, nc in affected:
            grid_cell = ind["influence_grid"][nr][nc]
            prev = grid_cell.get(service_type, 0)
            if prev <= 1:
                grid_cell.pop(service_type, None)
            else:
                grid_cell[service_type] = prev - 1

            res_uid = ind["residential_at"][nr][nc]
            if res_uid is not None:
                cov = ind["res_coverage"][res_uid]
                prev_cov = cov.get(service_type, 0)

                if prev_cov <= 1:
                    if service_type in cov:
                        cov.pop(service_type)
                    if prev_cov > 0:
                        capacity = ind["placements"][res_uid][4]
                        loss += capacity
                else:
                    cov[service_type] = prev_cov - 1

        for cell in cells:
            ind["occupied"][cell.r][cell.c] = False

        ind["score"] -= loss

    def add_new(ind, b_id, r, c):
        proj = city.get_project(b_id)
        cells = can_place(ind, proj, r, c)
        if cells is False:
            return False

        uid = ind["next_uid"]
        ind["next_uid"] += 1

        val = proj.capacity if proj.build_type == "R" else proj.service_type
        ind["placements"][uid] = [b_id, r, c, proj.build_type, val, cells]
        add_active_uid(ind, uid)

        if proj.build_type == "R":
            place_residential(ind, uid, proj.capacity, cells)
        else:
            place_utility(ind, uid, proj.service_type, cells)

        return uid

    def remove_uid(ind, uid):
        if uid not in ind["placements"]:
            return None

        b_id, r, c, b_type, val, cells = ind["placements"].pop(uid)

        if b_type == "R":
            remove_residential(ind, uid, val, cells)
        else:
            remove_utility(ind, uid, val, cells)

        remove_active_uid(ind, uid)
        return [b_id, r, c, b_type, val, cells]

    def snapshot_solution(ind):
        out = []
        for uid in ind["active_uids"]:
            b_id, r, c, _, _, _ = ind["placements"][uid]
            out.append((b_id, r, c))
        return out

    def clone_individual(ind):
        new_ind = make_empty_individual()
        for uid in ind["active_uids"]:
            b_id, r, c, _, _, _ = ind["placements"][uid]
            add_new(new_ind, b_id, r, c)
        return new_ind

    def internal_score(ind):
        return ind["score"]

    # =========================================================
    # Region helpers for coverage-aware repair
    # =========================================================
    def get_region_residential_uids(ind, r0, c0, r1, c1):
        out = []
        for uid in ind["active_uids"]:
            _, _, _, b_type, _, cells = ind["placements"][uid]
            if b_type == "R" and rect_intersects_cells(r0, c0, r1, c1, cells):
                out.append(uid)
        return out

    def estimate_utility_gain(ind, proj, r, c, region_res_uids):
        cells = can_place(ind, proj, r, c)
        if cells is False:
            return -1, None

        target_res = set(region_res_uids)
        affected_res = {}
        st = proj.service_type

        for cell in cells:
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                res_uid = ind["residential_at"][nr][nc]
                if res_uid is not None and res_uid in target_res:
                    affected_res[res_uid] = True

        gain = 0
        for res_uid in affected_res:
            cov = ind["res_coverage"].get(res_uid, {})
            if cov.get(st, 0) == 0:
                gain += ind["placements"][res_uid][4]

        return gain, cells

    def sample_positions_for_project_near_region(proj, r0, c0, r1, c1, samples=20):
        positions = []

        rr_min = max(0, r0 - D - 3)
        rr_max = min(H - proj.h, r1 + D + 3)
        cc_min = max(0, c0 - D - 3)
        cc_max = min(W - proj.w, c1 + D + 3)

        if rr_min > rr_max or cc_min > cc_max:
            return positions

        for _ in range(samples):
            rr = random.randint(rr_min, rr_max)
            cc = random.randint(cc_min, cc_max)
            positions.append((rr, cc))

        return positions

    # =========================================================
    # Construction / seeding
    # =========================================================
    def build_from_solution(solution):
        ind = make_empty_individual()
        for b_id, r, c in solution:
            add_new(ind, b_id, r, c)
        return ind

    def guided_random_fill_region(ind, r0, c0, r1, c1, attempts_scale=1.0):
        area = (r1 - r0 + 1) * (c1 - c0 + 1)

        util_rounds = max(8, min(40, int(area / 800) + int(8 * attempts_scale)))
        res_attempts = max(80, min(1200, int(area / 20) + int(80 * attempts_scale)))

        for _ in range(util_rounds):
            region_res_uids = get_region_residential_uids(ind, r0, c0, r1, c1)
            best_choice = None
            best_gain = -1

            shuffled_types = utility_types[:]
            random.shuffle(shuffled_types)

            for st in shuffled_types:
                for proj in utility_candidates_by_type[st]:
                    for rr, cc in sample_positions_for_project_near_region(
                        proj, r0, c0, r1, c1, samples=10
                    ):
                        gain, _ = estimate_utility_gain(ind, proj, rr, cc, region_res_uids)
                        if gain > best_gain:
                            best_gain = gain
                            best_choice = (proj.project_id, rr, cc)

            if best_choice is not None and best_gain >= 0:
                add_new(ind, best_choice[0], best_choice[1], best_choice[2])

        for _ in range(res_attempts):
            proj = random.choice(top_res_projects)

            if random.random() < 0.80:
                rr = random.randint(max(0, r0 - D), min(H - proj.h, r1 + D))
                cc = random.randint(max(0, c0 - D), min(W - proj.w, c1 + D))
            else:
                if ind["active_uids"]:
                    uid = random.choice(ind["active_uids"])
                    _, br, bc, _, _, _ = ind["placements"][uid]
                    rr = max(0, min(H - proj.h, br + random.randint(-D - 4, D + 4)))
                    cc = max(0, min(W - proj.w, bc + random.randint(-D - 4, D + 4)))
                else:
                    rr = random.randint(0, max(0, H - proj.h))
                    cc = random.randint(0, max(0, W - proj.w))

            add_new(ind, proj.project_id, rr, cc)

        second_util_rounds = max(4, util_rounds // 3)
        for _ in range(second_util_rounds):
            region_res_uids = get_region_residential_uids(ind, r0, c0, r1, c1)
            best_choice = None
            best_gain = -1

            shuffled_types = utility_types[:]
            random.shuffle(shuffled_types)

            for st in shuffled_types:
                for proj in utility_candidates_by_type[st]:
                    for rr, cc in sample_positions_for_project_near_region(
                        proj, r0, c0, r1, c1, samples=8
                    ):
                        gain, _ = estimate_utility_gain(ind, proj, rr, cc, region_res_uids)
                        if gain > best_gain:
                            best_gain = gain
                            best_choice = (proj.project_id, rr, cc)

            if best_choice is not None and best_gain > 0:
                add_new(ind, best_choice[0], best_choice[1], best_choice[2])

    def guided_random_individual():
        ind = make_empty_individual()
        guided_random_fill_region(ind, 0, 0, H - 1, W - 1, attempts_scale=2.0)
        return ind

    def greedy2_seed():
        # greedy2(city) -> solution
        sol = greedy2(city)
        return build_from_solution(sol)

    # =========================================================
    # Strong mutation: destroy & repair
    # =========================================================
    def destroy_and_repair(ind):
        child = clone_individual(ind)

        r0, c0, r1, c1 = random_window()

        to_remove = []
        for uid in list(child["active_uids"]):
            if placement_intersects_rect(child, uid, r0, c0, r1, c1):
                to_remove.append(uid)

        random.shuffle(to_remove)
        for uid in to_remove:
            remove_uid(child, uid)

        guided_random_fill_region(child, r0, c0, r1, c1, attempts_scale=1.0)
        return child

    # =========================================================
    # Population initialization
    # =========================================================
    population = [greedy2_seed()]

    while len(population) < population_size:
        base = clone_individual(population[0])
        n_mut = random.randint(8, 20)

        for _ in range(n_mut):
            if random.random() < 0.60:
                base = destroy_and_repair(base)
            else:
                if base["active_uids"]:
                    uid = random.choice(base["active_uids"])
                    old = remove_uid(base, uid)
                    if old is not None:
                        old_b, old_r, old_c, _, _, _ = old
                        proj = city.get_project(old_b)
                        rr = max(0, min(H - proj.h, old_r + random.randint(-neighborhood_radius, neighborhood_radius)))
                        cc = max(0, min(W - proj.w, old_c + random.randint(-neighborhood_radius, neighborhood_radius)))
                        if add_new(base, old_b, rr, cc) is False:
                            add_new(base, old_b, old_r, old_c)

        population.append(base)

    # =========================================================
    # Selection
    # =========================================================
    def tournament_select(pop):
        sample = random.sample(pop, min(tournament_size, len(pop)))
        return max(sample, key=internal_score)

    # =========================================================
    # Spatial crossover
    # =========================================================
    def crossover(parent1, parent2):
        r0, c0, r1, c1 = random_window()

        child1 = make_empty_individual()
        child2 = make_empty_individual()

        p1_inside, p1_outside = [], []
        p2_inside, p2_outside = [], []

        for uid in parent1["active_uids"]:
            b_id, r, c, _, _, cells = parent1["placements"][uid]
            if rect_intersects_cells(r0, c0, r1, c1, cells):
                p1_inside.append((b_id, r, c))
            else:
                p1_outside.append((b_id, r, c))

        for uid in parent2["active_uids"]:
            b_id, r, c, _, _, cells = parent2["placements"][uid]
            if rect_intersects_cells(r0, c0, r1, c1, cells):
                p2_inside.append((b_id, r, c))
            else:
                p2_outside.append((b_id, r, c))

        for b_id, r, c in p1_inside:
            add_new(child1, b_id, r, c)
        for b_id, r, c in p2_outside:
            add_new(child1, b_id, r, c)

        for b_id, r, c in p2_inside:
            add_new(child2, b_id, r, c)
        for b_id, r, c in p1_outside:
            add_new(child2, b_id, r, c)

        return child1, child2

    # =========================================================
    # Light mutation
    # =========================================================
    def light_mutate(ind):
        child = clone_individual(ind)

        if not child["active_uids"]:
            return guided_random_individual()

        op = random.choices(
            ["ADD", "MOVE", "CHANGE", "REMOVE"],
            weights=[0.20, 0.35, 0.25, 0.20],
            k=1,
        )[0]

        if op == "REMOVE":
            uid = random.choice(child["active_uids"])
            remove_uid(child, uid)
            return child

        if op == "ADD":
            for _ in range(max_mutation_tries):
                choose_res = random.random() < 0.70
                if choose_res and top_res_projects:
                    proj = random.choice(top_res_projects)
                else:
                    st = random.choice(utility_types)
                    proj = random.choice(utility_candidates_by_type[st])

                if child["active_uids"] and random.random() < 0.75:
                    uid = random.choice(child["active_uids"])
                    _, br, bc, _, _, _ = child["placements"][uid]
                    rr = max(0, min(H - proj.h, br + random.randint(-D - 4, D + 4)))
                    cc = max(0, min(W - proj.w, bc + random.randint(-D - 4, D + 4)))
                else:
                    rr = random.randint(0, max(0, H - proj.h))
                    cc = random.randint(0, max(0, W - proj.w))

                if add_new(child, proj.project_id, rr, cc) is not False:
                    break
            return child

        uid = random.choice(child["active_uids"])
        old = remove_uid(child, uid)
        if old is None:
            return child

        old_b, old_r, old_c, old_type, _, _ = old
        old_proj = city.get_project(old_b)

        if op == "MOVE":
            for _ in range(max_mutation_tries):
                rr = max(0, min(H - old_proj.h, old_r + random.randint(-neighborhood_radius, neighborhood_radius)))
                cc = max(0, min(W - old_proj.w, old_c + random.randint(-neighborhood_radius, neighborhood_radius)))
                if add_new(child, old_b, rr, cc) is not False:
                    return child

            add_new(child, old_b, old_r, old_c)
            return child

        if op == "CHANGE":
            for _ in range(max_mutation_tries):
                if random.random() < 0.70 and old_type == "R" and top_res_projects:
                    proj = random.choice(top_res_projects)
                elif random.random() < 0.50:
                    st = random.choice(utility_types)
                    proj = random.choice(utility_candidates_by_type[st])
                else:
                    proj = random.choice(projects)

                rr = max(0, min(H - proj.h, old_r))
                cc = max(0, min(W - proj.w, old_c))

                if add_new(child, proj.project_id, rr, cc) is not False:
                    return child

            add_new(child, old_b, old_r, old_c)
            return child

        add_new(child, old_b, old_r, old_c)
        return child

    def mutate(ind):
        if random.random() < destroy_repair_rate:
            return destroy_and_repair(ind)
        return light_mutate(ind)

    # =========================================================
    # Evolution loop
    # =========================================================
    best_individual = max(population, key=internal_score)
    best_score = internal_score(best_individual)
    best_generation = 0

    print(f"Initial best internal score: {best_score}")

    for gen in range(1, generations + 1):
        if time.time() - t0 > max_runtime_s * 0.98:
            print("  [INFO] Time budget reached: stopping GA early.")
            break

        ranked = sorted(population, key=internal_score, reverse=True)
        next_population = [clone_individual(ind) for ind in ranked[:elite_size]]

        while len(next_population) < population_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = clone_individual(p1), clone_individual(p2)

            if random.random() < mutation_rate:
                c1 = mutate(c1)
            if random.random() < mutation_rate:
                c2 = mutate(c2)

            next_population.append(c1)
            if len(next_population) < population_size:
                next_population.append(c2)

        population = next_population

        gen_best = max(population, key=internal_score)
        gen_best_score = internal_score(gen_best)

        if gen_best_score > best_score:
            best_individual = clone_individual(gen_best)
            best_score = gen_best_score
            best_generation = gen

        if gen % 5 == 0 or gen == 1:
            avg_score = sum(internal_score(ind) for ind in population) / len(population)
            best_len = len(best_individual["active_uids"])
            print(
                f"  Generation {gen}/{generations} | "
                f"best_internal={gen_best_score} | global_best={best_score} | "
                f"avg={avg_score:.2f} | placements_best={best_len}"
            )

    dt = time.time() - t0
    final_solution = snapshot_solution(best_individual)

    print(f"Final best internal score: {best_score}")
    print(f"Best found on generation: {best_generation}")
    print(f"Genetic Algorithm finished | placements={len(final_solution)} | time={dt:.2f}s")
    print("======== END GENETIC ALGORITHM ========")

    return final_solution
