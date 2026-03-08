import random
import time
from coordinates import Coordinates
from algorithms.greedy import greedy

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
    print(f"Population size: {population_size} | Generations: {generations}")
    print("Start evolving...")

    cell_cache = {}

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
        r0 = random.randint(0, max(0, H - h_span))
        r1 = min(H - 1, r0 + random.randint(max(5, h_span // 2), h_span))
        c0 = random.randint(0, max(0, W - w_span))
        c1 = min(W - 1, c0 + random.randint(max(5, w_span // 2), w_span))
        return r0, c0, r1, c1

    # =========================================================
    # 1. SPARSE GRIDS & FAST CLONING (Otimização Crítica)
    # =========================================================
    def make_empty_individual():
        return {
            "occupied": set(),         # Agora é um Set (Sparse)
            "influence_grid": {},      # Agora é um Dict (Sparse)
            "residential_at": {},      # Agora é um Dict (Sparse)
            "placements": {},      
            "res_coverage": {},    
            "active_uids": [],
            "uid_to_idx": {},
            "next_uid": 0,
            "score": 0,
        }

    def clone_individual(ind):
        # CÓPIA NATIVA INSTANTÂNEA em vez de reconstruir bloco a bloco!
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
            if (cell.r, cell.c) in ind["occupied"]:
                return False
        return cells

    def place_residential(ind, uid, capacity, cells):
        gain = 0
        cov = {}

        for cell in cells:
            ind["occupied"].add((cell.r, cell.c))
            ind["residential_at"][(cell.r, cell.c)] = uid
            if (cell.r, cell.c) in ind["influence_grid"]:
                for s, count in ind["influence_grid"][(cell.r, cell.c)].items():
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
            ind["occupied"].remove((cell.r, cell.c))
            del ind["residential_at"][(cell.r, cell.c)]
        ind["score"] -= loss

    def place_utility(ind, uid, service_type, cells):
        gain = 0
        affected = set()
        for cell in cells:
            ind["occupied"].add((cell.r, cell.c))
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr, nc))
        for nr, nc in affected:
            grid_cell = ind["influence_grid"].setdefault((nr, nc), {})
            grid_cell[service_type] = grid_cell.get(service_type, 0) + 1
            res_uid = ind["residential_at"].get((nr, nc))
            if res_uid is not None:
                cov = ind["res_coverage"][res_uid]
                cov[service_type] = cov.get(service_type, 0) + 1
                if cov[service_type] == 1:
                    gain += ind["placements"][res_uid][4]
        ind["score"] += gain

    def remove_utility(ind, uid, service_type, cells):
        loss = 0
        affected = set()
        for cell in cells:
            ind["occupied"].remove((cell.r, cell.c))
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr, nc))
        for nr, nc in affected:
            grid_cell = ind["influence_grid"][(nr, nc)]
            grid_cell[service_type] -= 1
            if grid_cell[service_type] == 0:
                del grid_cell[service_type]
                if not grid_cell:
                    del ind["influence_grid"][(nr, nc)]
            res_uid = ind["residential_at"].get((nr, nc))
            if res_uid is not None:
                cov = ind["res_coverage"][res_uid]
                cov[service_type] -= 1
                if cov[service_type] == 0:
                    loss += ind["placements"][res_uid][4]
        ind["score"] -= loss

    def add_new(ind, b_id, r, c):
        proj = city.get_project(b_id)
        cells = can_place(ind, proj, r, c)
        if cells is False: return False
        uid = ind["next_uid"]
        ind["next_uid"] += 1
        val = proj.capacity if proj.build_type == "R" else proj.service_type
        if proj.build_type == "R": place_residential(ind, uid, val, cells)
        else: place_utility(ind, uid, val, cells)
        ind["placements"][uid] = [b_id, r, c, proj.build_type, val, cells]
        add_active_uid(ind, uid)
        return uid

    def remove_uid(ind, uid):
        if uid not in ind["placements"]: return None
        b_id, r, c, b_type, val, cells = ind["placements"].pop(uid)
        if b_type == "R": remove_residential(ind, uid, val, cells)
        else: remove_utility(ind, uid, val, cells)
        remove_active_uid(ind, uid)
        return [b_id, r, c, b_type, val, cells]

    # =========================================================
    # População e Helpers
    # =========================================================
    def build_from_solution(solution):
        ind = make_empty_individual()
        for b_id, r, c in solution: add_new(ind, b_id, r, c)
        return ind

    def greedy_seed():
        return build_from_solution(greedy(city))

    def get_region_residential_uids(ind, r0, c0, r1, c1):
        out = []
        for uid in ind["active_uids"]:
            _, _, _, b_type, _, cells = ind["placements"][uid]
            if b_type == "R" and rect_intersects_cells(r0, c0, r1, c1, cells):
                out.append(uid)
        return out

    def estimate_utility_gain(ind, proj, r, c, region_res_uids):
        cells = can_place(ind, proj, r, c)
        if cells is False: return -1, None
        affected_res = {}
        st = proj.service_type
        for cell in cells:
            for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                res_uid = ind["residential_at"].get((nr, nc))
                if res_uid is not None and res_uid in region_res_uids:
                    affected_res[res_uid] = True
        gain = 0
        for res_uid in affected_res:
            if ind["res_coverage"].get(res_uid, {}).get(st, 0) == 0:
                gain += ind["placements"][res_uid][4]
        return gain, cells

    def guided_random_fill_region(ind, r0, c0, r1, c1, attempts_scale=1.0):
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        util_rounds = max(8, min(40, int(area / 800) + int(8 * attempts_scale)))
        res_attempts = max(80, min(1200, int(area / 20) + int(80 * attempts_scale)))

        for _ in range(util_rounds):
            region_res_uids = get_region_residential_uids(ind, r0, c0, r1, c1)
            best_choice = None
            best_gain = -1
            for st in random.sample(utility_types, min(len(utility_types), 3)):
                for proj in utility_candidates_by_type[st]:
                    for _ in range(10):
                        rr = random.randint(max(0, r0 - D - 3), min(H - proj.h, r1 + D + 3))
                        cc = random.randint(max(0, c0 - D - 3), min(W - proj.w, c1 + D + 3))
                        gain, _ = estimate_utility_gain(ind, proj, rr, cc, region_res_uids)
                        if gain > best_gain:
                            best_gain, best_choice = gain, (proj.project_id, rr, cc)
            if best_choice and best_gain >= 0:
                add_new(ind, *best_choice)

        for _ in range(res_attempts):
            proj = random.choice(top_res_projects)
            rr = random.randint(max(0, r0 - D), min(H - proj.h, r1 + D))
            cc = random.randint(max(0, c0 - D), min(W - proj.w, c1 + D))
            add_new(ind, proj.project_id, rr, cc)

    # =========================================================
    # Operadores Evolutivos
    # =========================================================
    def destroy_and_repair(ind):
        child = clone_individual(ind)
        r0, c0, r1, c1 = random_window()
        to_remove = [uid for uid in child["active_uids"] if placement_intersects_rect(child, uid, r0, c0, r1, c1)]
        for uid in to_remove: remove_uid(child, uid)
        guided_random_fill_region(child, r0, c0, r1, c1, attempts_scale=1.0)
        return child

    def crossover(parent1, parent2):
        r0, c0, r1, c1 = random_window()
        child1, child2 = make_empty_individual(), make_empty_individual()

        for uid in parent1["active_uids"]:
            b_id, r, c, _, _, cells = parent1["placements"][uid]
            if rect_intersects_cells(r0, c0, r1, c1, cells): add_new(child1, b_id, r, c)
            else: add_new(child2, b_id, r, c)

        for uid in parent2["active_uids"]:
            b_id, r, c, _, _, cells = parent2["placements"][uid]
            if rect_intersects_cells(r0, c0, r1, c1, cells): add_new(child2, b_id, r, c)
            else: add_new(child1, b_id, r, c)

        # Repara as "Cicatrizes" da fronteira do corte!
        guided_random_fill_region(child1, r0-D, c0-D, r1+D, c1+D, attempts_scale=0.5)
        guided_random_fill_region(child2, r0-D, c0-D, r1+D, c1+D, attempts_scale=0.5)
        return child1, child2

    def memetic_local_search(ind, steps=15):
        """Pequeno Hill Climbing após mutação para refinar espaços vazios."""
        for _ in range(steps):
            if not ind["active_uids"]: break
            op = random.choices(["ADD", "MOVE"], weights=[0.4, 0.6])[0]

            if op == "ADD":
                proj = random.choice(top_res_projects if random.random() < 0.7 else utility_candidates_by_type[random.choice(utility_types)])
                r, c = random.randint(0, H - proj.h), random.randint(0, W - proj.w)
                add_new(ind, proj.project_id, r, c)
            elif op == "MOVE":
                uid = random.choice(ind["active_uids"])
                old = remove_uid(ind, uid)
                old_b, old_r, old_c = old[0], old[1], old[2]
                proj = city.get_project(old_b)
                nr = max(0, min(H - proj.h, old_r + random.randint(-2, 2)))
                nc = max(0, min(W - proj.w, old_c + random.randint(-2, 2)))

                prev_score = ind["score"]
                if add_new(ind, old_b, nr, nc) is not False:
                    if ind["score"] < prev_score:
                        remove_uid(ind, ind["next_uid"] - 1)
                        add_new(ind, old_b, old_r, old_c) # Reverte se for pior
                else:
                    add_new(ind, old_b, old_r, old_c)
        return ind

    # =========================================================
    # Ciclo Principal
    # =========================================================
    population = [greedy_seed()]
    while len(population) < population_size:
        pop_ind = destroy_and_repair(population[0])
        population.append(memetic_local_search(pop_ind, steps=50))

    best_individual = max(population, key=lambda x: x["score"])
    best_score = best_individual["score"]
    best_generation = 0

    for gen in range(1, generations + 1):
        if time.time() - t0 > max_runtime_s * 0.98:
            break

        ranked = sorted(population, key=lambda x: x["score"], reverse=True)
        next_population = [clone_individual(ind) for ind in ranked[:elite_size]]

        while len(next_population) < population_size:
            # Torneio
            p1 = max(random.sample(population, tournament_size), key=lambda x: x["score"])
            p2 = max(random.sample(population, tournament_size), key=lambda x: x["score"])

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = clone_individual(p1), clone_individual(p2)

            # Mutação e Memetic Search
            if random.random() < destroy_repair_rate:
                c1 = destroy_and_repair(c1)
            c1 = memetic_local_search(c1)

            next_population.append(c1)

            if len(next_population) < population_size:
                if random.random() < destroy_repair_rate:
                    c2 = destroy_and_repair(c2)
                c2 = memetic_local_search(c2)
                next_population.append(c2)

        population = next_population

        gen_best = max(population, key=lambda x: x["score"])
        if gen_best["score"] > best_score:
            best_individual = clone_individual(gen_best)
            best_score = gen_best["score"]
            best_generation = gen

        if gen % 5 == 0 or gen == 1:
            avg_score = sum(ind["score"] for ind in population) / len(population)
            print(f"  Generation {gen}/{generations} | global_best={best_score} | avg={avg_score:.2f}")

    dt = time.time() - t0
    final_solution = [(b_id, r, c) for b_id, r, c, _, _, _ in best_individual["placements"].values()]

    print(f"Final best internal score: {best_score}")
    print(f"Best found on generation: {best_generation}")
    print(f"Genetic Algorithm finished | placements={len(final_solution)} | time={dt:.2f}s")
    print("======== END GENETIC ALGORITHM ========")

    return final_solution
