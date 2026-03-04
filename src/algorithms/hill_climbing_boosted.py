import random
from coordinates import Coordinates

def hill_climbing_boosted(city, max_iterations=10000, patience=500, min_delta=2, use_restart=True, max_restarts=1):

    def run_single_hc():
        H, W, D = city.H, city.W, city.D

        occupied = [[False] * W for _ in range(H)]
        influence_grid = [[{} for _ in range(W)] for _ in range(H)]
        placements = []
        cell_cache = {}

        # --- Helpers ---
        def get_cells(b, r, c):
            key = (b, r, c)
            if key not in cell_cache:
                cell_cache[key] = city.get_project(b).absolute_hash_cells(Coordinates(r, c))
            return cell_cache[key]

        def get_influence_diamond(r, c, dist):
            for dr in range(-dist, dist+1):
                rem = dist - abs(dr)
                for dc in range(-rem, rem+1):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        yield nr, nc

        def update_influence(proj, r, c, delta):
            s_type = proj.service_type
            affected = set()
            for cell in get_cells(proj.project_id, r, c) if hasattr(proj, "project_id") else []:
                for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                    affected.add((nr, nc))
            for nr, nc in affected:
                grid_cell = influence_grid[nr][nc]
                grid_cell[s_type] = grid_cell.get(s_type, 0) + delta
                if grid_cell[s_type] <= 0: grid_cell.pop(s_type, None)

        def calculate_score():
            total = 0
            for p in placements:
                b_id, r, c, b_type, val, cells = p
                if b_type == "R":
                    accessible = set()
                    for cell in cells:
                        accessible.update(influence_grid[cell.r][cell.c].keys())
                    total += val * len(accessible)
            return total

        def can_place(proj, r, c):
            if r < 0 or r + proj.h > H or c < 0 or c + proj.w > W: return False
            abs_cells = proj.absolute_hash_cells(Coordinates(r, c))
            for cell in abs_cells:
                if occupied[cell.r][cell.c]: return False
            return abs_cells

        def place(b_id, r, c):
            proj = city.get_project(b_id)
            abs_cells = can_place(proj, r, c)
            if abs_cells is False:
                return False

            for cell in abs_cells:
                occupied[cell.r][cell.c] = True
            val = proj.capacity if proj.build_type == "R" else proj.service_type
            placements.append([b_id, r, c, proj.build_type, val, abs_cells])

            if proj.build_type == "U":
                update_influence(proj, r, c, 1)
            return True

        def remove(idx):
            p = placements.pop(idx)
            b_id, r, c, b_type, val, cells = p
            proj = city.get_project(b_id)
            for cell in cells:
                occupied[cell.r][cell.c] = False
            if b_type == "U":
                update_influence(proj, r, c, -1)
            return p

        # Initial Solution
        best_res_id = max((pid for pid, proj in enumerate(city.projects) if proj.build_type == "R"),
                          key=lambda pid: city.get_project(pid).capacity / max(1, city.get_project(pid).h + city.get_project(pid).w))
        best_utils = []
        seen = set()
        for pid, proj in enumerate(city.projects):
            if proj.build_type == "U" and proj.service_type not in seen:
                seen.add(proj.service_type)
                best_utils.append(pid)

        res_proj = city.get_project(best_res_id)
        res_h, res_w = res_proj.h, res_proj.w

        for r in range(0, H, res_h):
            for c in range(0, W, res_w):
                if place(best_res_id, r, c):

                    for u_id in best_utils:
                        placed = False
                        u_proj = city.get_project(u_id)

                        for dr in range(-D, D + 1):
                            for dc in range(-D, D + 1):
                                ur, uc = r + dr, c + dc
                                if can_place(u_proj, ur, uc):
                                    place(u_id, ur, uc)
                                    placed = True
                                    break
                            if placed:
                                break

        current_score = calculate_score()
        best_score = current_score
        print(f"Initial Solution -> Score: {current_score}, buildings: {len(placements)}")

        # Hill Climbing
        iterations_without_improvement = 0
        for i in range(max_iterations):
            if i % 500 == 0:
                print(f"  Iteration {i}, current score: {current_score}, best score: {best_score}")
            op = random.choices(["MOVE", "CHANGE", "ADD", "REMOVE"], weights=[0.4, 0.3, 0.2, 0.1])[0]
            improved = False

            if op == "ADD":
                b = random.randint(0, city.B - 1)
                r, c = random.randint(0, H - 1), random.randint(0, W - 1)
                if place(b, r, c):
                    new_score = calculate_score()
                    if new_score >= current_score:
                        current_score = new_score
                        improved = True
                    else:
                        remove(len(placements)-1)

            elif placements:
                idx = random.randint(0, len(placements)-1)
                old = remove(idx)

                if op == "MOVE":
                    new_r = max(0, min(H - 1, old[1] + random.randint(-3, 3)))
                    new_c = max(0, min(W - 1, old[2] + random.randint(-3, 3)))
                    if place(old[0], new_r, new_c):
                        new_score = calculate_score()
                        if new_score >= current_score:
                            current_score = new_score
                            improved = True
                        else:
                            remove(len(placements)-1)
                            place(old[0], old[1], old[2])
                    else:
                        place(old[0], old[1], old[2])

                elif op == "CHANGE":
                    new_b = random.randint(0, city.B - 1)
                    if place(new_b, old[1], old[2]):
                        new_score = calculate_score()
                        if new_score >= current_score:
                            current_score = new_score
                            improved = True
                        else:
                            remove(len(placements)-1)
                            place(old[0], old[1], old[2])
                    else:
                        place(old[0], old[1], old[2])

                elif op == "REMOVE":
                    new_score = calculate_score()
                    if new_score >= current_score:
                        current_score = new_score
                        improved = True
                    else:
                        place(old[0], old[1], old[2])

            if improved:
                if current_score > best_score + min_delta:
                    best_score = current_score
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= patience:
                break

        return [(p[0], p[1], p[2]) for p in placements], best_score

    best_solution, best_s = [], -1
    for r in range(max_restarts if use_restart else 1):
        if use_restart:
            print(f"--- Restart {r+1} ---")
        sol, s = run_single_hc()
        if s > best_s:
            best_solution, best_s = sol, s

    return best_solution
