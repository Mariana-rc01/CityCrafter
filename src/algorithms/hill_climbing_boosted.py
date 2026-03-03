import random
from coordinates import Coordinates


def hill_climbing_boosted(city, max_iterations=10000, patience=500,
                          min_delta=2, use_restart=True, max_restarts=2):

    def run_single_hc():
        H, W, D = city.H, city.W, city.D

        occupied = [[False] * W for _ in range(H)]
        placements = []

        # Helpers
        def to_city_format():
            return [(p[0], p[1], p[2]) for p in placements]  # b_id, r, c

        def can_place(proj, r, c):
            if r < 0 or c < 0:
                return False
            if r + proj.h > H or c + proj.w > W:
                return False

            cells = proj.absolute_hash_cells(Coordinates(r, c))
            for cell in cells:
                if occupied[cell.r][cell.c]:
                    return False
            return cells

        def place(b_id, r, c):
            proj = city.get_project(b_id)
            abs_cells = can_place(proj, r, c)
            if abs_cells is False:
                return False

            for cell in abs_cells:
                occupied[cell.r][cell.c] = True

            placements.append([b_id, r, c, abs_cells])
            return True

        def remove(idx):
            b_id, r, c, cells = placements.pop(idx)
            for cell in cells:
                occupied[cell.r][cell.c] = False
            return b_id, r, c, cells

        # Initial State
        print("Begin")
        best_res_id = max(
            (pid for pid, proj in enumerate(city.projects)
             if proj.build_type == "R"),
            key=lambda pid:
            city.get_project(pid).capacity /
            max(1, city.get_project(pid).h * city.get_project(pid).w)
        )

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
                        for dr in range(-D, D + 1):
                            for dc in range(-D, D + 1):
                                ur, uc = r + dr, c + dc
                                if place(u_id, ur, uc):
                                    placed = True
                                    break
                            if placed:
                                break

        current_score = city.get_score(to_city_format())
        best_score = current_score

        print(f"Initial State -> Score: {current_score}, buildings: {len(placements)}")

        # Hill Climbing
        print("Hill Climbing...")
        iterations_without_improvement = 0

        for iteration in range(max_iterations):

            op = random.choices(
                ["MOVE", "CHANGE", "ADD", "REMOVE"],
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0]

            improved = False
            if iteration % 500 == 0:
                print(f"Iteration {iteration}, current_score: {current_score}")

            if op == "ADD":
                b = random.randint(0, city.B - 1)
                r = random.randint(0, H - 1)
                c = random.randint(0, W - 1)

                if place(b, r, c):
                    new_score = city.get_score(to_city_format())
                    if new_score >= current_score:
                        current_score = new_score
                        improved = True
                    else:
                        remove(len(placements) - 1)

            elif placements:

                idx = random.randint(0, len(placements) - 1)
                old = remove(idx)

                if op == "MOVE":
                    new_r = max(0, min(H - 1, old[1] + random.randint(-3, 3)))
                    new_c = max(0, min(W - 1, old[2] + random.randint(-3, 3)))

                    if place(old[0], new_r, new_c):
                        new_score = city.get_score(to_city_format())
                        if new_score >= current_score:
                            current_score = new_score
                            improved = True
                        else:
                            remove(len(placements) - 1)
                            place(old[0], old[1], old[2])
                    else:
                        place(old[0], old[1], old[2])

                elif op == "CHANGE":
                    new_b = random.randint(0, city.B - 1)

                    if place(new_b, old[1], old[2]):
                        new_score = city.get_score(to_city_format())
                        if new_score >= current_score:
                            current_score = new_score
                            improved = True
                        else:
                            remove(len(placements) - 1)
                            place(old[0], old[1], old[2])
                    else:
                        place(old[0], old[1], old[2])

                elif op == "REMOVE":
                    new_score = city.get_score(to_city_format())
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

        return to_city_format(), best_score

    best_solution = []
    best_score = -1

    for r in range(max_restarts if use_restart else 1):
        if use_restart:
            print(f"\n--- Restart {r+1} ---")

        sol, score = run_single_hc()

        print(f"\nRestart {r+1} finished. Score: {score}, buildings placed: {len(sol)}")

        if score > best_score:
            best_solution = sol
            best_score = score

    return best_solution
