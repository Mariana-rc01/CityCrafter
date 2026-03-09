import random, time
from collections import defaultdict
from coordinates import Coordinates
from algorithms.greedy import greedy

#def hill_climbing(city, max_iterations=8000, patience=1500, min_delta=1, use_restart=True, max_restarts=1):
#def hill_climbing(city, max_iterations=4000, patience=800, min_delta=2, use_restart=False, max_restarts=0):
def hill_climbing(city, max_iterations=5000, patience=500, min_delta=2, use_restart=True, max_restarts=1):

    t0 = time.time()
    H, W, D = city.H, city.W, city.D

    print("========== HILL CLIMBING ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")
    print("Start building...")

    # Single HC execution function
    def run_single_hc():
        occupied = set()  # tuples (r, c)
        influence_grid = defaultdict(lambda: defaultdict(int))  # (r,c) -> {service_type: count}
        placements = []  # [b_id, r, c, type, val]

        def get_influence_diamond(r, c, dist):
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    if abs(dr) + abs(dc) > dist:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        yield nr, nc

        def update_influence(proj, r, c, delta):
            if proj.build_type != "U": return
            s_type = proj.service_type
            affected = set()
            for dr, dc in proj.hash_offsets:
                for nr, nc in get_influence_diamond(r + dr, c + dc, D):
                    affected.add((nr, nc))
            for nr, nc in affected:
                influence_grid[(nr, nc)][s_type] += delta
                if influence_grid[(nr, nc)][s_type] <= 0:
                    del influence_grid[(nr, nc)][s_type]
                    if not influence_grid[(nr, nc)]:
                        del influence_grid[(nr, nc)]

        def calculate_score():
            total = 0
            for b_id, r, c, b_type, val in placements:
                if b_type != "R":
                    continue
                proj = city.get_project(b_id)
                accessible = set()
                for dr, dc in proj.hash_offsets:
                    accessible.update(influence_grid.get((r + dr, c + dc), {}).keys())
                total += val * len(accessible)
            return total

        def can_place(proj, r, c):
            if r < 0 or r + proj.h > H or c < 0 or c + proj.w > W:
                return False
            for dr, dc in proj.hash_offsets:
                if (r + dr, c + dc) in occupied:
                    return False
            return True

        def place(b_id, r, c):
            proj = city.get_project(b_id)
            if not can_place(proj, r, c):
                return False
            for dr, dc in proj.hash_offsets:
                occupied.add((r + dr, c + dc))
            val = proj.capacity if proj.build_type == "R" else proj.service_type
            placements.append([b_id, r, c, proj.build_type, val])
            if proj.build_type == "U":
                update_influence(proj, r, c, 1)
            return True

        def remove(idx):
            p = placements.pop(idx)
            b_id, r, c, b_type, val = p
            proj = city.get_project(b_id)
            for dr, dc in proj.hash_offsets:
                occupied.remove((r + dr, c + dc))
            if b_type == "U":
                update_influence(proj, r, c, -1)
            return p

        # Initial Solution
        # Check if city matches requested characteristics:
        print("[MODE] Initial Solution => Running GREEDY Algorithm Seed")

        # Calls greedy algorithm with reasonable time limit
        greedy_placements = greedy(city, max_runtime_s=540)

        # Inject greedy results into Hill Climbing state
        for b_id, r, c in greedy_placements:
            place(b_id, r, c)

        print(f"  -> Greedy Seed placed {len(placements)} buildings.")

        current_score = calculate_score()
        best_score = current_score

        # Hill Climbing
        print("[MODE] Hill climbing fine-tuning")
        iterations_without_improvement = 0
        for i in range(max_iterations):
            if i % 500 == 0:
                print(f"  Iteration {i}/{max_iterations} | placements: {len(placements)} | score={current_score}")

            op = random.choices(["MOVE", "CHANGE", "ADD", "REMOVE"], weights=[0.4,0.3,0.2,0.1])[0]
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
                proj = city.get_project(old[0])

                if op == "MOVE":
                    new_r = max(0, min(H - proj.h, old[1] + random.randint(-3, 3)))
                    new_c = max(0, min(W - proj.w, old[2] + random.randint(-3, 3)))
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
                    new_proj = city.get_project(new_b)
                    if new_proj.build_type == proj.build_type and place(new_b, old[1], old[2]):
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
                print("  [INFO] Patience reached: stopping hill climbing early.")
                break

        return [(p[0], p[1], p[2]) for p in placements], best_score

    # Multi-Restart
    best_solution, best_s = [], -1
    for r in range(max_restarts if use_restart else 1):
        if use_restart:
            print(f"--- Restart {r+1} ---")
        sol, s = run_single_hc()
        if s > best_s:
            best_solution, best_s = sol, s

    dt = time.time() - t0
    print(f"Hill Climbing finished: placements={len(best_solution)} | time={dt:.2f}s")
    print("======== END HILL CLIMBING ========")

    return best_solution
