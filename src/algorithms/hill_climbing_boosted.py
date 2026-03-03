import random
from coordinates import Coordinates

def hill_climbing_boosted(city,
                             max_iterations=5000,
                             patience=300,
                             min_delta=2,
                             use_restart=False,
                             max_restarts=1):

    def run_single_hc():
        H, W = city.H, city.W
        occupied = [[False]*W for _ in range(H)]
        placements = []

        def can_place(b, r, c):
            proj = city.get_project(b)
            top_left = Coordinates(r, c)
            if not proj.fits_in_grid(top_left, H, W):
                return False
            for cell in proj.absolute_hash_cells(top_left):
                if occupied[cell.r][cell.c]:
                    return False
            return True

        def place(b, r, c):
            proj = city.get_project(b)
            for cell in proj.absolute_hash_cells(Coordinates(r, c)):
                occupied[cell.r][cell.c] = True
            placements.append((b, r, c))

        def remove(idx):
            b, r, c = placements[idx]
            proj = city.get_project(b)
            for cell in proj.absolute_hash_cells(Coordinates(r, c)):
                occupied[cell.r][cell.c] = False
            placements.pop(idx)

        # Phase 1: Initial explosion with a structured pattern
        print("-> Initial city explosion...")

        # Find the best residential project (highest capacity relative to size)
        best_res_id = max(
            (pid for pid, proj in enumerate(city.projects) if proj.build_type=="R"),
            key=lambda pid: city.get_project(pid).capacity / max(1, city.get_project(pid).h + city.get_project(pid).w))
        best_res_proj = city.get_project(best_res_id)

        # --- Collect one utility of each type ---
        utility_types_seen = set()
        best_utilities = []
        for pid, proj in enumerate(city.projects):
            if proj.build_type == "U" and proj.service_type not in utility_types_seen:
                utility_types_seen.add(proj.service_type)
                best_utilities.append(pid)

        # Base cell size
        cell_h = best_res_proj.h
        cell_w = best_res_proj.w
        for u_id in best_utilities:
            u_proj = city.get_project(u_id)
            cell_h = max(cell_h, u_proj.h)
            cell_w = max(cell_w, u_proj.w)

        # Fill the map in 3x3 blocks
        block_h = cell_h * 3
        block_w = cell_w * 3
        util_idx = 0
        for r in range(0, H, block_h):
            for c in range(0, W, block_w):
                # Block center: utility
                u_id = best_utilities[util_idx % len(best_utilities)]
                center_r = r + cell_h
                center_c = c + cell_w
                if can_place(u_id, center_r, center_c):
                    place(u_id, center_r, center_c)
                    util_idx += 1
                # Residential around
                for dr in [0, cell_h, cell_h*2]:
                    for dc in [0, cell_w, cell_w*2]:
                        if dr == cell_h and dc == cell_w:
                            continue
                        cand_r = r + dr
                        cand_c = c + dc
                        if can_place(best_res_id, cand_r, cand_c):
                            place(best_res_id, cand_r, cand_c)

        # --- Fill free spaces with extra utilities ---
        step_r_fill = max(1, cell_h//2)
        step_c_fill = max(1, cell_w//2)
        search_radius = city.D + max(cell_h, cell_w)
        for r in range(0, H, step_r_fill):
            for c in range(0, W, step_c_fill):
                if occupied[r][c]:
                    continue
                for u_id in best_utilities:
                    if can_place(u_id, r, c):
                        # Only place if there is nearby residential
                        has_nearby_res = any(
                            city.get_project(b).build_type=="R" and abs(b_r-r)+abs(b_c-c)<=search_radius
                            for b,b_r,b_c in placements
                        )
                        if has_nearby_res:
                            place(u_id, r, c)
                            break

        current_score = city.get_score(placements)
        best_score = current_score
        iterations_without_improvement = 0
        print(f"-> Initial explosion complete! Score: {current_score}, buildings placed: {len(placements)}")

        # Phase 2: Hill Climbing fine-tuning optimization
        print("-> Starting Hill Climbing optimization...")
        for i in range(max_iterations):
            if i % 50 == 0:
                print(f"Iteration {i}, current score: {current_score}, best score: {best_score}")

            # Operations focused on local improvements
            op = random.choices(["MOVE","CHANGE_TYPE","REMOVE","ADD"], weights=[0.35,0.35,0.15,0.15])[0]
            improved = False

            if op=="ADD":
                b = random.randint(0, city.B-1)
                if placements:
                    _, ext_r, ext_c = random.choice(placements)
                    r = max(0, min(H-1, ext_r + random.randint(-4,4)))
                    c = max(0, min(W-1, ext_c + random.randint(-4,4)))
                else:
                    r, c = random.randint(0,H-1), random.randint(0,W-1)
                if can_place(b,r,c):
                    place(b,r,c)
                    new_score = city.get_score(placements)
                    if new_score>=current_score:
                        current_score = new_score
                        improved=True
                    else:
                        remove(len(placements)-1)

            elif op=="REMOVE" and placements:
                idx = random.randint(0,len(placements)-1)
                b,r,c = placements[idx]
                remove(idx)
                new_score = city.get_score(placements)
                if new_score>=current_score:
                    current_score = new_score
                    improved=True
                else:
                    place(b,r,c)

            elif op=="MOVE" and placements:
                idx = random.randint(0,len(placements)-1)
                b, old_r, old_c = placements[idx]
                remove(idx)
                new_r = max(0, min(H-1, old_r+random.randint(-3,3)))
                new_c = max(0, min(W-1, old_c+random.randint(-3,3)))
                if can_place(b,new_r,new_c):
                    place(b,new_r,new_c)
                    new_score = city.get_score(placements)
                    if new_score>=current_score:
                        current_score = new_score
                        improved=True
                    else:
                        remove(len(placements)-1)
                        place(b, old_r, old_c)
                else:
                    place(b, old_r, old_c)

            elif op=="CHANGE_TYPE" and placements:
                idx = random.randint(0,len(placements)-1)
                old_b,r,c = placements[idx]
                remove(idx)
                new_b = random.randint(0,city.B-1)
                if new_b!=old_b and can_place(new_b,r,c):
                    place(new_b,r,c)
                    new_score = city.get_score(placements)
                    if new_score>=current_score:
                        current_score=new_score
                        improved=True
                    else:
                        remove(len(placements)-1)
                        place(old_b,r,c)
                else:
                    place(old_b,r,c)

            if improved:
                if current_score>best_score+min_delta:
                    best_score=current_score
                    iterations_without_improvement=0
                else:
                    iterations_without_improvement+=1
            else:
                iterations_without_improvement+=1

            if iterations_without_improvement>=patience:
                print(f"  No improvement for {patience} iterations, stopping run.")
                break

        return placements, best_score

    # Execution with optional restart
    if not use_restart:
        placements, _ = run_single_hc()
        return placements

    global_best_score=-1
    global_best_solution=[]
    restart=0
    while restart<max_restarts:
        print(f"--- Restart {restart+1} ---")
        placements, score = run_single_hc()
        if score>global_best_score:
            global_best_score = score
            global_best_solution = placements
        restart +=1

    print(f"Best score found: {global_best_score}")
    return global_best_solution