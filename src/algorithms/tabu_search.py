import random, time
from collections import deque
from coordinates import Coordinates
from algorithms.greedy import greedy

#def tabu_search(city, max_iterations=3500, tabu_tenure=40, neighborhood_size=40, max_runtime_s=1000):
#def tabu_search(city, max_iterations=3000, tabu_tenure=25, neighborhood_size=25, max_runtime_s=1000):
def tabu_search(city, max_iterations=2500, tabu_tenure=50, neighborhood_size=30, max_runtime_s=1000):

    t0 = time.time()
    H, W, D = city.H, city.W, city.D

    occupied = set()           # Stores only occupied tuples (r, c)
    influence_grid = {}        # Maps (r, c) -> {service_type: count} only where there is influence
    residential_at = {}        # Maps (r, c) -> uid

    placements = {}
    res_coverage = {}
    active_uids = []
    uid_to_idx = {}
    next_uid = 0
    current_score = 0

    best_global_score = 0
    best_global_placements = []

    tabu_list = deque(maxlen=tabu_tenure)

    print("========== TABU SEARCH OPTIMIZED ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")
    print("Start building...")

    def get_influence_diamond(r, c, dist):
        for dr in range(-dist, dist+1):
            rem = dist - abs(dr)
            for dc in range(-rem, rem+1):
                nr, nc = r+dr, c+dc
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
        if r < 0 or r + proj.h > H or c < 0 or c + proj.w > W:
            return False
        for dr, dc in proj.hash_offsets:
            if (r + dr, c + dc) in occupied:
                return False
        return True

    def place_residential(uid, b_id, r, c, val):
        gain = 0
        cov = {}
        proj = city.get_project(b_id)
        for dr, dc in proj.hash_offsets:
            cr, cc = r + dr, c + dc
            occupied.add((cr, cc))
            residential_at[(cr, cc)] = uid
            # Check if this cell receives influence
            if (cr, cc) in influence_grid:
                for s, count in influence_grid[(cr, cc)].items():
                    cov[s] = cov.get(s, 0) + count
        for s, count in cov.items():
            if count > 0:
                gain += val
        res_coverage[uid] = cov
        return gain

    def remove_residential(uid, b_id, r, c, val):
        loss = 0
        cov = res_coverage.pop(uid)
        for s, count in cov.items():
            if count > 0: loss += val
        proj = city.get_project(b_id)
        for dr, dc in proj.hash_offsets:
            cr, cc = r + dr, c + dc
            occupied.remove((cr, cc))
            del residential_at[(cr, cc)]
        return loss

    def place_utility(b_id, r, c, s_type):
        gain = 0
        affected = set()
        proj = city.get_project(b_id)
        for dr, dc in proj.hash_offsets:
            cr, cc = r + dr, c + dc
            occupied.add((cr, cc))
            for nr, nc in get_influence_diamond(cr, cc, D):
                affected.add((nr, nc))
        for nr, nc in affected:
            ig = influence_grid.setdefault((nr, nc), {})
            ig[s_type] = ig.get(s_type, 0) + 1
            ruid = residential_at.get((nr, nc))
            if ruid is not None:
                cov = res_coverage[ruid]
                cov[s_type] = cov.get(s_type, 0) + 1
                if cov[s_type] == 1:
                    gain += placements[ruid][4]
        return gain

    def remove_utility(b_id, r, c, s_type):
        loss = 0
        affected = set()
        proj = city.get_project(b_id)
        for dr, dc in proj.hash_offsets:
            cr, cc = r + dr, c + dc
            occupied.remove((cr, cc))
            for nr, nc in get_influence_diamond(cr, cc, D):
                affected.add((nr, nc))
        for nr, nc in affected:
            ig = influence_grid[(nr, nc)]
            ig[s_type] -= 1
            if ig[s_type] == 0:
                del ig[s_type]
                if not ig:
                    del influence_grid[(nr, nc)]
            ruid = residential_at.get((nr, nc))
            if ruid is not None:
                cov = res_coverage[ruid]
                cov[s_type] -= 1
                if cov[s_type] == 0:
                    loss += placements[ruid][4]
        return loss

    def place(b, r, c):
        nonlocal current_score, next_uid
        proj = city.get_project(b)
        if not can_place(proj, r, c):
            return False, None

        uid = next_uid
        next_uid += 1
        val = proj.capacity if proj.build_type == "R" else proj.service_type

        if proj.build_type == "R":
            gain = place_residential(uid, b, r, c, val)
        else:
            gain = place_utility(b, r, c, val)

        current_score += gain
        placements[uid] = [b, r, c, proj.build_type, val]
        add_active_uid(uid)
        return True, uid

    def remove(uid):
        nonlocal current_score
        p = placements.pop(uid)
        b, r, c, b_type, val = p
        if b_type == "R":
            loss = remove_residential(uid, b, r, c, val)
        else:
            loss = remove_utility(b, r, c, val)
        current_score -= loss
        remove_active_uid(uid)
        return p

    def insert_with_uid(uid, b, r, c, b_type, val):
        nonlocal current_score
        if b_type == "R":
            gain = place_residential(uid, b, r, c, val)
        else:
            gain = place_utility(b, r, c, val)
        current_score += gain
        placements[uid] = [b, r, c, b_type, val]
        add_active_uid(uid)

    # Initial Solution
    t_init_start = time.time()

    print("[MODE] Initial Solution => Running GREEDY Algorithm Seed")

    greedy_placements = greedy(city, max_runtime_s=270)
    for b_id, r, c in greedy_placements:
        place(b_id, r, c)

    print(f"  -> Greedy Seed placed {len(placements)} buildings.")

    best_global_score = current_score
    best_global_placements = [(p[0], p[1], p[2]) for p in placements.values()]

    t_init_end = time.time()
    print(f"  -> Initial Phase finished: placements={len(placements)} | score={current_score} | time={t_init_end - t_init_start:.2f}s")


    # Tabu Search
    print("[MODE] Tabu Search fine-tuning")
    for it in range(max_iterations):
        if time.time()-t0 > max_runtime_s*0.98:
            print("  [INFO] Time budget reached: stopping tabu search early.")
            break

        best_neighbor_score = -1
        best_action = None
        best_signature = None

        for _ in range(neighborhood_size):
            op = random.choice(["MOVE", "CHANGE", "ADD", "REMOVE"])
            sig = None
            action_data = None
            n_score = -1

            if op == "ADD":
                b = random.randint(0, city.B-1)
                r, c = random.randint(0, H-1), random.randint(0, W-1)
                success, uid = place(b, r, c)
                if success:
                    n_score = current_score
                    sig = f"ADD_{b}_{r}_{c}"
                    action_data = ("ADD", b, r, c)
                    remove(uid)
            elif active_uids:
                target_uid = random.choice(active_uids)
                old_data = remove(target_uid)
                old_b, old_r, old_c, old_type, old_val = old_data

                if op == "MOVE":
                    nr = max(0, min(H-1, old_r + random.randint(-3, 3)))
                    nc = max(0, min(W-1, old_c + random.randint(-3, 3)))
                    success, new_uid = place(old_b, nr, nc)
                    if success:
                        n_score = current_score
                        sig = f"MOVE_{old_b}_TO_{old_r}_{old_c}"
                        action_data = ("MOVE", target_uid, old_b, old_r, old_c, nr, nc)
                        remove(new_uid)
                elif op == "CHANGE":
                    new_b = random.randint(0, city.B-1)
                    success, new_uid = place(new_b, old_r, old_c)
                    if success:
                        n_score = current_score
                        sig = f"CHANGE_{old_r}_{old_c}_TO_{new_b}"
                        action_data = ("CHANGE", target_uid, old_b, old_r, old_c, new_b)
                        remove(new_uid)
                elif op == "REMOVE":
                    n_score = current_score
                    sig = f"REMOVE_{old_b}_{old_r}_{old_c}"
                    action_data = ("REMOVE", target_uid, old_b, old_r, old_c)

                insert_with_uid(target_uid, *old_data)

            if action_data:
                is_tabu = sig in tabu_list
                if n_score > best_global_score:
                    is_tabu = False
                if not is_tabu and n_score > best_neighbor_score:
                    best_neighbor_score = n_score
                    best_action = action_data
                    best_signature = sig

        if best_action:
            op_type = best_action[0]
            if op_type == "ADD":
                _, b, r, c = best_action; place(b, r, c)
            elif op_type == "MOVE":
                _, uid, b, r, c, nr, nc = best_action;
                remove(uid);
                place(b, nr, nc)
            elif op_type == "CHANGE":
                _, uid, b, r, c, new_b = best_action;
                remove(uid);
                place(new_b, r, c)
            elif op_type == "REMOVE":
                _, uid, b, r, c = best_action;
                remove(uid)

            if best_signature: tabu_list.append(best_signature)

            if current_score > best_global_score:
                best_global_score = current_score
                best_global_placements = [(p[0], p[1], p[2]) for p in placements.values()]

        if it % 500 == 0:
            print(f"  Iteration {it}/{max_iterations} | placements: {len(placements)} | score={current_score}")

    dt = time.time() - t0
    print(f"Tabu Search finished: placements={len(best_global_placements)} | time={dt:.2f}s")
    print("======== END TABU SEARCH ========")

    return best_global_placements
