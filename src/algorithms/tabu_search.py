import random, time
from collections import deque
from coordinates import Coordinates

def tabu_search(city, max_iterations=2500, tabu_tenure=50, neighborhood_size=30, max_runtime_s=540):

    t0 = time.time()
    H, W, D = city.H, city.W, city.D

    occupied = [[False]*W for _ in range(H)]
    influence_grid = [[{} for _ in range(W)] for _ in range(H)]
    residential_at = [[None]*W for _ in range(H)]

    placements = {}       # uid -> [b_id, r, c, b_type, val, cells]
    res_coverage = {}     # uid -> {service_type: count}
    active_uids = []
    uid_to_idx = {}
    next_uid = 0
    current_score = 0

    best_global_score = 0
    best_global_placements = []

    tabu_list = deque(maxlen=tabu_tenure)
    cell_cache = {}

    residential = [p for p in city.projects if p.build_type == "R"]
    utilities = [p for p in city.projects if p.build_type == "U"]
    residential_sorted = sorted(residential, key=lambda p: p.capacity/(max(1, p.h*p.w)), reverse=True)
    top_res = residential_sorted[:40]

    best_utils = []
    seen = set()
    for u in utilities:
        if u.service_type not in seen:
            seen.add(u.service_type)
            best_utils.append(u.project_id)

    print("========== TABU SEARCH OPTIMIZED ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")
    print(f"Selected residential candidates: {len(top_res)} (top_k_res=40)")
    print(f"Selected utility types (max): {len(best_utils)}")
    print("Start building...")

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
        cells = get_cells(proj.project_id, r, c)
        for cell in cells:
            if occupied[cell.r][cell.c]:
                return False
        return cells

    def place_residential(uid, val, cells):
        gain = 0
        cov = {}
        for cell in cells:
            occupied[cell.r][cell.c] = True
            residential_at[cell.r][cell.c] = uid
            for s, count in influence_grid[cell.r][cell.c].items():
                cov[s] = cov.get(s,0)+count
        for s, count in cov.items():
            if count > 0:
                gain += val
        res_coverage[uid] = cov
        return gain

    def remove_residential(uid, val, cells):
        loss = 0
        cov = res_coverage.pop(uid)
        for s,count in cov.items():
            if count>0: loss += val
        for cell in cells:
            occupied[cell.r][cell.c] = False
            residential_at[cell.r][cell.c] = None
        return loss

    def place_utility(s_type, cells):
        gain = 0
        affected = set()
        for cell in cells:
            occupied[cell.r][cell.c] = True
            for nr,nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr,nc))
        for nr,nc in affected:
            influence_grid[nr][nc][s_type] = influence_grid[nr][nc].get(s_type,0)+1
            ruid = residential_at[nr][nc]
            if ruid is not None:
                cov = res_coverage[ruid]
                cov[s_type] = cov.get(s_type,0)+1
                if cov[s_type]==1:
                    gain += placements[ruid][4]
        return gain

    def remove_utility(s_type, cells):
        loss = 0
        affected = set()
        for cell in cells:
            occupied[cell.r][cell.c] = False
            for nr,nc in get_influence_diamond(cell.r, cell.c, D):
                affected.add((nr,nc))
        for nr,nc in affected:
            ig = influence_grid[nr][nc]
            ig[s_type]-=1
            if ig[s_type]==0: del ig[s_type]
            ruid = residential_at[nr][nc]
            if ruid is not None:
                cov = res_coverage[ruid]
                cov[s_type]-=1
                if cov[s_type]==0: loss += placements[ruid][4]
        return loss

    def place(b, r, c):
        nonlocal current_score, next_uid
        proj = city.get_project(b)
        cells = can_place(proj, r, c)
        if not cells: return False, None
        uid = next_uid
        next_uid += 1
        val = proj.capacity if proj.build_type=="R" else proj.service_type
        gain = place_residential(uid,val,cells) if proj.build_type=="R" else place_utility(val,cells)
        current_score += gain
        placements[uid] = [b,r,c,proj.build_type,val,cells]
        add_active_uid(uid)
        return True, uid

    def remove(uid):
        nonlocal current_score
        p = placements.pop(uid)
        b,r,c,b_type,val,cells = p
        loss = remove_residential(uid,val,cells) if b_type=="R" else remove_utility(val,cells)
        current_score -= loss
        remove_active_uid(uid)
        return p

    def insert_with_uid(uid,b,r,c,b_type,val,cells):
        nonlocal current_score
        gain = place_residential(uid,val,cells) if b_type=="R" else place_utility(val,cells)
        current_score += gain
        placements[uid] = [b,r,c,b_type,val,cells]
        add_active_uid(uid)

    # Initial Solution
    t_init_start = time.time()
    print("[MODE] Initial Solution => Generating guided blocks")

    placed_res = 0
    placed_util = 0
    for r in range(0,H,3):
        if r % 60 == 0:
            print(f"  Scan row {r}/{H} | placements: {len(placements)} | R={placed_res} U={placed_util}")

        for c in range(0,W,3):
            for uid in best_utils:
                if place(uid,r,c)[0]:
                    placed_util += 1
            for res in top_res:
                if place(res.project_id,r+1,c+1)[0]:
                    placed_res += 1

    best_global_score = current_score
    best_global_placements = [(p[0],p[1],p[2]) for p in placements.values()]

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
            op = random.choice(["MOVE","CHANGE","ADD","REMOVE"])
            sig = None
            action_data = None
            n_score = -1

            if op=="ADD":
                b = random.randint(0,city.B-1)
                r,c = random.randint(0,H-1), random.randint(0,W-1)
                success, uid = place(b,r,c)
                if success:
                    n_score = current_score
                    sig = f"ADD_{b}_{r}_{c}"
                    action_data = ("ADD",b,r,c)
                    remove(uid)
            elif active_uids:
                target_uid = random.choice(active_uids)
                old_data = remove(target_uid)
                old_b,old_r,old_c = old_data[0],old_data[1],old_data[2]

                if op=="MOVE":
                    nr = max(0,min(H-1,old_r+random.randint(-3,3)))
                    nc = max(0,min(W-1,old_c+random.randint(-3,3)))
                    success,new_uid = place(old_b,nr,nc)
                    if success:
                        n_score = current_score
                        sig = f"MOVE_{old_b}_TO_{old_r}_{old_c}"
                        action_data = ("MOVE",target_uid,old_b,old_r,old_c,nr,nc)
                        remove(new_uid)
                elif op=="CHANGE":
                    new_b = random.randint(0,city.B-1)
                    success,new_uid = place(new_b,old_r,old_c)
                    if success:
                        n_score = current_score
                        sig = f"CHANGE_{old_r}_{old_c}_TO_{new_b}"
                        action_data = ("CHANGE",target_uid,old_b,old_r,old_c,new_b)
                        remove(new_uid)
                elif op=="REMOVE":
                    n_score = current_score
                    sig = f"REMOVE_{old_b}_{old_r}_{old_c}"
                    action_data = ("REMOVE",target_uid,old_b,old_r,old_c)
                insert_with_uid(target_uid,*old_data)

            if action_data:
                is_tabu = sig in tabu_list
                if n_score > best_global_score:
                    is_tabu=False
                if not is_tabu and n_score > best_neighbor_score:
                    best_neighbor_score = n_score
                    best_action = action_data
                    best_signature = sig

        # Apply best action
        if best_action:
            op_type = best_action[0]
            if op_type=="ADD": _,b,r,c = best_action; place(b,r,c)
            elif op_type=="MOVE": _,uid,b,r,c,nr,nc = best_action; remove(uid); place(b,nr,nc)
            elif op_type=="CHANGE": _,uid,b,r,c,new_b = best_action; remove(uid); place(new_b,r,c)
            elif op_type=="REMOVE": _,uid,b,r,c = best_action; remove(uid)
            if best_signature: tabu_list.append(best_signature)
            if current_score>best_global_score:
                best_global_score = current_score
                best_global_placements = [(p[0],p[1],p[2]) for p in placements.values()]

        if it % 500 == 0:
            print(f"  Iteration {it}/{max_iterations} | placements: {len(placements)} | score={current_score}")

    dt = time.time() - t0
    print(f"Tabu Search finished: placements={len(best_global_placements)} | time={dt:.2f}s")
    print("======== END TABU SEARCH ========")

    return best_global_placements
