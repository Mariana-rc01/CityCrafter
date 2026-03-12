import random
import time
from coordinates import Coordinates

def greedy(city,
           seed=0,
           max_runtime_s=540,
           top_k_res=40,
           max_utility_types=8,
           hub_step=None,
           max_hubs=None,
           max_candidates_per_hub=900,
           top_local_res=5):
    """
    Greedy algorithm for urban planning with adaptive strategies.

    Uses different strategies based on the influence distance D:
    - MODE A (D <= 5): Dense-block sweep with local residential placements
    - MODE B (D > 5): Hub-based strategy with spatial binning

    Args:
        city: City instance containing grid, projects and constraints
        seed: Random seed for reproducibility
        max_runtime_s: Maximum runtime in seconds
        top_k_res: Top K residential projects to consider
        max_utility_types: Maximum number of utility types to select
        hub_step: Step size for hub grid (auto-calculated if None)
        max_hubs: Maximum number of hubs to try (auto-calculated if None)
        max_candidates_per_hub: Maximum candidate positions per hub
        top_local_res: Top residential projects for dense mode

    Returns:
        List of tuples (building_id, row, col) representing the solution
    """
    random.seed(seed)
    t0 = time.time()

    H, W, D = city.H, city.W, city.D
    projects = city.projects

    # ---- Helpers: placement + overlaps ----
    occupied = [[False] * W for _ in range(H)]
    placements = []  # list of (project_id, r, c)

    def can_place(pid, r, c):
        """Check if a project can be placed at position (r,c) without overlaps."""
        proj = city.get_project(pid)
        tl = Coordinates(r, c)
        if not proj.fits_in_grid(tl, H, W):
            return False
        for cell in proj.absolute_hash_cells(tl):
            if occupied[cell.r][cell.c]:
                return False
        return True

    def do_place(pid, r, c):
        """Place a project at position (r,c) and mark cells as occupied."""
        proj = city.get_project(pid)
        tl = Coordinates(r, c)
        for cell in proj.absolute_hash_cells(tl):
            occupied[cell.r][cell.c] = True
        placements.append((pid, r, c))

    # ---- Split projects ----
    residential = [p for p in projects if p.build_type == "R"]
    utilities = [p for p in projects if p.build_type == "U"]

    def res_quality(p):
        """Calculate quality metric for residential projects (capacity vs footprint)."""
        # dense capacity is generally good
        hcnt = max(1, len(p.hash_offsets))
        return p.capacity / (hcnt ** 0.85)  # small penalty for large projects

    residential_sorted = sorted(residential, key=res_quality, reverse=True)
    top_res = residential_sorted[:top_k_res]

    # Pick smallest utility per service type
    best_utility_by_type = {}
    for u in utilities:
        st = u.service_type
        cur = best_utility_by_type.get(st)
        if cur is None:
            best_utility_by_type[st] = u
        else:
            a = (len(u.hash_offsets), u.h * u.w)
            b = (len(cur.hash_offsets), cur.h * cur.w)
            if a < b:
                best_utility_by_type[st] = u

    utility_types = sorted(
        best_utility_by_type.keys(),
        key=lambda st: (
            len(best_utility_by_type[st].hash_offsets),
            best_utility_by_type[st].h * best_utility_by_type[st].w
        )
    )
    utility_types = utility_types[:max_utility_types]

    print("========== GREEDY ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")
    print(f"Selected residential candidates: {len(top_res)} (top_k_res={top_k_res})")
    print(f"Selected utility types (max): {len(utility_types)} (max_utility_types={max_utility_types})")
    print("Start building...")

    # ---- MODE A: D <= 5 ----
    if D <= 5:
        print("[MODE] D <= 5 => Dense-block sweep (greedy2-style + top local residentials)")

        # Top residential projects to test in each block
        best_res_candidates = sorted(
            residential,
            key=lambda p: p.capacity / max(1, p.h + p.w),
            reverse=True
        )[:top_local_res]

        if not best_res_candidates:
            print("No residential projects available.")
            print("======== END GREEDY ========")
            return placements

        # Uses the best one only to define the sweep grid
        base_res = best_res_candidates[0]
        res_h, res_w = base_res.h, base_res.w

        # One utility per type
        best_utils = [best_utility_by_type[st].project_id for st in utility_types]

        placed_res = 0
        placed_util = 0

        # Block sweep
        for r in range(0, H, res_h):
            if time.time() - t0 > max_runtime_s * 0.98:
                print("  [INFO] Time budget: stopping dense sweep early.")
                break

            if r % 60 == 0:
                print(f"  Scan row {r}/{H} | placements: {len(placements)} | R={placed_res} U={placed_util}")

            for c in range(0, W, res_w):
                if time.time() - t0 > max_runtime_s * 0.98:
                    break

                best_pid = None
                best_score = -1
                best_pos = None

                # Tests top 3/top 5 residential projects in this block
                for proj_r in best_res_candidates:
                    rr = min(r, H - proj_r.h)
                    cc = min(c, W - proj_r.w)

                    if rr < 0 or cc < 0:
                        continue
                    if not can_place(proj_r.project_id, rr, cc):
                        continue

                    score = proj_r.capacity / max(1, proj_r.h + proj_r.w)

                    if score > best_score:
                        best_score = score
                        best_pid = proj_r.project_id
                        best_pos = (rr, cc)

                if best_pid is None:
                    continue

                # Places the best residential found for this block
                do_place(best_pid, best_pos[0], best_pos[1])
                placed_res += 1

                placed_r, placed_c = best_pos

                # Tries to place one utility of each type around the residential
                for u_id in best_utils:
                    if time.time() - t0 > max_runtime_s * 0.98:
                        break

                    placed = False
                    for dr in range(-D, D + 1):
                        for dc in range(-D, D + 1):
                            if abs(dr) + abs(dc) > D:
                                continue

                            ur, uc = placed_r + dr, placed_c + dc

                            if can_place(u_id, ur, uc):
                                do_place(u_id, ur, uc)
                                placed_util += 1
                                placed = True
                                break
                        if placed:
                            break

        dt = time.time() - t0
        print(f"Greedy finished (MODE A): placements={len(placements)} | time={dt:.2f}s")
        print("======== END GREEDY ========")
        return placements

    # ---- MODE B: D > 2  (hubs + fill, with spatial bins) ----
    print("[MODE] D > 2 => Hub strategy + spatial bins")

    # ---- bbox helpers ----
    def project_hash_bbox(proj):
        """Calculate the bounding box of a project's hash offsets."""
        drs = [dr for dr, _ in proj.hash_offsets]
        dcs = [dc for _, dc in proj.hash_offsets]
        return min(drs), max(drs), min(dcs), max(dcs)

    proj_bbox = {}
    for p in projects:
        proj_bbox[p.project_id] = project_hash_bbox(p)

    def abs_hash_bbox(pid, r, c):
        """Get absolute bounding box for a project placed at position (r,c)."""
        min_dr, max_dr, min_dc, max_dc = proj_bbox[pid]
        return (r + min_dr, r + max_dr, c + min_dc, c + max_dc)

    def rect_manhattan_dist(a, b):
        """Calculate Manhattan distance between two rectangles."""
        a_r0, a_r1, a_c0, a_c1 = a
        b_r0, b_r1, b_c0, b_c1 = b

        if a_r1 < b_r0:
            dr = b_r0 - a_r1
        elif b_r1 < a_r0:
            dr = a_r0 - b_r1
        else:
            dr = 0

        if a_c1 < b_c0:
            dc = b_c0 - a_c1
        elif b_c1 < a_c0:
            dc = a_c0 - b_c1
        else:
            dc = 0

        return dr + dc

    def min_dist_hash_sets(cells_a, cells_b, cutoff):
        """Calculate minimum Manhattan distance between two sets of cells with early cutoff."""
        if not cells_a or not cells_b:
            return 10**9
        if len(cells_a) > len(cells_b):
            cells_a, cells_b = cells_b, cells_a

        best = 10**9
        for a in cells_a:
            ar, ac = a.r, a.c
            for b in cells_b:
                d = abs(ar - b.r) + abs(ac - b.c)
                if d < best:
                    best = d
                    if best <= cutoff:
                        return best
        return best

    # ---- utility storage with spatial bins ----
    utility_places = []  # dicts: {pid,r,c,stype,hash_cells,bbox}
    bin_size = max(10, D + 8)

    def bin_key_from_bbox(b):
        """Get spatial bin key from bounding box center."""
        r0, r1, c0, c1 = b
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        return (cr // bin_size, cc // bin_size)

    bins = {}  # (br, bc) -> list of utility indices

    def register_utility(pid, r, c, st):
        """Register a placed utility in the spatial index for fast lookups."""
        proj = city.get_project(pid)
        tl = Coordinates(r, c)
        cells = proj.absolute_hash_cells(tl)
        bbox = abs_hash_bbox(pid, r, c)
        idx = len(utility_places)
        utility_places.append({"pid": pid, "r": r, "c": c, "stype": st, "hash_cells": cells, "bbox": bbox})
        bk = bin_key_from_bbox(bbox)
        bins.setdefault(bk, []).append(idx)

    def nearby_utility_indices_for_bbox(bbox):
        """Get utility indices in nearby spatial bins for a given bounding box."""
        r0, r1, c0, c1 = bbox
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        br = cr // bin_size
        bc = cc // bin_size

        # search a small neighborhood of bins
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.extend(bins.get((br + dr, bc + dc), []))
        return out

    def reachable_types_for_res(pid_r, rr, rc):
        """Count how many distinct utility types are reachable from a residential placement."""
        if not utility_places:
            return 0
        proj_r = city.get_project(pid_r)
        r_cells = proj_r.absolute_hash_cells(Coordinates(rr, rc))
        r_bbox = abs_hash_bbox(pid_r, rr, rc)

        seen = set()
        for idx in nearby_utility_indices_for_bbox(r_bbox):
            u = utility_places[idx]
            st = u["stype"]
            if st in seen:
                continue

            if rect_manhattan_dist(r_bbox, u["bbox"]) > D:
                continue

            d = min_dist_hash_sets(r_cells, u["hash_cells"], D)
            if d <= D:
                seen.add(st)
                if len(seen) >= len(utility_types):
                    return len(seen)
        return len(seen)

    # ---- Hub layout ----
    max_res_h = max(p.h for p in top_res)
    max_res_w = max(p.w for p in top_res)

    if hub_step is None:
        hub_step = max(14, 2 * D + max(max_res_h, max_res_w) + 6)

    hub_positions = [(r, c) for r in range(0, H, hub_step) for c in range(0, W, hub_step)]
    random.shuffle(hub_positions)

    if max_hubs is None:
        if H * W >= 400_000:
            max_hubs = 450
        else:
            max_hubs = max(20, min(len(hub_positions), (H * W) // (hub_step * hub_step * 2)))

    hub_positions = hub_positions[:max_hubs]
    print(f"Hub step: {hub_step} | Hubs to try: {len(hub_positions)}")

    # Spiral around hub for packing utilities
    def spiral_offsets(radius):
        """Generate spiral offsets for placing utilities around a hub center."""
        offsets = [(0, 0)]
        for d in range(1, radius + 1):
            for dc in range(-d, d + 1):
                offsets.append((-d, dc))
                offsets.append((d, dc))
            for dr in range(-d + 1, d):
                offsets.append((dr, -d))
                offsets.append((dr, d))
        # unique
        seen = set()
        res = []
        for o in offsets:
            if o not in seen:
                seen.add(o)
                res.append(o)
        return res

    utility_pack_offsets = spiral_offsets(radius=max(5, D // 2 + 3))

    # ---- Place utilities in hubs ----
    hubs_built = 0
    for i, (hr, hc) in enumerate(hub_positions):
        if time.time() - t0 > max_runtime_s * 0.45:
            print("  [INFO] Time budget: stopping hub placement early.")
            break

        if i % 10 == 0:
            print(f"  Hub {i}/{len(hub_positions)} | utilities: {len(utility_places)} | placements: {len(placements)}")

        placed_any = False
        for st in utility_types:
            uproj = best_utility_by_type[st]
            pid = uproj.project_id

            for dr, dc in utility_pack_offsets:
                rr = max(0, min(H - uproj.h, hr + dr))
                cc = max(0, min(W - uproj.w, hc + dc))
                if can_place(pid, rr, cc):
                    do_place(pid, rr, cc)
                    register_utility(pid, rr, cc, st)
                    placed_any = True
                    break

        if placed_any:
            hubs_built += 1

    print(f"Utility hubs built: {hubs_built} | total utilities: {len(utility_places)}")

    # ---- Fill residential around subset of hubs ----
    radius = D + max(max_res_h, max_res_w) + 6

    hubs_for_fill = hub_positions[:hubs_built]
    if len(hubs_for_fill) > 350:
        random.shuffle(hubs_for_fill)
        hubs_for_fill = hubs_for_fill[:350]

    def sample_positions_around(hub_r, hub_c, radius, budget):
        """Sample candidate positions around a hub within the given radius."""
        candidates = []
        r0 = max(0, hub_r - radius)
        r1 = min(H - 1, hub_r + radius)
        c0 = max(0, hub_c - radius)
        c1 = min(W - 1, hub_c + radius)

        step = 3 if radius > 50 else 2
        for rr in range(r0, r1 + 1, step):
            for cc in range(c0, c1 + 1, step):
                candidates.append((rr, cc))
                if len(candidates) >= budget:
                    return candidates

        while len(candidates) < budget:
            candidates.append((random.randint(r0, r1), random.randint(c0, c1)))
        return candidates

    for i, (hr, hc) in enumerate(hubs_for_fill):
        if time.time() - t0 > max_runtime_s * 0.92:
            print("  [INFO] Time budget: stopping residential fill early.")
            break

        if i % 10 == 0:
            print(f"  Fill hub {i}/{len(hubs_for_fill)} | placements: {len(placements)}")

        candidates = sample_positions_around(hr, hc, radius, max_candidates_per_hub)

        placed_here = 0
        for rr, cc in candidates:
            if time.time() - t0 > max_runtime_s * 0.92:
                break

            best_pid = None
            best_val = -1
            best_pos = None
            best_reach = 0

            for proj_r in top_res:
                rrr = min(rr, H - proj_r.h)
                ccc = min(cc, W - proj_r.w)
                if rrr < 0 or ccc < 0:
                    continue
                if not can_place(proj_r.project_id, rrr, ccc):
                    continue

                reach = reachable_types_for_res(proj_r.project_id, rrr, ccc)
                if reach == 0:
                    continue

                # More aggressive than before: accept reach>=1
                hcnt = max(1, len(proj_r.hash_offsets))
                val = (proj_r.capacity * reach) / (hcnt ** 0.35)

                if val > best_val:
                    best_val = val
                    best_pid = proj_r.project_id
                    best_pos = (rrr, ccc)
                    best_reach = reach

            if best_pid is not None:
                # lower threshold => place many more residences
                if best_reach >= 1 and best_val >= 8:
                    do_place(best_pid, best_pos[0], best_pos[1])
                    placed_here += 1

            if placed_here >= 180:
                break

    dt = time.time() - t0
    print(f"Greedy finished (MODE B): placements={len(placements)} | time={dt:.2f}s")
    print("======== END GREEDY ========")
    return placements