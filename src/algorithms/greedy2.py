import time
from coordinates import Coordinates

def greedy2(city):
    """
    Heurística Construtiva O(1) extraída da Tabu Search.
    Constrói a cidade através de um padrão de blocos em saltos de 3x3.
    """
    t0 = time.time()
    H, W, D = city.H, city.W, city.D

    print("========== GUIDED BLOCKS greedy2 ==========")
    print(f"Grid: {H} x {W} | D={D} | Projects={city.B}")

    # --- 1. PREPARAÇÃO DO ESTADO O(1) ---
    occupied = [[False] * W for _ in range(H)]
    influence_grid = [[{} for _ in range(W)] for _ in range(H)]
    residential_at = [[None] * W for _ in range(H)]

    res_coverage = {}
    res_capacity = {} # Guarda a capacidade de cada casa colocada
    
    final_placements = []
    current_score = 0
    next_uid = 0
    cell_cache = {}

    # --- 2. SELEÇÃO DE PEÇAS ---
    residential = [p for p in city.projects if p.build_type == "R"]
    utilities = [p for p in city.projects if p.build_type == "U"]
    
    # Ordenar as casas pelo melhor rácio (Capacidade / Tamanho)
    residential_sorted = sorted(residential, key=lambda p: p.capacity / max(1, p.h * p.w), reverse=True)
    top_res = residential_sorted[:40]

    # Escolher 1 serviço (utility) de cada tipo
    best_utils = []
    seen = set()
    for u in utilities:
        if u.service_type not in seen:
            seen.add(u.service_type)
            best_utils.append(u.project_id)

    # --- 3. FUNÇÕES AUXILIARES ---
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

    def place(b, r, c):
        nonlocal current_score, next_uid
        proj = city.get_project(b)
        
        # Verificar limites da cidade
        if r < 0 or r + proj.h > H or c < 0 or c + proj.w > W:
            return False
            
        cells = get_cells(b, r, c)
        
        # Verificar colisões
        for cell in cells:
            if occupied[cell.r][cell.c]:
                return False

        uid = next_uid
        next_uid += 1
        gain = 0

        # Calcular pontuação instantânea e preencher grelha
        if proj.build_type == "R":
            cap = proj.capacity
            res_capacity[uid] = cap
            cov = {}
            for cell in cells:
                occupied[cell.r][cell.c] = True
                residential_at[cell.r][cell.c] = uid
                for s_type, count in influence_grid[cell.r][cell.c].items():
                    cov[s_type] = cov.get(s_type, 0) + count
            
            for s_type, count in cov.items():
                if count > 0:
                    gain += cap
            res_coverage[uid] = cov
            
        else: # Build Type == "U"
            s_type = proj.service_type
            affected = set()
            for cell in cells:
                occupied[cell.r][cell.c] = True
                for nr, nc in get_influence_diamond(cell.r, cell.c, D):
                    affected.add((nr, nc))
                    
            for nr, nc in affected:
                influence_grid[nr][nc][s_type] = influence_grid[nr][nc].get(s_type, 0) + 1
                ruid = residential_at[nr][nc]
                if ruid is not None:
                    cov = res_coverage[ruid]
                    cov[s_type] = cov.get(s_type, 0) + 1
                    if cov[s_type] == 1:
                        gain += res_capacity[ruid]

        current_score += gain
        final_placements.append((b, r, c))
        return True

    # --- 4. O MOTOR CONSTRUTIVO PRINCIPAL (SALTOS 3x3) ---
    placed_res = 0
    placed_util = 0
    
    print("Start building guided blocks...")
    for r in range(0, H, 3):
        if r % 60 == 0:
            print(f"  Scan row {r}/{H} | placements: {len(final_placements)} | R={placed_res} U={placed_util}")
            
        for c in range(0, W, 3):
            # 1º Passo: Tentar colocar todos os tipos de serviços possíveis na raiz (r, c)
            for uid in best_utils:
                if place(uid, r, c):
                    placed_util += 1
                    
            # 2º Passo: Tentar circundar com as melhores casas na diagonal (r+1, c+1)
            for res in top_res:
                if place(res.project_id, r + 1, c + 1):
                    placed_res += 1

    dt = time.time() - t0
    print(f"Guided Blocks greedy2 finished: placements={len(final_placements)} | score={current_score} | time={dt:.2f}s")
    print("======== END GUIDED BLOCKS greedy2 ========")

    return final_placements