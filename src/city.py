import importlib
import time, tracemalloc, inspect

from buildings import *
from algorithms.hill_climbing import *
from algorithms.greedy import *
from utils.metrics import save_metrics_to_csv

class City:
    """Holds one problem instance: grid size, walking distance, and building projects."""
    def __init__(self, H, W, D, B, projects, dataset_name="Unknown"):
        self.H = H
        self.W = W
        self.D = D
        self.B = B
        self.projects = projects
        self.dataset_name = dataset_name

        if self.B != len(self.projects):
            raise ValueError("B does not match number of parsed projects")

    def get_project(self, project_id):
        return self.projects[project_id]

    def print_city(self, show_plans=False):
        """Print full city instance information."""
        print("========== CITY ==========")
        print(f"Grid size      : {self.H} x {self.W}")
        print(f"Max distance D : {self.D}")
        print(f"Num projects B : {self.B}")
        print()

        for project in self.projects:
            print(f"--- Project {project.project_id} ---")
            print(f"Type       : {project.build_type}")
            print(f"Dimensions : {project.h} x {project.w}")

            if project.build_type == "R":
                print(f"Capacity   : {project.capacity}")
            else:
                print(f"Service type: {project.service_type}")

            print(f"Num '#' cells: {len(project.hash_offsets)}")

            if show_plans:
                print("Plan:")
                for row in project.plan_rows:
                    print(row)

            print()

        print("======== END CITY ========")


    def make_city(self, algorithm, *args, **kwargs):
        """
        Runs the chosen algorithm (implemented in ./algorithms), then computes and returns:
        (solution, score)
        """
        name = (algorithm or "").strip().lower()

        mapping = {
            "hill climbing": ("algorithms.hill_climbing", "hill_climbing"),
            "hc": ("algorithms.hill_climbing", "hill_climbing"),

            "simulated annealing": ("algorithms.simulated_annealing", "simulated_annealing"),
            "sa": ("algorithms.simulated_annealing", "simulated_annealing"),

            "tabu search": ("algorithms.tabu_search", "tabu_search"),
            "tabu": ("algorithms.tabu_search", "tabu_search"),

            "genetic algorithm": ("algorithms.genetic_algorithm", "genetic_algorithm"),
            "genetic": ("algorithms.genetic", "genetic_algorithm"),
            "ga": ("algorithms.genetic_algorithm", "genetic_algorithm"),

            # Optional baseline
            "greedy": ("algorithms.greedy", "greedy"),
            "greedy2": ("algorithms.greedy2", "greedy2"),
        }

        if name not in mapping:
            raise ValueError(
                "Unknown algorithm. Use: Hill Climbing, Simulated Annealing, Tabu Search, Genetic Algorithm (or Greedy)."
            )

        module_name, func_name = mapping[name]
        module = importlib.import_module(module_name)

        if not hasattr(module, func_name):
            raise NotImplementedError(
                f"Function '{func_name}' not found in '{module_name}'. Implement it in your algorithms folder."
            )

        func = getattr(module, func_name)

        # 1. Ready to extract parameters and their values (including defaults)
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)

        # 2. Fill in default values for any parameters not explicitly passed
        bound_args.apply_defaults()

        # 3. Convert bound arguments to a simple dict for easier formatting
        params_dict = dict(bound_args.arguments)

        # 4. Removes 'city' from the parameters dict, as they are not hyperparameters we want to log
        params_dict.pop('city', None)

        # 5. Format parameters into a string for CSV logging (e.g., "param1=value1, param2=value2")
        params_str = ", ".join([f"{k}={v}" for k, v in params_dict.items()])

        # 6. Run the algorithm while measuring execution time and peak memory usage
        tracemalloc.start()
        start_time = time.perf_counter()

        solution = func(self, *args, **kwargs)

        end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = end_time - start_time
        peak_mem_mb = peak_mem / (1024 * 1024)
        score = self.get_score(solution)

        placements_list = self._extract_placements(solution)
        total_p = len(placements_list)
        res_p = 0
        util_p = 0

        for p in placements_list:
            proj = self.get_project(p["project_id"])
            if proj.build_type == "R":
                res_p += 1
            else:
                util_p += 1

        save_metrics_to_csv(self, func_name, execution_time, peak_mem_mb, score, params_str, total_p, res_p, util_p)

        return solution, score

    def get_score(self, solution):
        placements = self._extract_placements(solution)

        H, W, D = self.H, self.W, self.D

        influence_grid = [[set() for _ in range(W)] for _ in range(H)]

        for p in placements:
            proj = self.get_project(p["project_id"])
            if proj.build_type == "U":
                s_type = proj.service_type
                cells = proj.absolute_hash_cells(p["top_left"])
                affected = set()

                for c in cells:
                    for dr in range(-D, D + 1):
                        rem = D - abs(dr)
                        for dc in range(-rem, rem + 1):
                            nr, nc = c.r + dr, c.c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                affected.add((nr, nc))

                for nr, nc in affected:
                    influence_grid[nr][nc].add(s_type)

        total = 0
        for p in placements:
            proj = self.get_project(p["project_id"])
            if proj.build_type == "R":
                capacity = proj.capacity
                cells = proj.absolute_hash_cells(p["top_left"])

                reachable_services = set()
                for c in cells:
                    reachable_services.update(influence_grid[c.r][c.c])

                total += capacity * len(reachable_services)

        return total


    # Internal helpers
    def _extract_placements(self, solution):
        """
        Normalizes different solution formats into a list of dictionaries:
        { 'project_id': int, 'top_left': Coordinates }
        Accepts:
        - solution.placements (list)
        - a list itself
        Each placement can be:
        - object with .project_id and .top_left
        - tuple (project_id, r, c)
        - tuple (project_id, Coordinates)
        - tuple (project_id, (r, c))
        """
        if solution is None:
            return []

        if hasattr(solution, "placements"):
            raw = solution.placements
        else:
            raw = solution

        placements = []
        for item in raw:
            pid, tl = self._parse_one_placement(item)
            placements.append({"project_id": pid, "top_left": tl})
        return placements

    def _parse_one_placement(self, item):
        """Parse one placement in a flexible way (see _extract_placements)."""
        # Object case: item.project_id + item.top_left
        if hasattr(item, "project_id") and hasattr(item, "top_left"):
            pid = item.project_id
            tl = item.top_left
            if not isinstance(tl, Coordinates):
                tl = Coordinates(tl[0], tl[1])
            return pid, tl

        # Tuple/list cases
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                pid, r, c = item
                return pid, Coordinates(r, c)

            if len(item) == 2:
                pid, pos = item
                if isinstance(pos, Coordinates):
                    return pid, pos
                if isinstance(pos, (tuple, list)) and len(pos) == 2:
                    return pid, Coordinates(pos[0], pos[1])

        raise ValueError("Invalid placement format in solution")

    def _min_manhattan_between_hash_sets(self, cells_a, cells_b):
        """Return min Manhattan distance between any coord in cells_a and any coord in cells_b."""
        if not cells_a or not cells_b:
            return 10**9

        # Iterate smaller set first
        if len(cells_a) > len(cells_b):
            cells_a, cells_b = cells_b, cells_a

        best = 10**9
        for a in cells_a:
            for b in cells_b:
                d = abs(a.r - b.r) + abs(a.c - b.c)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

