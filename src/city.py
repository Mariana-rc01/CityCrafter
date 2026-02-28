from buildings import *
import importlib

class City:
    """Holds one problem instance: grid size, walking distance, and building projects."""
    def __init__(self, H, W, D, B, projects):
        self.H = H
        self.W = W
        self.D = D
        self.B = B
        self.projects = projects

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
            "genetic algorithms": ("algorithms.genetic_algorithm", "genetic_algorithm"),
            "ga": ("algorithms.genetic_algorithm", "genetic_algorithm"),

            # Optional baseline
            "greedy": ("algorithms.greedy", "greedy"),
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

        # Convention: algorithm(city, *args, **kwargs) -> solution
        solution = func(self, *args, **kwargs)

        # Compute official score for the returned solution
        score = self.get_score(solution)

        return solution, score


    def get_score(self, solution):
        """
        Computes the official score:
        For each residential building with capacity r, earn r points for each DISTINCT
        utility service type reachable within distance <= D (shortest Manhattan distance
        between any '#' cells).
        """
        placements = self._extract_placements(solution)

        # Separate residential and utility placements
        res = []
        util = []
        for p in placements:
            proj = self.get_project(p["project_id"])
            if proj.build_type == "R":
                res.append(p)
            else:
                util.append(p)

        # Precompute absolute '#' cells per placement (simple and clear)
        for p in placements:
            proj = self.get_project(p["project_id"])
            p["hash_cells"] = proj.absolute_hash_cells(p["top_left"])

        total = 0
        D = self.D

        for r_place in res:
            r_proj = self.get_project(r_place["project_id"])
            capacity = r_proj.capacity
            reachable_services = set()

            for u_place in util:
                u_proj = self.get_project(u_place["project_id"])
                stype = u_proj.service_type

                # Skip if this service type already counted for this residential
                if stype in reachable_services:
                    continue

                d = self._min_manhattan_between_hash_sets(r_place["hash_cells"], u_place["hash_cells"])
                if d <= D:
                    reachable_services.add(stype)

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