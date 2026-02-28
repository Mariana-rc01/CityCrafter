from coordinates import *
from collections import deque

class BuildingProject:
    """
    One building project (plan) from input.
    Validates plan constraints: edge-occupied, connected '#', no holes.
    """
    def __init__(
        self,
        project_id,
        h,
        w,
        plan_rows,                  # List[str]
        build_type,                 # 'R' or 'U'
        capacity = 0,               # only for 'R'
        service_type = -1,          # only for 'U'
        validate = True
    ):
        self.project_id = project_id
        self.h = h
        self.w = w
        self.plan_rows = plan_rows
        self.build_type = build_type
        self.capacity = capacity
        self.service_type = service_type

        if self.build_type not in ("R", "U"):
            raise ValueError("build_type must be 'R' or 'U'")
        if len(plan_rows) != h or any(len(row) != w for row in plan_rows):
            raise ValueError("plan_rows dimensions do not match h x w")
        if any(ch not in ("#", ".") for row in plan_rows for ch in row):
            raise ValueError("plan_rows must contain only '#' and '.'")

        # Precompute '#' offsets for fast placement checks
        self.hash_offsets = []
        for i in range(h):
            for j in range(w):
                if plan_rows[i][j] == "#":
                    self.hash_offsets.append((i, j))

        if validate:
            self._validate_plan()

    def absolute_hash_cells(self, top_left):
        """Return absolute coordinates of occupied '#' cells when placed at top_left."""
        return [Coordinates(top_left.r + dr, top_left.c + dc) for (dr, dc) in self.hash_offsets]

    def fits_in_grid(self, top_left, H, W):
        """Check if the plan bounding box fits inside HxW."""
        return 0 <= top_left.r <= H - self.h and 0 <= top_left.c <= W - self.w

    def _validate_plan(self):
        """Validate all plan constraints required by the statement."""
        if not self.hash_offsets:
            raise ValueError(f"Project {self.project_id}: plan has no '#' cells")
        self._check_edges_occupied()
        self._check_hash_connected()
        self._check_no_holes()

    def _check_edges_occupied(self):
        """At least one '#' on each of the four edges."""
        top = any(self.plan_rows[0][c] == "#" for c in range(self.w))
        bottom = any(self.plan_rows[self.h - 1][c] == "#" for c in range(self.w))
        left = any(self.plan_rows[r][0] == "#" for r in range(self.h))
        right = any(self.plan_rows[r][self.w - 1] == "#" for r in range(self.h))
        if not (top and bottom and left and right):
            raise ValueError(f"Project {self.project_id}: edge-occupied constraint failed")

    def _check_hash_connected(self):
        """All '#' cells must form a single 4-connected component."""
        start = self.hash_offsets[0]
        q = deque([start])
        visited = set([start])

        while q:
            r, c = q.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < self.h and 0 <= nc < self.w and (nr, nc) not in visited:
                    if self.plan_rows[nr][nc] == "#":
                        visited.add((nr, nc))
                        q.append((nr, nc))

        if len(visited) != len(self.hash_offsets):
            raise ValueError(f"Project {self.project_id}: '#' cells are not 4-connected")

    def _check_no_holes(self):
        """All '.' must be reachable from the plan border through '.' (4-neighborhood)."""
        q = deque()
        visited = set()

        # Seed BFS with all border '.' cells
        for c in range(self.w):
            if self.plan_rows[0][c] == ".":
                q.append((0, c)); visited.add((0, c))
            if self.plan_rows[self.h - 1][c] == ".":
                q.append((self.h - 1, c)); visited.add((self.h - 1, c))
        for r in range(self.h):
            if self.plan_rows[r][0] == ".":
                q.append((r, 0)); visited.add((r, 0))
            if self.plan_rows[r][self.w - 1] == ".":
                q.append((r, self.w - 1)); visited.add((r, self.w - 1))

        while q:
            r, c = q.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < self.h and 0 <= nc < self.w and (nr, nc) not in visited:
                    if self.plan_rows[nr][nc] == ".":
                        visited.add((nr, nc))
                        q.append((nr, nc))

        for r in range(self.h):
            for c in range(self.w):
                if self.plan_rows[r][c] == "." and (r, c) not in visited:
                    raise ValueError(f"Project {self.project_id}: hole detected in plan")