class Coordinates:
    """Simple (row, col) coordinate for the city grid."""
    def __init__(self, r, c):
        self.r = r
        self.c = c

    def as_tuple(self):
        return (self.r, self.c)

    def manhattan(self, other):
        return abs(self.r - other.r) + abs(self.c - other.c)

    def __repr__(self):
        return f"({self.r},{self.c})"

    def __hash__(self):
        return hash((self.r, self.c))

    def __eq__(self, other):
        return isinstance(other, Coordinates) and self.r == other.r and self.c == other.c


class CityCoordinate(Coordinates):
    """City cell with optional building metadata."""
    def __init__(
        self,
        r,
        c,
        content,
        build_type = None,      # 'R' / 'U' / None
        project_id = None,
        service_type = None
    ):
        super().__init__(r, c)
        self.content = content  # '#' or '.'
        self.build_type = build_type
        self.project_id = project_id
        self.service_type = service_type

        if self.content not in ("#", "."):
            raise ValueError("content must be '#' or '.'")
        if self.build_type not in (None, "R", "U"):
            raise ValueError("build_type must be None, 'R' or 'U'")

    def is_hash(self):
        return self.content == "#"