from coordinates import Coordinates

def visualize_city(city, placements):
    """
    Visualize the city grid with the given placements. Each cell will show the building type (b) or '.' if empty.
    Placements is expected to be a list of tuples: (b, r, c)
    """

    if placements is None:
        placements = []

    print(len(placements))
    for b, r, c in placements:
        print(f"{b} {r} {c}")

    H, W = city.H, city.W
    grid = [['.'] * W for _ in range(H)]

    for _, (b, r, c) in enumerate(placements):
        proj = city.get_project(b)
        top_left = Coordinates(r, c)
        for cell in proj.absolute_hash_cells(top_left):
            grid[cell.r][cell.c] = str(b)

    for row in grid:
        print("".join(row))
