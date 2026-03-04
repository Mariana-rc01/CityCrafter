import random
from coordinates import Coordinates

def hill_climbing(city,
                  max_iterations=10000,
                  patience=500,
                  min_delta=2,
                  use_restart=False,
                  max_restarts=1):

    def run_single_hc():

        H, W = city.H, city.W
        occupied = [[False] * W for _ in range(H)]
        placements = []
        current_score = 0
        best_score = 0
        iterations_without_improvement = 0

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

        def get_nearby_coordinates(radius):
            if not placements or random.random() < 0.2:
                return random.randint(0, H - 1), random.randint(0, W - 1)

            _, ext_r, ext_c = random.choice(placements)
            new_r = max(0, min(H - 1, ext_r + random.randint(-radius, radius)))
            new_c = max(0, min(W - 1, ext_c + random.randint(-radius, radius)))
            return new_r, new_c

        for i in range(max_iterations):
            if i % 500 == 0:
                print(f"  Iteration {i}, current score: {current_score}, best score: {best_score}")

            op = random.choices(
                ["ADD", "REMOVE", "MOVE", "CHANGE_TYPE"],
                weights=[0.6, 0.1, 0.15, 0.15]
            )[0]

            improved = False

            if op == "ADD":
                b = random.randint(0, city.B - 1)
                r, c = get_nearby_coordinates(city.D + 2)

                if can_place(b, r, c):
                    place(b, r, c)
                    new_score = city.get_score(placements)

                    if new_score >= current_score:
                        current_score = new_score
                        improved = True
                    else:
                        remove(len(placements) - 1)

            elif op == "REMOVE" and placements:
                idx = random.randint(0, len(placements) - 1)
                b, r, c = placements[idx]
                remove(idx)

                new_score = city.get_score(placements)

                if new_score >= current_score:
                    current_score = new_score
                    improved = True
                else:
                    place(b, r, c)

            elif op == "MOVE" and placements:
                idx = random.randint(0, len(placements) - 1)
                b, old_r, old_c = placements[idx]

                remove(idx)

                new_r = max(0, min(H - 1, old_r + random.randint(-3, 3)))
                new_c = max(0, min(W - 1, old_c + random.randint(-3, 3)))

                if can_place(b, new_r, new_c):
                    place(b, new_r, new_c)
                    new_score = city.get_score(placements)

                    if new_score >= current_score:
                        current_score = new_score
                        improved = True
                    else:
                        remove(len(placements) - 1)
                        place(b, old_r, old_c)
                else:
                    place(b, old_r, old_c)

            elif op == "CHANGE_TYPE" and placements:
                idx = random.randint(0, len(placements) - 1)
                old_b, r, c = placements[idx]

                remove(idx)

                new_b = random.randint(0, city.B - 1)

                if new_b != old_b and can_place(new_b, r, c):
                    place(new_b, r, c)
                    new_score = city.get_score(placements)

                    if new_score >= current_score:
                        current_score = new_score
                        improved = True
                    else:
                        remove(len(placements) - 1)
                        place(old_b, r, c)
                else:
                    place(old_b, r, c)

            if improved:
                if current_score > best_score + min_delta:
                    best_score = current_score
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= patience:
                print(f"  No improvement for {patience} iterations, stopping this run.")
                break

        return placements, best_score

    if not use_restart:
        placements, _ = run_single_hc()
        return placements

    global_best_score = -1
    global_best_solution = []

    restart = 0
    while restart < max_restarts:

        print(f"Restart {restart + 1}")

        placements, score = run_single_hc()

        if score > global_best_score:
            global_best_score = score
            global_best_solution = placements

        restart += 1

    print(f"Best score found: {global_best_score}")
    return global_best_solution
