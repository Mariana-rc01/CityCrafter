import sys

from read_file import parse_input_file
from utils.visualization import visualize_city

def choose_algorithm():
    algorithms = [
        ("greedy", "Greedy"),
        ("hill climbing", "Hill Climbing"),
        ("hill climbing boosted", "Hill Climbing Boosted")
    ]

    print("\n=== Choose Algorithm (or 'q' to quit) ===")
    for i, (_, label) in enumerate(algorithms, start=1):
        print(f"{i}) {label}")

    while True:
        choice = input("Algorithm number: ").strip().lower()
        if choice == "q":
            return "q"
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(algorithms):
                algo_key = algorithms[idx - 1][0]
                print(f"Selected algorithm: {algorithms[idx - 1][1]}\n")
                return algo_key
        print("Invalid option. Please enter a valid number or 'q'.")
        
        
def choose_dataset():
    datasets = {
        "a": "data/a_example.in",
        "b": "data/b_short_walk.in",
        "c": "data/c_going_green.in",
        "d": "data/d_wide_selection.in",
        "e": "data/e_precise_fit.in",
        "f": "data/f_different_footprints.in",
    }

    print("=== Choose Dataset or 'q' to quit ===")
    print("Available: a, b, c, d, e, f")

    while True:
        letter = input("Dataset letter (a-f): ").strip().lower()
        if letter == "q":
            return "q"
        if letter in datasets:
            path = datasets[letter]
            print(f"Selected dataset: {letter} -> {path}\n")
            return path
        print("Invalid dataset. Please choose a letter between a and f (lowercase) or 'q'.")


def interactive_menu():
    while True:
        print("\n==============================")
        print("      CITY CRAFTER MENU       ")
        print("==============================")

        algorithm = choose_algorithm()
        if algorithm == "q":
            print("\nExiting program.")
            break

        dataset_path = choose_dataset()
        if dataset_path == "q":
            print("\nExiting program.")
            break

        print("\n--- Running algorithm ---")
        city = parse_input_file(dataset_path)

        solution, score = city.make_city(algorithm)

        print("\n==============================")
        print(f"Final Score: {score}")
        print("==============================")

        # Optional visualization
        # visualize_city(city, solution)

        print("\nPress ENTER to return to menu or type 'q' to quit.")
        exit_choice = input().strip().lower()
        if exit_choice == "q":
            print("\nExiting program. Goodbye 👋")
            break
        
        
def main():
    if len(sys.argv) != 1:
        print("Usage: python3 src/main.py")
        sys.exit(1)
    else:
        interactive_menu()

if __name__ == "__main__":
    main()