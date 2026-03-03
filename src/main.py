import sys

from read_file import parse_input_file
from algorithms.hill_climbling import hill_climbing
from utils.visualization import visualize_city

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/main.py data/<input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    city = parse_input_file(input_file)

    solution = hill_climbing(city, use_restart=True)

    score = city.get_score(solution)
    print(f"Final Score: {score}")

    visualize_city(city, solution)

if __name__ == "__main__":
    main()
