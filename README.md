# CityCrafter – City Plan Optimization

## Grade: 19.6/20 :star:

- Luana Filipa de Matos Lima (up202206845)
- Mariana Rocha Cristino (up202502528)

CityCrafter implements heuristics and metaheuristics to generate optimized city layouts. Algorithms included: Greedy, Hill Climbing, Simulated Annealing, Tabu Search, and Genetic Algorithm.

## Project Structure

```
CITYCRAFTER
├── data/                  # Input datasets
├── results/               # Generated metrics and results
├── src/                   # Source code
│   ├── algorithms/        # Algorithm implementations
│   ├── utils/             # Metrics & visualization helpers
│   ├── buildings.py
│   ├── city.py
│   ├── coordinates.py
│   ├── read_file.py
│   ├── main.py            # Main executable
│   └── Prediction&Results.ipynb  # Notebook for results & prediction
└── tests/                 # Additional test datasets
```

## Requirements

* Python 3.10+
* Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Running the Program

Execute the main menu:

```bash
python3 src/main.py
```

The program will prompt you to select:

1. **Algorithm** (`Greedy`, `Hill Climbing`, `Simulated Annealing`, `Tabu Search`, `Genetic`)
2. **Dataset** (letters `a-j`)

It will then generate a city layout and save results in the `results/` folder.

## Results & Prediction

All experimental analysis and predictive modeling are in the Jupyter Notebook:

```bash
jupyter notebook src/Prediction\&Results.ipynb
```

## Notes

* **Greedy** is fast and resource-light
* **Tabu Search** and **Simulated Annealing** achieve best solution quality
* **Genetic Algorithm** is often inconsistent and time-consuming
* The predictive model guides algorithm choice for new datasets
