import os
from city import *

def parse_input_file(path, validate_plans = True):
    """
    Parse a City Plan input file and return a City instance.

    Format:
    First line: H W D B
    Then for each project i in [0..B-1]:
      line: tp hp wp x   (x = capacity if tp='R', service_type if tp='U')
      hp lines: plan rows with '#' or '.'
    """

    # Extract dataset name from file path
    dataset_name = os.path.splitext(os.path.basename(path))[0]

    with open(path, "r", encoding="utf-8") as f:
        # Read first non-empty line
        first = ""
        while first == "":
            first = f.readline()
            if first == "":
                raise ValueError("Empty input file")
            first = first.strip()

        H, W, D, B = map(int, first.split())

        projects = []
        for pid in range(B):
            header = f.readline().strip()
            while header == "":
                header = f.readline().strip()

            parts = header.split()
            if len(parts) != 4:
                raise ValueError(f"Project {pid}: invalid header line")

            tp = parts[0]
            hp = int(parts[1])
            wp = int(parts[2])
            x = int(parts[3])

            plan_rows = []
            for _ in range(hp):
                row = f.readline().rstrip("\n")
                if len(row) != wp:
                    raise ValueError(f"Project {pid}: plan row length mismatch")
                plan_rows.append(row)

            if tp == "R":
                proj = BuildingProject(
                    project_id=pid, h=hp, w=wp, plan_rows=plan_rows,
                    build_type="R", capacity=x, service_type=-1,
                    validate=validate_plans
                )
            elif tp == "U":
                proj = BuildingProject(
                    project_id=pid, h=hp, w=wp, plan_rows=plan_rows,
                    build_type="U", capacity=0, service_type=x,
                    validate=validate_plans
                )
            else:
                raise ValueError(f"Project {pid}: unknown project type '{tp}'")

            projects.append(proj)

        return City(H=H, W=W, D=D, B=B, projects=projects, dataset_name=dataset_name)
