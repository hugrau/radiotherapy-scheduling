from globals import *
from API.Problem import *
from data_treatments.read_timeplanner import parse_timeplanner
from data_treatments.read_patients import prescription_to_patients
from models.cp_sat_2_stages_extensive import SolverCPSAT2Stages

import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / Path("data")
    timeplanner_filepath = str(DATA_DIR / "schedule_080724.csv")
    patients_filepath = str(DATA_DIR / "pflow08072024_13072024.csv")
    # locations = file_to_locations(str(DATA_DIR / locations_file_name), oncopole_linacs)

    problem_parameters = problem_parameters_tuple(
        allow_change_for_mirrors_linacs=False,
        objective_weights={
            "earliest": 10.0,
            "treatment_range": 5.0,
            "machine_preference": 1.0,
        },
    )

    patient_queue = prescription_to_patients(patients_filepath, LOCATIONS)
    problem = Problem(
        linacs=oncopole_linacs,
        locations=LOCATIONS,
        patients_queue=patient_queue,
        existing_schedule=parse_timeplanner(timeplanner_filepath),
        horizon_start=datetime.datetime(2024, 7, 1, 0, 0, 0),
        horizon=100,
        problem_parameters=problem_parameters,
        solver_time_limit=900,
    )

    # 3 Axis for sensitivity analysis : ratio certain/uncertain, nb of patients and number of scenarios

    df = []
    nb_iter = 1
    nb_scenarios = 1
    for k in range(10, 11, 5):
        for _ in range(nb_iter):
            # Take k random patients
            problem.patients_queue = random.sample(
                problem.patients_queue, k=min(k, len(problem.patients_queue))
            )
            print(f"Nb total patient : {k} and nb of certain : {k // 2}")
            solver = SolverCPSAT2Stages(nb_scenarios=nb_scenarios, nb_certain_patients=k//2)
            solver.solve(problem, fix_seed=False)
            solution = solver.retrieve_solution(problem)
            # print(solution.statistics)
            solution.statistics["nb_patients"] = k
            df.append(solution.statistics)
            # Reset the patient queue
            for patient in problem.patients_queue:
                patient.is_certain = False
    df = pd.DataFrame(df)
    print(df)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="nb_patients", y="solver_time", palette="tab10")

    plt.title(f"Solving time as a function of patients ({nb_scenarios} scenarios)")
    plt.xlabel("Number of patients")
    plt.ylabel("Solver time (s)")
    plt.savefig(f"boxplot_iter{nb_iter}_scenarios{nb_scenarios}.png")

    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, x="nb_patients", y="solver_time", palette="tab10", cut=0)
    plt.title(f"Solving time as a function of patients ({nb_scenarios} scenarios)")
    plt.xlabel("Number of patients")
    plt.ylabel("Solver time (s)")
    plt.savefig(f"violinplot_iter{nb_iter}_scenarios{nb_scenarios}.png")

    plt.show()
