import datetime

from API.Problem import *
from API.MultiStageSolution import MultiStageSolution,  display_scheduling_fullcalendar_by_resource
from data_structures.Patient import PreferredTimerange
from data_treatments.read_timeplanner import parse_timeplanner
from data_treatments.read_patients import prescription_to_patients
from globals import *

from models.cp_sat_2_stages_extensive import SolverCPSAT2Stages
import numpy as np


def run_cp_sat_2_stages_monolith(
        patient_subset_count : int = 5,
        nb_scenarios : int = 2,
        solver_time_limit : int = 120
) -> MultiStageSolution :

    rng = np.random.default_rng()
    DATA_DIR = Path(__file__).resolve().parent.parent / Path("data")
    timeplanner_filepath = str(DATA_DIR / "schedule_080724.csv")
    patients_filepath = str(DATA_DIR / "pflow08072024_13072024.csv")
    # locations = file_to_locations(str(DATA_DIR / locations_file_name), oncopole_linacs)

    problem_parameters = problem_parameters_tuple(
        allow_change_for_mirrors_linacs=False,
        objective_weights={
            "earliest": 10.0,
            "treatment_range": 5.0,
            "time_regularity": 0.0,
            "machine_preference": 0.0,
            "time_preference": 0.0,
            "machine_balance": 0.0,
        },
    )
    existing_schedule = parse_timeplanner(timeplanner_filepath)

    problem = Problem(
        linacs=oncopole_linacs,
        locations=LOCATIONS,
        patients_queue=prescription_to_patients(patients_filepath, LOCATIONS),
        existing_schedule=existing_schedule,
        horizon_start=datetime.datetime(2024, 7, 1, 0, 0, 0),
        horizon=90,
        problem_parameters=problem_parameters,
        solver_time_limit=solver_time_limit,
    )
    patients_certain = problem.patients_queue[:patient_subset_count]
    patients_uncertain = problem.patients_queue[patient_subset_count:patient_subset_count + 10]

    for patient in patients_certain:
        patient.is_certain = True
        patient.preferred_timerange = PreferredTimerange(rng.integers(low=0, high=5))
        print(f"{patient.id} : {patient.preferred_timerange}")

    problem.patients_queue = patients_uncertain + patients_certain

    solver = SolverCPSAT2Stages(
        nb_scenarios=nb_scenarios,
    )
    solver.solve(problem, fix_seed=False)
    solution = solver.retrieve_solution(problem)
    solution.qi_treatment_ranges()

    return solution


if __name__ == "__main__":
    solution_ = run_cp_sat_2_stages_monolith(
        patient_subset_count = 3,
        nb_scenarios = 2,
        solver_time_limit = 20,
    )
    # for linac in oncopole_linacs:
    #     ressource_existing_schedule = [session
    #                                    for session in solution_.existing_schedule
    #                                    if session[1] == linac.id]
    #     display_scheduling_fullcalendar_by_resource(
    #         schedule= ressource_existing_schedule + solution_.certain_patients_schedule,
    #         linac_id=linac.id
    #     )