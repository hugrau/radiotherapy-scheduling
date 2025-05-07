import json
import sys
import argparse

import pandas as pd
import os
from typing import List, Dict
from itertools import product
from collections import namedtuple
import tomllib
from datetime import datetime, timedelta


from API.Problem import Problem, problem_parameters_tuple
from data_structures.Location import Location
from data_treatments.read_proportions import read_proportions_file
# from data_treatments.read_timeplanner import parse_timeplanner
from globals import *
from data_treatments.calendars import generate_business_day_array
from models.cp_sat_2_stages_extensive import SolverCPSAT2Stages
from simulations.generate_patients import generate_hidden_truth, generate_patients_from_instance


def create_simulation_config_files(
        instances_files: List[str],
):
    """
    :param instances_files:
    """
    all_simulations_file = Path("simulations") / Path("simulations_list.txt")
    Param = collections.namedtuple(
        "Param",
        (
            "instance_data_file",
            "scenarios",
            "aggregation_criterion",
            "first_stage_weight_proportion",
            "queue_length",
            "one_scenario_strategy"
        )
    )
    for instance_file in instances_files:
        config_prefix = instance_file.split("/")[-1].split(".")[0]

        # HERE WE DEFINE PARAMETERS FOR THE SIMULATION.
        # TODO : maybe replace str by enum types.
        parameter_instance_data_file = (instance_file,)
        # parameter_number_days_simulated =
        parameter_nb_scenarios = (0, 1, 2, 5, 10)
        # NOTE : if nb_scenarios = 0 all parameters below won't matter.
        parameter_aggregation_criterion = ("AVG", "MAX")
        parameter_first_stage_weight_proportion = (0.8,)
        parameter_queue_length = ("LIMITED",) # LIMITED = 15 MAX
        parameter_one_scenario_strategy = ("AVERAGE_SCENARIO", "EARLIEST")

        # Cartesian product of all parameters sets.
        configs = product(
            parameter_instance_data_file,
            parameter_nb_scenarios,
            parameter_aggregation_criterion,
            parameter_first_stage_weight_proportion,
            parameter_queue_length,
            parameter_one_scenario_strategy
        )

        deterministic_flag = False
        # Write a config file for each config.
        for config in configs:
            config = Param(*config)
            hash_config = hash(config) % ((sys.maxsize + 1) * 2)
            if config.scenarios == 0:
                # We don't want to generate useless configs (by cartesian product) for deterministic model.
                results_directory_name = (
                        config_prefix + "_deterministic"
                )
                results_directory_path = Path("simulations") / Path(results_directory_name)

            else :
                # Create the directory where the results are stored.
                results_directory_name = (
                     config_prefix + "_" + str(hash_config)
                )
                results_directory_path = Path("simulations") / Path(results_directory_name)

            try:
                results_directory_path.mkdir()
                print(f"Directory '{results_directory_path}' created successfully.")
            except FileExistsError:
                print(f"Directory '{results_directory_path}' already exists.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{results_directory_path}'.")
            except Exception as e:
                print(f"An error occurred: {e}")

            if config.scenarios == 0 and not deterministic_flag:
                config_filename = f"{results_directory_path}/" + config_prefix + "_deterministic.toml"
                with open(all_simulations_file, "a+") as f:
                    f.writelines(f"{config_filename}\n")
                deterministic_flag = True
            elif config.scenarios == 0 and deterministic_flag:
                continue
            else :
                config_filename = f"{results_directory_path}/" + "config_" + config_prefix + "_" + str(hash_config) + ".toml"
                with open(all_simulations_file, "a+") as f:
                    f.writelines(f"{config_filename}\n")
            # + f"_{config.scenarios}_aggreg{config.aggregation_criterion}_weight{config.first_stage_weight_proportion}_queue{config.queue_length}.toml"
            with open(config_filename, "w") as f:
                toml_str = (
                f"[general]\n"
                f"config_hash = \"{hash_config}\"\n"
                f"\n"
                f"[simulation]\n"
                f"instance_data_file = \"{config.instance_data_file}\"\n"
                f"scenarios = {config.scenarios}\n"
                f"aggregation_criterion = \"{config.aggregation_criterion}\"\n"
                f"first_stage_weight_proportion = {config.first_stage_weight_proportion}\n"
                f"queue_length = \"{config.queue_length}\"\n"
                f"one_scenario_strategy = \"{config.one_scenario_strategy}\"\n"
                )
                f.writelines(toml_str)


def simulate_over_horizon(
        start_date: datetime,
        number_days_simulated: int,
        simulation_config_file: str,
        linacs: List[Linac],
        locations: Dict[str, Location],
):
    """
    Simulate a batch scheduling process over a given number of days.
    :param start_date:
    :param number_days_simulated:
    :param simulation_config_file:
    :param linacs:
    :param locations:
    """
    # TODO: Find a more robust way to do that. Issue if absolute path.
    dir_path_list = simulation_config_file.split('/')
    results_directory_path = f"{dir_path_list[0]}/{dir_path_list[1]}"
    # print(results_directory_path)

    # Read the simulation parameters.
    with open(simulation_config_file, "rb") as f:
        simulation_parameters = tomllib.load(f)

    print("Simulation parameters:")
    for name, parameter in simulation_parameters["simulation"].items():
        print(f"\t - {name} : {parameter}")

    # Generate all the patients according to instance file and proportions.
    patients = generate_patients_from_instance(
        instance_file=simulation_parameters["simulation"]["instance_data_file"],
        dict_locations=locations,
    )

    # Define the local problem parameters.
    problem_parameters = problem_parameters_tuple(
        allow_change_for_mirrors_linacs=False,
        objective_weights={
            "earliest": 400.0,
            "treatment_range": 200.0,
            "due_dates_penalties": 1.0
            # "time_regularity": 1.0,
            # "machine_preference": 1.0,
            # "time_preference": 1.0,
            # "machine_balance": 1.0,
        },
    )

    # Set the current date, to be updated in iterations.
    current_date = start_date

    # Define the global quality indicateurs
    kpis_df = []
    solution_stats = []
    qi_overall_treatment_dates = 0
    qi_overall_count_violated = 0
    qi_overall_treatment_ranges = 0
    qi_daily_machines_occupancies = {day:
                                         {linac.id: 0 for linac in linacs}
                                     for day in range(number_days_simulated + 90)
                                     }

    # If we start with a given schedule, the following value should be modified adequately, such as
    # horizon_start value.
    # current_schedule = parse_timeplanner(timeplanner_filepath)

    # Define the current problem.
    current_problem = Problem(
        linacs=linacs,
        locations=locations,
        patients_queue=[],
        existing_schedule=[],
        horizon_start=current_date,
        horizon=90,  # TODO: How to compute better this horizon ?
        problem_parameters=problem_parameters,
        solver_time_limit=3600
    )

    business_days_array = generate_business_day_array(
        start_date=start_date,
        num_days=number_days_simulated,
    )

    for k in range(number_days_simulated):

        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
              "%                                                                                                   %\n"
              f"%                               Scheduling day {current_date.date()} - n°{k}                                     %\n"
              "%                                                                                                   %\n"
              f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # If the current day is not a working day, there is no scheduling process on that day.
        if not business_days_array[k]:
            print(f"Day {current_date.date()} is not a working day, skipping.")
            # Must update the date before next iteration.
            current_date = current_date + timedelta(days=1)
            current_problem.horizon_start = current_date
            continue

        certain_patients = [
            patient
            for patient in patients
            if patient.scanner_date <= current_date and patient.medical_validation_date == current_date
        ]
        if simulation_parameters["simulation"]["scenarios"] != 0 :   # Stochastic case.
            uncertain_patients = [
                patient
                for patient in patients
                if patient.scanner_date <= current_date < patient.medical_validation_date
            ]
        else:
            uncertain_patients = []

        for patient in certain_patients:
            # Set the proper patient status (used by solver).
            patient.is_certain = True
            print(f"Patient {patient.id} is certain : "
                  f"scanner date {patient.scanner_date} and medical validation date {patient.medical_validation_date}")
        for patient in uncertain_patients:
            # Set the proper patient status (used by solver).
            patient.is_certain = False
            print(f"Patient {patient.id} is uncertain : "
                  f"scanner date {patient.scanner_date} and medical validation date {patient.medical_validation_date}")

        # Sort the uncertain patients, the ones with the longest time in queue are used for scenarios since they are
        # the most likely to be validated soon.
        uncertain_patients.sort(key=lambda x: current_date - x.scanner_date, reverse=True)

        # Draw uncertain patients among the previous sorted list.
        if simulation_parameters["simulation"]["queue_length"] == "LIMITED" :
            uncertain_patients_filtered = uncertain_patients[:15]
        else :
            uncertain_patients_filtered = uncertain_patients

        current_problem.patients_queue = certain_patients + uncertain_patients_filtered

        # Create an instance of the solver object, number of certain patients is pre-attributed here but can change
        # later.
        if simulation_parameters["simulation"]["scenarios"] == 0 :
            # Ugly fix for now, but issue if nb_scenarios == 0.
            solver = SolverCPSAT2Stages(
                nb_scenarios=simulation_parameters["simulation"]["scenarios"]+1,
                first_stage_weight_proportion = simulation_parameters["simulation"]["first_stage_weight_proportion"],
                aggregation_criterion = simulation_parameters["simulation"]["aggregation_criterion"],
                one_scenario_strategy = simulation_parameters["simulation"]["one_scenario_strategy"],
            )
        else :
            solver = SolverCPSAT2Stages(
                nb_scenarios=simulation_parameters["simulation"]["scenarios"],
                first_stage_weight_proportion=simulation_parameters["simulation"]["first_stage_weight_proportion"],
                aggregation_criterion=simulation_parameters["simulation"]["aggregation_criterion"],
                one_scenario_strategy=simulation_parameters["simulation"]["one_scenario_strategy"],
            )

        # Solve the problem with the newly generated patients and retrieve solution.
        solver.solve(current_problem, fix_seed=False)
        solution = solver.retrieve_solution(current_problem)
        current_solution_qi_treatment_dates, _ = solution.qi_treatment_dates(verbose=False)
        current_solution_qi_count_violated, certain_patient_count_periods = solution.qi_periods_respected(verbose=False)
        current_solution_qi_treatment_ranges, _ = solution.qi_treatment_ranges(verbose=False)

        # Update the quality indicators.
        qi_overall_treatment_dates += current_solution_qi_treatment_dates
        qi_overall_count_violated += current_solution_qi_count_violated
        qi_overall_treatment_ranges += current_solution_qi_treatment_ranges

        # Update the daily machine occupancies.
        for p in solution.machine_occupancy.keys():
            for q in qi_daily_machines_occupancies[k + p].keys():
                qi_daily_machines_occupancies[k + p][q] += solution.machine_occupancy[p][q]

        dict_current_solution_kpis = {
            'day': k,
            'sum_qi_treatment_dates': qi_overall_treatment_dates,
            'sum_qi_treatment_ranges': qi_overall_treatment_ranges,
            'sum_qi_count_violated': qi_overall_count_violated,
            'machine_id_0_load': 100 * qi_daily_machines_occupancies[k][0] / 684,
            'machine_id_1_load': 100 * qi_daily_machines_occupancies[k][1] / 720,
            'machine_id_2_load': 100 * qi_daily_machines_occupancies[k][2] / 740,
            'machine_id_3_load': 100 * qi_daily_machines_occupancies[k][3] / 720,
            'machine_id_4_load': 100 * qi_daily_machines_occupancies[k][4] / 720,
            'machine_id_5_load': 100 * qi_daily_machines_occupancies[k][5] / 460,
            'machine_id_6_load': 100 * qi_daily_machines_occupancies[k][6] / 720,
        }
        # current_solution_kpis = pd.DataFrame([dict_current_solution_kpis])
        # pd.concat([kpis_df, current_solution_kpis], ignore_index=True)
        kpis_df.append(dict_current_solution_kpis)
        # print(kpis_df)

        # Write current solution stats.
        solution.statistics['day'] = k
        solution.statistics['nb_certain'] = len(certain_patients)
        solution.statistics['nb_uncertain'] = len(uncertain_patients)
        solution_stats.append(solution.statistics)

        # Update the current schedule with the sessions scheduled for certain patients.
        current_problem.existing_schedule.extend(
            solution.to_patient_objects()
        )

        # Update uncertain patients uncertainty.

        # Update the date.
        current_date = current_date + timedelta(days=1)
        current_problem.horizon_start = current_date

    # print(f"Overall QI treatment dates: {global_qi_treatment_dates}")
    solution_stats_df = pd.DataFrame(solution_stats)
    kpis_df = pd.DataFrame(kpis_df)

    # Save the results.
    kpis_df.to_csv(f"{results_directory_path}/kpis_instance_.csv", index=False)
    solution_stats_df.to_csv(f"{results_directory_path}/solution_stats.csv", index=False)

    # Note: the last problem.existing_schedule is updated with the last batch of patients.
    # overall_schedule = existing_schedule_to_list(current_problem)
    # display_scheduling_fullcalendar(overall_schedule)
    # for linac in linacs:
    #    display_scheduling_fullcalendar_by_resource(overall_schedule, linac.id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("instance_file", help="Path to csv file with instance data.")
    parser.add_argument("simulation_config_file", help="Path to simulation configuration file.")
    args = parser.parse_args()

    proportions_df = read_proportions_file(proportions_file_name)
    reference_date = datetime.datetime(2025, 4, 20, 0, 0, 0)

    number_days_simulated_ = 50
    simulation_file = args.simulation_config_file

    # generate_hidden_truth(
    #     start_date=reference_date,
    #     number_days_simulated=number_days_simulated_,
    #     average_patients_per_day=25,
    #     proportions_df=proportions_df,
    #     dict_locations=LOCATIONS,
    # )

    # create_simulation_config_files(
    #     [   
    #         "simulations/instances/instance_135224002065681560.csv",
    #         "simulations/instances/instance_250111777107487511.csv",
    #         "simulations/instances/instance_2192742545724921127.csv",
    #         "simulations/instances/instance_2775139937030573620.csv",
    #         "simulations/instances/instance_3616442042494646722.csv",
    #         "simulations/instances/instance_4929011461548171347.csv",
    #         "simulations/instances/instance_6096551986938506664.csv",
    #         "simulations/instances/instance_7465116319130191447.csv",
    #         "simulations/instances/instance_9707043526940373274.csv",
    #         "simulations/instances/instance_10397321157149497167.csv",
    #         "simulations/instances/instance_10987222697483944105.csv",
    #         "simulations/instances/instance_12766632818714929165.csv",
    #         "simulations/instances/instance_15681825288749210743.csv",
    #         "simulations/instances/instance_16060245573606360052.csv",
    #         "simulations/instances/instance_17870085318395153772.csv"
    #     ],
    # )

    simulate_over_horizon(
            start_date=reference_date,
            number_days_simulated=number_days_simulated_,
            simulation_config_file=simulation_file,
            linacs=oncopole_linacs,
            locations=LOCATIONS,
    )
