import random
import logging
import math
from abc import ABC
from typing import Dict, Optional

from ortools.sat.python import cp_model
from datetime import timedelta, datetime, time

from API.Problem import Problem
from API.MultiStageSolution import MultiStageSolution
from data_structures.Location import Priority

from globals import *
from time import perf_counter

from models.Solver import Solver
from data_treatments.calendars import (
    merged_schedule_to_list,
    compute_unavailable_slots_by_machine,
)
from data_structures.Patient import PreferredTimerange

INFINITY = 1000000


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """
    Print intermediate solutions.
    """

    def __init__(
            self,
            dict_certain_patient_vars: Dict,
            dict_uncertain_patient_vars: Dict,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__dict_certain_patients_vars = dict_certain_patient_vars
        self.__dict_uncertain_patient_vars = dict_uncertain_patient_vars
        self.__solution_count = 0
        self.__solution_limit = INFINITY
        self.start_time = perf_counter()

    def on_solution_callback(self) -> None:
        """
        Print intermediate solution when solver finds one.
        :return: None
        """
        print(
            f"========== Solution found @ t = {perf_counter() - self.start_time:.3f} s "
            f"/ Value : {self.ObjectiveValue():.2f} "
            f"- Best bound : {self.BestObjectiveBound():.2f} "
            f"- Gap : {100 * (self.ObjectiveValue() / self.BestObjectiveBound() - 1 if self.BestObjectiveBound() > 0 else 0):.2f} %"
        )
        self.__solution_count += 1
        if self.__solution_count >= self.__solution_limit:
            print(f"Stop search after {self.__solution_limit} solution.")
            self.stop_search()

    @property
    def solution_count(self) -> int:
        return self.__solution_count


class SolverCPSAT(Solver, ABC):

    def __init__(self,
                 name: str,
                 nb_scenarios: int,
                 first_stage_weight_proportion: int,
                 aggregation_criterion: str = "MAX",
                 one_scenario_strategy: str = "AVERAGE_SCENARIO",
                 ):
        super().__init__(cp_model.CpModel(), "CP-SAT", name, {})
        # Model parameters
        self.nb_scenarios: int = nb_scenarios
        self.first_stage_weight_proportion = first_stage_weight_proportion
        self.aggregation_criterion = aggregation_criterion
        self.one_scenario_strategy = one_scenario_strategy
        # Variables related attributes
        self.dict_certain_patient_vars = None
        self.dict_uncertain_patient_vars = None
        self.dict_machine_vars = None
        self.max_due_dates_penalties = None
        # Objective related attributes
        self.first_stage_earliest_var = None
        self.second_stage_earliest_var = None
        self.first_stage_treatment_range_var = None
        self.second_stage_treatment_range_var = None
        self.max_due_dates_penalties_var = None
        # Data preprocessing related attributes
        self.new_tasks_count = None
        self.occupancy_count_by_linac: Dict[int: int] = {}
        self.total_occupancy_count: int = 0
        self.average_occupancy: int = 0
        self.minutes_horizon = None
        self.ready_dates = None
        self.due_dates = None
        self.ready_dates_uncertain = None
        self.due_dates_uncertain = None
        self.task_processing_time_data = None
        self.preferences_by_patient = None
        # Solver related attributes
        self.solver = None
        self.status = None
        self.solution_callback = None
        self.objective_first_stage_earliest = 0
        self.objective_first_stage_treatment_range = 0
        self.objective_second_stage_earliest = 0
        self.objective_second_stage_treatment_range = 0
        self.objective_second_stage_due_dates_penalties = 0

    def preprocess_problem_data(self, problem: Problem):

        # =======================================================================
        # Compute the current occupancy on each linac.
        # =======================================================================
        for linac in problem.linacs:
            self.occupancy_count_by_linac[linac.id] = 0
        self.total_occupancy_count = 0
        self.average_occupancy = int(sum(self.occupancy_count_by_linac.values()) / len(self.occupancy_count_by_linac))
        # =======================================================================
        # Compute the ready dates and due dates in minutes for each patient.
        # =======================================================================
        ready_dates = {}
        due_dates = {}

        ready_dates_uncertain = {}
        due_dates_uncertain = {}

        for patient in problem.patients_queue:
            if patient.is_certain:
                # Get the number of minutes from start of horizon to the ready_date of patient.
                ready_date_min = int(
                    (
                            datetime.datetime.combine(patient.ready_date, time(0, 0, 0))
                            - problem.horizon_start
                    ).total_seconds()
                    / 60
                )
                # Get the number of minutes from start of horizon to the due_date of patient.
                due_date_min = int(
                    (
                            datetime.datetime.combine(patient.due_date, time(0, 0, 0))
                            - problem.horizon_start
                    ).total_seconds()
                    / 60
                )
                ready_dates[patient.id] = ready_date_min
                due_dates[patient.id] = due_date_min
            else:
                ready_dates_by_scenario = {}
                due_dates_by_scenario = {}
                print(f"One scenario strategy is {self.one_scenario_strategy}.")
                for scenario in range(self.nb_scenarios):
                    if scenario == 0 and self.one_scenario_strategy == "AVERAGE_SCENARIO":
                        # In this case, for the first scenario we take as the patient's ready_date, the supposed average
                        # delay = scanner_date + 8 days.
                        scenario_ready_date_min = int(
                            (
                                    datetime.datetime.combine(patient.scanner_date, time(0, 0, 0))
                                    - problem.horizon_start + timedelta(days=8)
                            ).total_seconds()
                            / 60
                        )
                        # Get the number of minutes from start of horizon to the due_date of patient.
                        scenario_due_date_min = int(
                            (
                                    datetime.datetime.combine(patient.scanner_date, time(0, 0, 0))
                                    - problem.horizon_start + timedelta(days=20)
                            ).total_seconds()
                            / 60
                        )
                        ready_date = (datetime.datetime.combine(patient.scanner_date, time(0, 0, 0))
                                      + timedelta(days=8))
                        due_date = (datetime.datetime.combine(patient.scanner_date, time(0, 0, 0))
                                    + timedelta(days=20))
                        print(
                            f"Uncertain patient {patient.id} : real medical validation date is {patient.medical_validation_date}, "
                            f"in scenario {scenario} ready_date is {ready_date}, due_date is {due_date}."
                        )
                        ready_dates_by_scenario[scenario] = scenario_ready_date_min
                        due_dates_by_scenario[scenario] = scenario_due_date_min
                    elif scenario == 0 and self.one_scenario_strategy == "EARLIEST":
                        # In this case, for the first scenario we are being excessively optimistic , we assume that the
                        # patient has medical validation on this day and is being treated at best 48 hours later.
                        ready_date_ = patient.scanner_date + timedelta(days=2)
                        due_date_ = patient.due_date
                        scenario_ready_date_min = int(
                            (
                                    datetime.datetime.combine(ready_date_, time(0, 0, 0))
                                    - problem.horizon_start
                            ).total_seconds()
                            / 60
                        )
                        # Get the number of minutes from start of horizon to the due_date of patient.
                        scenario_due_date_min = int(
                            (
                                    datetime.datetime.combine(due_date_, time(0, 0, 0))
                                    - problem.horizon_start
                            ).total_seconds()
                            / 60
                        )
                        ready_date = (datetime.datetime.combine(ready_date_, time(0, 0, 0)))
                        due_date = (datetime.datetime.combine(due_date_, time(0, 0, 0)))
                        print(
                            f"Uncertain patient {patient.id} : real medical validation date is {patient.medical_validation_date}, "
                            f"in scenario {scenario} ready_date is {ready_date}, due_date is {due_date}."
                        )
                        ready_dates_by_scenario[scenario] = scenario_ready_date_min
                        due_dates_by_scenario[scenario] = scenario_due_date_min
                    else :
                        # HOW TO QUANTIFY UNCERTAINTY ?
                        delta = (patient.due_date - problem.horizon_start).days  # Number of days between today and due date
                        # print(f"Delta : {delta}, ready_date: {patient.ready_date}, due_date: {patient.due_date}"
                        #       f", scanner_date: {patient.scanner_date}, "
                        #       f"horizon_start (= current date): {problem.horizon_start}")
                        # Get the number of minutes from start of horizon to the ready_date of patient.
                        # TODO : Latin Hypercube to draw scenarios ?
                        variance_on_validation_date = int(
                            random.uniform(0, delta - 3))  # Minus the 48h before the patient is ready and minus 24 because in fact 13-02 0:00 == 12-02 0:00 -> 23h59
                        # print(f"Variance for scenario {scenario} of patient {patient.id}: {variance_on_validation_date}.")
                        scenario_medical_validation_date = (
                                problem.horizon_start + timedelta(
                            days=variance_on_validation_date))  # Horizon start = today
                        # Ready_date = medical validation date + 48h
                        ready_date_ = scenario_medical_validation_date + timedelta(days=2)
                        due_date_ = patient.due_date
                        # print(f"Ready date before treatment: {ready_date_}")
                        if ready_date_.weekday() > 4:
                            ready_date_ = ready_date_ + timedelta(days=2)
                            due_date_ = due_date_ + timedelta(days=2)
                        if ready_date_ >= due_date_:
                            print("CAREFUL READY_DATE > DUE_DATE")
                        scenario_ready_date_min = int(
                            (
                                    ready_date_ - problem.horizon_start
                            ).total_seconds()
                            / 60
                        )
                        # Get the number of minutes from start of horizon to the due_date of patient.
                        scenario_due_date_min = int(
                            (
                                    due_date_ - problem.horizon_start
                            ).total_seconds()
                            / 60
                        )
                        ready_dates_by_scenario[scenario] = scenario_ready_date_min
                        due_dates_by_scenario[scenario] = scenario_due_date_min
                        print(
                            f"Uncertain patient {patient.id} : real medical validation date is {patient.medical_validation_date}, "
                            f"in scenario {scenario} ready_date is {ready_date_}, due_date is {due_date_}."
                        )
                ready_dates_uncertain[patient.id] = ready_dates_by_scenario
                due_dates_uncertain[patient.id] = due_dates_by_scenario

        # =======================================================================
        # Build the durations according to patients locations
        # =======================================================================
        task_processing_time_data = {}
        for patient in problem.patients_queue:
            for t in range(patient.nb_fractions):
                durations_per_task = {linac.id: INFINITY for linac in problem.linacs}
                for linac in problem.linacs:
                    try:
                        time_linac = patient.location.duration_by_linac[linac]
                        durations_per_task[linac.id] = int(time_linac)
                    except KeyError:
                        durations_per_task[linac.id] = INFINITY
                task_processing_time_data[patient.id] = durations_per_task

        # =======================================================================
        # Preferred, accepted and forbidden linacs for each patient
        # =======================================================================
        preferences_by_patient = {}

        for patient in problem.patients_queue:
            linacs_prefs = {}

            for linac in patient.location.linacs_by_priority[Priority.FORBIDDEN]:
                linacs_prefs[linac.id] = 0
            for linac in patient.location.linacs_by_priority[Priority.ACCEPTED]:
                linacs_prefs[linac.id] = 1
            for linac in patient.location.linacs_by_priority[Priority.PREFERRED]:
                linacs_prefs[linac.id] = 2

            preferences_by_patient[patient.id] = linacs_prefs

        # Allocating pre-processed computed values to class attributes.
        self.new_tasks_count = sum(
            patient.nb_fractions for patient in problem.patients_queue
        )

        self.minutes_horizon = 1440 * problem.horizon
        self.ready_dates = ready_dates
        self.due_dates = due_dates
        self.ready_dates_uncertain = ready_dates_uncertain
        self.due_dates_uncertain = due_dates_uncertain
        self.task_processing_time_data = task_processing_time_data
        self.preferences_by_patient = preferences_by_patient

    def init_model(self, problem: Problem):
        session = collections.namedtuple("session", "is_used start end")

        # VARIABLES
        self.dict_certain_patient_vars = {
            patient.id: {
                # Not a patient variable but very convenient to retrieve the ids of authorized linacs, it also gives the
                # preference for each authorized machine.
                "machines_authorized": {
                    machine.id: self.preferences_by_patient[patient.id][machine.id]
                    for machine in problem.linacs
                    if self.preferences_by_patient[patient.id][machine.id]
                       >= 1
                       in self.preferences_by_patient[patient.id]
                },
                # This set of sessions variables (for a given patient) is created each session of the patient within
                # their number of fractions AND for each machine if the machine is at least accepted. These three
                # following variables are used to create an OptionalInterval : - one boolean variable that states if
                # the machine is used for this slot. - two integer variables, the first for the start of the interval
                # and the second for the end.
                "sessions": {
                    machine.id: [
                        session(
                            self.model.NewBoolVar(
                                f"is_used_machine_id{machine.id}_name{machine.name}_{s}"
                            ),
                            self.model.NewIntVar(
                                self.ready_dates[patient.id],
                                self.minutes_horizon,
                                f"session_{s}_start_patient{patient.id}_m{machine.id}",
                            ),
                            self.model.NewIntVar(
                                0,
                                self.minutes_horizon,
                                f"session_{s}_end_patient{patient.id}_m{machine.id}",
                            ),
                        )
                        for s in range(patient.nb_fractions)
                    ]
                    for machine in problem.linacs
                    if self.preferences_by_patient[patient.id][machine.id]
                       >= 1
                       in self.preferences_by_patient[patient.id]
                },
                "sessions_intervals": [
                    (
                        self.model.NewIntVar(
                            self.ready_dates[patient.id],
                            self.minutes_horizon,
                            f"start_{s}_of_{patient.id}",
                        ),  # 0: start
                        self.model.NewIntVar(
                            0, self.minutes_horizon, f"end_{s}_of_{patient.id}"
                        ),  # 1: end
                        self.model.NewIntVar(
                            0,
                            self.minutes_horizon,
                            f"size_interval_{s}_of_{patient.id}",
                        ),  # 2: size
                    )
                    for s in range(patient.nb_fractions)
                ],
                "treatment_range": self.model.NewIntVar(
                    # at least nb_fractions days are required if period >= 1
                    patient.nb_fractions - 1 if patient.location.period >= 1 else 0,
                    # the range cannot be greater than the scheduling horizon
                    problem.horizon,
                    f"treatment_range_of_patient_{patient.id}",
                ),
            }
            for patient in problem.patients_queue
            if patient.is_certain
        }

        self.dict_uncertain_patient_vars = {
            scenario: {
                patient.id: {
                    # Not a patient variable but very convenient to retrieve the ids of authorized linacs, it also gives the
                    # preference for each authorized machine.
                    "machines_authorized": {
                        machine.id: self.preferences_by_patient[patient.id][machine.id]
                        for machine in problem.linacs
                        if self.preferences_by_patient[patient.id][machine.id]
                           >= 1
                           in self.preferences_by_patient[patient.id]
                    },
                    # This set of sessions variables (for a given patient) is created each session of the patient within
                    # their number of fractions AND for each machine if the machine is at least accepted. These three
                    # following variables are used to create an OptionalInterval : - one boolean variable that states if
                    # the machine is used for this slot. - two integer variables, the first for the start of the interval
                    # and the second for the end.
                    "sessions": {
                        machine.id: [
                            session(
                                self.model.NewBoolVar(
                                    f"is_used_machine_id{machine.id}_name{machine.name}_{s}"
                                ),
                                self.model.NewIntVar(
                                    self.ready_dates_uncertain[patient.id][scenario],
                                    self.minutes_horizon,
                                    f"session_{s}_start_patient{patient.id}_m{machine.id}",
                                ),
                                self.model.NewIntVar(
                                    0,
                                    self.minutes_horizon,
                                    f"session_{s}_end_patient{patient.id}_m{machine.id}",
                                ),
                            )
                            for s in range(patient.nb_fractions)
                        ]
                        for machine in problem.linacs
                        if self.preferences_by_patient[patient.id][machine.id]
                           >= 1
                           in self.preferences_by_patient[patient.id]
                    },
                    "sessions_intervals": [
                        (
                            self.model.NewIntVar(
                                self.ready_dates_uncertain[patient.id][scenario],
                                self.minutes_horizon,
                                f"start_{s}_of_{patient.id}",
                            ),  # 0: start
                            self.model.NewIntVar(
                                0, self.minutes_horizon, f"end_{s}_of_{patient.id}"
                            ),  # 1: end
                            self.model.NewIntVar(
                                0,
                                self.minutes_horizon,
                                f"size_interval_{s}_of_{patient.id}",
                            ),  # 2: size
                        )
                        for s in range(patient.nb_fractions)
                    ],
                    # Delta period is the number of minutes between two sessions and their period.
                    # delta = (end_session_t - end_session_t-1) - period
                    "delta_period": [
                        self.model.NewIntVar(
                            0,
                            self.minutes_horizon,
                            f"delta_period_btw_sessions_{t - 1}_and_{t}",
                        )
                        for t in range(0, patient.nb_fractions - 1)
                    ],
                    # Delta period is the number of days between two sessions and their period.
                    "delta_period_days": [
                        self.model.NewIntVar(
                            0,
                            problem.horizon,
                            f"delta_period_days_btw_sessions_{t - 1}_and_{t}",
                        )
                        for t in range(0, patient.nb_fractions - 1)
                    ],
                    # Integer variable used to define a patient makespan (i.e. the date of the latest task for this
                    # patient).
                    "makespan": self.model.NewIntVar(
                        0, self.minutes_horizon, f"makespan_of_patient_{patient.id}"
                    ),
                    "treatment_range": self.model.NewIntVar(
                        # at least nb_fractions days are required if period >= 1
                        patient.nb_fractions - 1 if patient.location.period >= 1 else 0,
                        # the range cannot be greater than the scheduling horizon
                        problem.horizon,
                        f"treatment_range_of_patient_{patient.id}",
                    ),
                }
                for patient in problem.patients_queue
                if not patient.is_certain
            }
            for scenario in range(self.nb_scenarios)
        }

        self.dict_machine_vars = {
            linac.id: {
                "unavailability_slots": compute_unavailable_slots_by_machine(
                    existing_schedule=problem.existing_schedule,
                    start_horizon=problem.horizon_start,
                    days_horizon=problem.horizon,
                    model=self.model,
                    machine_id=linac.id,
                ),
            }
            for linac in problem.linacs
        }

        for p, pvars in self.dict_certain_patient_vars.items():
            # REAL SESSIONS INTERVALS for certain patients
            for s_id, session in enumerate(pvars["sessions_intervals"]):
                session_interval = self.model.NewIntervalVar(
                    start=session[0],
                    size=session[2],
                    end=session[1],
                    name=f"session_interval_{s_id}_of_patient_{p}",
                )
                # noinspection PyTypeChecker
                self.dict_certain_patient_vars[p]["sessions_intervals"][s_id] = (
                        session + (session_interval,)
                )
        # Interval variable creation for patients, we add the variables created below to
        # the dict dict_patient_vars_by_scenario.
        for scenario in range(self.nb_scenarios):
            for p, s_pvars in self.dict_uncertain_patient_vars[scenario].items():
                # REAL SESSIONS INTERVALS for uncertain patients
                for s_id, session in enumerate(s_pvars["sessions_intervals"]):
                    session_interval = self.model.NewIntervalVar(
                        start=session[0],
                        size=session[2],
                        end=session[1],
                        name=f"session_interval_{s_id}_of_patient_{p}",
                    )
                    # noinspection PyTypeChecker
                    self.dict_uncertain_patient_vars[scenario][p]["sessions_intervals"][
                        s_id
                    ] = session + (session_interval,)

    # add_constraints abstract

    # add_objectives abstract

    def solve(self, problem: Problem, fix_seed: bool = False):
        self.logger.info(f"Start solving...")
        self.preprocess_problem_data(problem)
        self.init_model(problem)
        self.add_constraints(problem)
        self.add_objectives(problem)

        self.solver = cp_model.CpSolver()
        self.logger.info(f"Solver parameter max_time_in_seconds = {problem.solver_time_limit}")
        self.solver.parameters.max_time_in_seconds = problem.solver_time_limit
        # Fix seed to make the solver deterministic.
        if fix_seed:
            self.solver.parameters.random_seed = 42
        self.solution_callback = SolutionCallback(
            dict_certain_patient_vars=self.dict_certain_patient_vars,
            dict_uncertain_patient_vars=self.dict_uncertain_patient_vars,
        )
        self.status = self.solver.Solve(self.model, self.solution_callback)
        self.logger.info("Solving Done !")
        # self.kpi_df = self.solution_callback.kpi_data

    def retrieve_solution(self, problem: Problem) -> MultiStageSolution:
        # Return the solution.
        t0 = perf_counter()
        certain_schedule = []
        uncertain_schedule_by_scenario = {s: [] for s in range(self.nb_scenarios)}
        machines_occupancy = {day:
                                  {linac.id: 0 for linac in problem.linacs}
                              for day in range(problem.horizon)
                              }

        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            local_date = problem.horizon_start
            # Retrieve schedules for certain patients first.
            for patient in problem.patients_queue:
                if patient.is_certain:
                    # print(f"Certain patient {patient.id} schedule is :")
                    pvars = self.dict_certain_patient_vars[patient.id]
                    # sessions_ = sorted([(solver.Value(session[0]), solver.Value(session[1]))
                    #                     for session in pvars['sessions_intervals']], key=lambda x: x[0])
                    # We need to sort the sessions by start date.
                    sessions = sorted(
                        [
                            (
                                self.solver.Value(session.start),
                                self.solver.Value(session.end),
                                m,
                            )
                            for m in pvars["machines_authorized"].keys()
                            for s_id, session in enumerate(pvars["sessions"][m])
                            if self.solver.Value(session.is_used)
                        ],
                        key=lambda x: x[0],
                    )
                    # print(self.solver.Value(self.dict_certain_patient_vars[patient.id]["first_day_of_treatment"]))
                    # print(solver.Value(dict_patient_vars[patient.id]["last_day_of_treatment"]))
                    # print(solver.Value(dict_patient_vars[patient.id]["treatment_range"]))
                    for session in sessions:
                        # print(session)
                        # TODO = Warning linacs_ids heavily depends on oncopole_linacs, should be changed to use machine
                        # TODO = id instead of position in list -> many such usages in checker.
                        resource = linacs_ids[session[2]]
                        # resource_name = linacs_ids[session[2]]
                        # print(f"Machine id {resource}"
                        #       f" - start = {session[0]} and end = {session[1]}")
                        start = local_date + timedelta(minutes=session[0])
                        end = local_date + timedelta(minutes=session[1])
                        session_day_from_horizon_start = (start - local_date).days
                        machines_occupancy[session_day_from_horizon_start][resource] += (session[1] - session[0])
                        # TODO : check if delta is correct.
                        # print(session_day_from_horizon_start)
                        # print(f"{start} - {end} - {resource}")
                        certain_schedule.append((patient, resource, (start, end)))
            # Then retrieve schedules for uncertain patients.
            for patient in problem.patients_queue:
                if not patient.is_certain:
                    for scenario in range(self.nb_scenarios):
                        # We need to add first the schedule of certain patients in each scenario.
                        uncertain_schedule_by_scenario[scenario].extend(certain_schedule)
                        # print(
                        #     f"Uncertain patient {patient.id} in Scenario {scenario} schedule is (ready_date = "
                        #     f"{local_date + timedelta(minutes=self.ready_dates_uncertain[patient.id][scenario])}) :"
                        # )
                        s_pvars = self.dict_uncertain_patient_vars[scenario][patient.id]
                        # print(f"Patient {patient.id} in scenario {scenario} has {self.solver.Value(s_pvars["delta_to_due_date"])} delta to due date.")
                        # We need to sort the sessions by start date.
                        sessions = sorted(
                            [
                                (
                                    self.solver.Value(session.start),
                                    self.solver.Value(session.end),
                                    m,
                                )
                                for m in s_pvars["machines_authorized"].keys()
                                for s_id, session in enumerate(s_pvars["sessions"][m])
                                if self.solver.Value(session.is_used)
                            ],
                            key=lambda x: x[0],
                        )
                        for session in sessions:
                            # print(session)
                            resource = linacs_ids[session[2]]
                            # machines_occupancy[resource] += (session[1] - session[0])
                            # print(f"Machine id {resource}"
                            #       f" - start = {session[0]} and end = {session[1]}")
                            start = local_date + timedelta(minutes=session[0])
                            end = local_date + timedelta(minutes=session[1])
                            # print(f"{start} - {end} - {resource}")
                            uncertain_schedule_by_scenario[scenario].append(
                                (patient, resource, (start, end))
                            )
            print(f"================== Done ==================")
            self.objective_first_stage_earliest = self.solver.Value(self.first_stage_earliest_var) * \
                                             problem.problem_parameters.objective_weights["earliest"]
            self.objective_first_stage_treatment_range = self.solver.Value(self.first_stage_treatment_range_var)*problem.problem_parameters.objective_weights["treatment_range"]
            self.objective_second_stage_earliest = self.solver.Value(self.second_stage_earliest_var)*math.ceil(problem.problem_parameters.objective_weights["earliest"]*(1-self.first_stage_weight_proportion))
            self.objective_second_stage_treatment_range = self.solver.Value(self.second_stage_treatment_range_var)*math.ceil(problem.problem_parameters.objective_weights["treatment_range"]*(1-self.first_stage_weight_proportion))
            self.objective_second_stage_due_dates_penalties = self.solver.Value(self.max_due_dates_penalties)

            print(f"Objectives :")
            print(f"\t - First stage objective \"earliest\" : {self.objective_first_stage_earliest}")
            print(f"\t - First Stage objective \"treatment range\" : {self.objective_first_stage_treatment_range}")
            print(f"\t - Second Stage objective \"earliest\" : {self.objective_second_stage_earliest}")
            print(f"\t - Second Stage objective \"treatment range\" : {self.objective_second_stage_treatment_range}")
            print(f"\t - Second Stage \"due dates penalties\" : {self.objective_second_stage_due_dates_penalties}")


            print(f"\t - Sum is {
                self.objective_first_stage_earliest
                + self.objective_first_stage_treatment_range
                + self.objective_second_stage_earliest
                + self.objective_second_stage_treatment_range
                + self.objective_second_stage_due_dates_penalties
            }")
            # print(f"Machine occupancy : {machines_occupancy}")
            print(f"\t - Number of tasks : {self.new_tasks_count}")
            print("Solver : ")
            print(f"\t - Status : {self.solver.StatusName(self.status)}")
            print(f"\t - Objective value : {self.solver.ObjectiveValue()}")
            print(f"\t - Best bound : {self.solver.BestObjectiveBound()}")
            print(
                f"\t - Gap : {100 * (self.solver.ObjectiveValue() / self.solver.BestObjectiveBound() - 1 if self.solver.BestObjectiveBound() > 0 else 0):.2f} %"
            )
            print(
                f"\t - Number of solutions found : {self.solution_callback.solution_count}"
            )
        elif self.status == cp_model.MODEL_INVALID:
            print("Invalid model.")
        else:
            print(f"Status = {self.solver.StatusName(self.status)}")
        print("Statistics : ")
        print(f"\t - conflicts : {self.solver.NumConflicts()}")
        print(f"\t - branches  : {self.solver.NumBranches()}")
        print(f"\t - wall time : {self.solver.WallTime():.2f}s")
        print(f"\t - retrieve time : {perf_counter() - t0:.2f} s")

        solution = MultiStageSolution(problem=problem, nb_scenarios=self.nb_scenarios)
        solution.certain_patients = [patient for patient in problem.patients_queue if patient.is_certain]
        solution.uncertain_patients = [patient for patient in problem.patients_queue if not patient.is_certain]
        solution.machine_occupancy = machines_occupancy

        solution.certain_patients_schedule = certain_schedule
        # print(certain_schedule)
        solution.dict_schedules = uncertain_schedule_by_scenario
        solution.existing_schedule = merged_schedule_to_list(
            linacs=problem.linacs,
            horizon_start=problem.horizon_start,
            existing_schedule=problem.existing_schedule,
        )

        solution.statistics["objective_value"] = self.solver.ObjectiveValue()
        solution.statistics["objective_first_stage_earliest"] = self.objective_first_stage_earliest
        solution.statistics["objective_first_stage_treatment_range"] = self.objective_first_stage_treatment_range
        solution.statistics["objective_second_stage_earliest"] = self.objective_second_stage_earliest
        solution.statistics["objective_second_stage_treatment_range"] = self.objective_second_stage_treatment_range
        solution.statistics["objective_second_stage_due_dates_penalties"] = self.objective_second_stage_due_dates_penalties
        solution.statistics["gap"] = 100 * (
            self.solver.ObjectiveValue() / self.solver.BestObjectiveBound() - 1 if self.solver.BestObjectiveBound() > 0 else 0)
        solution.statistics["solver_status"] = str(self.solver.StatusName(self.status))
        solution.statistics["solver_time"] = self.solver.WallTime()
        return solution


class SolverCPSAT2Stages(SolverCPSAT):

    def __init__(self,
                 nb_scenarios: int,
                 first_stage_weight_proportion: int,
                 aggregation_criterion: str = "MAX",
                 one_scenario_strategy: str = "AVERAGE_SCENARIO",
                ):
        super().__init__(
            name=__name__,
            nb_scenarios=nb_scenarios,
            first_stage_weight_proportion=first_stage_weight_proportion,
            aggregation_criterion=aggregation_criterion,
            one_scenario_strategy=one_scenario_strategy,
        )

    def preprocess_problem_data(self, problem: Problem):
        super().preprocess_problem_data(problem)

    def init_model(self, problem: Problem):
        super().init_model(problem)

    def add_constraints(self, problem: Problem):
        interval_session = collections.namedtuple(
            "interval_session", "is_used start end interval"
        )

        # OPTIONAL INTERVALS for certain patients
        for p, pvars in self.dict_certain_patient_vars.items():
            for m, _ in pvars["machines_authorized"].items():
                for s_id, session_schedules in enumerate(pvars["sessions"][m]):
                    slot_duration = int(self.task_processing_time_data[p][m])
                    interval = self.model.NewOptionalIntervalVar(
                        start=session_schedules.start,
                        size=slot_duration,
                        end=session_schedules.end,
                        is_present=session_schedules.is_used,
                        name=f"session_{s_id}_of_patient_{p}_on_machine{m}",
                    )
                    # Here we link the optionals intervals to the sessions intervals, the latter will be used to
                    # enforce the periods constraints, otherwise bugs when we change machines.
                    self.model.Add(
                        self.dict_certain_patient_vars[p]["sessions_intervals"][s_id][0]
                        == session_schedules.start
                    ).OnlyEnforceIf(session_schedules.is_used)
                    self.model.Add(
                        self.dict_certain_patient_vars[p]["sessions_intervals"][s_id][1]
                        == session_schedules.end
                    ).OnlyEnforceIf(session_schedules.is_used)
                    self.model.Add(
                        self.dict_certain_patient_vars[p]["sessions_intervals"][s_id][2]
                        == slot_duration
                    ).OnlyEnforceIf(session_schedules.is_used)
                    # The comma after interval is to create a tuple of only one element and return a tuple of n+1 elems.
                    # We add the optional interval to the "sessions" in patients_vars.
                    session_schedules = interval_session(
                        *session_schedules + (interval,)
                    )
                    # Don't forget to update the list of schedules.
                    self.dict_certain_patient_vars[p]["sessions"][m][
                        s_id
                    ] = session_schedules

        for scenario in range(self.nb_scenarios):
            for p, s_pvars in self.dict_uncertain_patient_vars[scenario].items():
                for m, _ in s_pvars["machines_authorized"].items():
                    for s_id, session_schedules in enumerate(s_pvars["sessions"][m]):
                        slot_duration = int(self.task_processing_time_data[p][m])
                        interval = self.model.NewOptionalIntervalVar(
                            start=session_schedules.start,
                            size=slot_duration,
                            end=session_schedules.end,
                            is_present=session_schedules.is_used,
                            name=f"session_{s_id}_of_patient_{p}_on_machine{m}",
                        )
                        # Here we link the optionals intervals to the sessions intervals, the latter will be used to
                        # enforce the periods constraints, otherwise bugs when we change machines.
                        self.model.Add(
                            self.dict_uncertain_patient_vars[scenario][p][
                                "sessions_intervals"
                            ][s_id][0]
                            == session_schedules.start
                        ).OnlyEnforceIf(session_schedules.is_used)
                        self.model.Add(
                            self.dict_uncertain_patient_vars[scenario][p]["sessions_intervals"][s_id][1]
                            == session_schedules.end
                        ).OnlyEnforceIf(session_schedules.is_used)
                        self.model.Add(
                            self.dict_uncertain_patient_vars[scenario][p]["sessions_intervals"][s_id][2]
                            == slot_duration
                        ).OnlyEnforceIf(session_schedules.is_used)
                        # The comma after interval is to create a tuple of only one element and return a tuple of n+1 elems.
                        # We add the optional interval to the "sessions" in patients_vars.
                        session_schedules = interval_session(
                            *session_schedules + (interval,)
                        )
                        # Don't forget to update the list of schedules.
                        self.dict_uncertain_patient_vars[scenario][p]["sessions"][m][
                            s_id
                        ] = session_schedules

        # The following dict allows to retrieve the boolean variable indicating if the slot is used for a given machine,
        # for each authorized machine, for each slot and for each patient.
        machines_by_session = {}
        for patient, pvars in self.dict_certain_patient_vars.items():
            for m in pvars["machines_authorized"].keys():
                machines_by_session[patient] = [
                    [
                        pvars["sessions"][m][i].is_used
                        for m in pvars["machines_authorized"].keys()
                    ]
                    for i in range(len(pvars["sessions"][m]))
                ]

                # CONSTRAINTS : Order the slots.
                for ms in pvars["sessions"].values():
                    for i in range(len(ms) - 1):
                        self.model.Add(ms[i].end <= ms[i + 1].start)

                sessions = pvars["sessions_intervals"]
                for i in range(len(sessions) - 1):
                    self.model.Add(sessions[i][1] <= sessions[i + 1][0])

                # CONSTRAINTS : ready dates and due dates for a given patient.
                self.model.Add(
                    pvars["sessions"][m][0].start >= self.ready_dates[patient]
                ).OnlyEnforceIf(pvars["sessions"][m][0].is_used)
                self.model.Add(
                    pvars["sessions"][m][0].start
                    <= self.due_dates[
                        patient
                    ]  # ready_dates[patient]+20*60 #+due_dates[patient]
                ).OnlyEnforceIf(pvars["sessions"][m][0].is_used)

            # CONSTRAINTS : only one machine can be selected at a time for one patient
            self.model.AddExactlyOne(machines_by_session[patient][0])
            for k in range(1, len(machines_by_session[patient])):
                for m in range(len(pvars["machines_authorized"])):
                    self.model.Add(
                        machines_by_session[patient][k][m] == 1
                    ).OnlyEnforceIf(machines_by_session[patient][0][m])
                    self.model.Add(
                        machines_by_session[patient][k][m] == 0
                    ).OnlyEnforceIf(machines_by_session[patient][0][m].Not())

        # CONSTRAINTS for uncertain patients
        for scenario in range(self.nb_scenarios):
            machines_by_session = {}
            for patient, s_pvars in self.dict_uncertain_patient_vars[scenario].items():
                for m in s_pvars["machines_authorized"].keys():
                    machines_by_session[patient] = [
                        [
                            s_pvars["sessions"][m][i].is_used
                            for m in s_pvars["machines_authorized"].keys()
                        ]
                        for i in range(len(s_pvars["sessions"][m]))
                    ]

                    # CONSTRAINTS : Order the slots.
                    for ms in s_pvars["sessions"].values():
                        for i in range(len(ms) - 1):
                            self.model.Add(ms[i].end <= ms[i + 1].start)

                    sessions = s_pvars["sessions_intervals"]
                    for i in range(len(sessions) - 1):
                        self.model.Add(sessions[i][1] <= sessions[i + 1][0])

                    # CONSTRAINTS : ready dates and due dates for a given patient.
                    self.model.Add(
                        s_pvars["sessions"][m][0].start
                        >= self.ready_dates_uncertain[patient][scenario]
                    ).OnlyEnforceIf(s_pvars["sessions"][m][0].is_used)
                    # self.model.Add(
                    #     s_pvars["sessions"][m][0].start
                    #     <= self.due_dates_uncertain[patient][scenario]
                    # ).OnlyEnforceIf(s_pvars["sessions"][m][0].is_used)

                # CONSTRAINTS : only one machine can be selected at a time for one patient
                self.model.AddExactlyOne(machines_by_session[patient][0])
                for k in range(1, len(machines_by_session[patient])):
                    for m in range(len(s_pvars["machines_authorized"])):
                        self.model.Add(
                            machines_by_session[patient][k][m] == 1
                        ).OnlyEnforceIf(machines_by_session[patient][0][m])
                        self.model.Add(
                            machines_by_session[patient][k][m] == 0
                        ).OnlyEnforceIf(machines_by_session[patient][0][m].Not())

        # CONSTRAINTS : no overlapping between tasks on same machine.
        for linac in problem.linacs:
            certain_sessions = []
            for pvars in self.dict_certain_patient_vars.values():
                if linac.id in pvars["machines_authorized"]:
                    for session in pvars["sessions"][linac.id]:
                        certain_sessions.append(session.interval)
            for scenario in range(self.nb_scenarios):
                uncertain_sessions_by_scenario = []
                for s_pvars in self.dict_uncertain_patient_vars[scenario].values():
                    if linac.id in s_pvars["machines_authorized"]:
                        for session in s_pvars["sessions"][linac.id]:
                            uncertain_sessions_by_scenario.append(session.interval)
                self.model.AddNoOverlap(
                    certain_sessions
                    + uncertain_sessions_by_scenario
                    + self.dict_machine_vars[linac.id]["unavailability_slots"][0]
                )
                self.model.AddNoOverlap(
                    certain_sessions
                    + uncertain_sessions_by_scenario
                    + self.dict_machine_vars[linac.id]["unavailability_slots"][1]
                )

        # CONSTRAINTS : periods enforcement
        for patient in problem.patients_queue:
            if patient.is_certain:
                pvars = self.dict_certain_patient_vars[patient.id]
                period = patient.location.period

                sessions = pvars["sessions_intervals"]
                for i in range(len(sessions) - 1):
                    self.model.Add(
                        sessions[i + 1][0] - sessions[i][1]
                        >= int(
                            (((period - 1) * 24 + 12) if period >= 1 else (period * 12))
                            * 60
                        )
                    )
                    # Not allowing delays of more than 3 days for period 1 and 4 in other cases.
                    # if period <= 1:
                    #     self.model.Add(sessions[i + 1][0] - sessions[i][1] <= 1440 * 3)
                    # else:
                    #     self.model.Add(sessions[i + 1][0] - sessions[i][1] <= 1440 * 4)
            else:
                for scenario in range(self.nb_scenarios):
                    s_pvars = self.dict_uncertain_patient_vars[scenario][patient.id]
                    period = patient.location.period

                    sessions = s_pvars["sessions_intervals"]
                    for i in range(len(sessions) - 1):
                        self.model.Add(
                            sessions[i + 1][0] - sessions[i][1]
                            >= int(
                                (
                                    ((period - 1) * 24 + 12)
                                    if period >= 1
                                    else (period * 12)
                                )
                                * 60
                            )
                        )
                        self.model.Add(
                            s_pvars["delta_period"][i]
                            == sessions[i + 1][0] - sessions[i][1]
                        )
                    
    def add_objective_first_stage_earliest(self, problem: Problem):
        # Create "first_day_of_treatment" variable for each certain patient, then link it to the value of the interval variable
        # value of the first session.
        for patient_id, pvars in self.dict_certain_patient_vars.items():
            pvars["first_day_of_treatment"] = (
                self.model.NewIntVar(
                    self.ready_dates[patient_id] % 1440,  # first day of treatment cannot be earlier than release date
                    problem.horizon,
                    name=f"startday_{patient_id}",
                )
            )
            self.model.AddDivisionEquality(
                pvars["first_day_of_treatment"],
                pvars["sessions_intervals"][0][0] - self.ready_dates[patient_id],
                # the start of the first session.
                #
                denom=1440,
            )

        # Create "earliest" objective for certain patients.
        earliest = cp_model.LinearExpr.Sum(
            [
                (4 - patient.location.urgency.value)
                * self.dict_certain_patient_vars[patient.id]["first_day_of_treatment"]
                for patient in problem.patients_queue
                if patient.is_certain
            ]
        )

        self.first_stage_earliest_var = self.model.NewIntVar(
            0,
            # self.ready_dates_uncertain[patient_id][scenario] % 1440,  # first day of treatment cannot be earlier than release date
            100000,
            name=f"first_stage_earliest_var")
        self.model.Add(
            self.first_stage_earliest_var == earliest,
        )

    def add_objective_second_stage_earliest(self, problem: Problem, aggregation_criterion: str = "MAX"):
        # Create "first_day_of_treatment" variable for each uncertain patient, in each scenario, then link it to the value of the
        # interval variable value of the first session.
        scenarios_earliest = []
        for scenario in range(self.nb_scenarios):
            for patient_id in self.dict_uncertain_patient_vars[scenario]:
                self.dict_uncertain_patient_vars[scenario][patient_id][
                    "first_day_of_treatment"
                ] = self.model.NewIntVar(
                    0,
                    # self.ready_dates_uncertain[patient_id][scenario] % 1440,  # first day of treatment cannot be earlier than release date
                    problem.horizon,
                    name=f"startday_{patient_id}",
                )
                self.model.AddDivisionEquality(
                    self.dict_uncertain_patient_vars[scenario][patient_id]["first_day_of_treatment"],
                    self.dict_uncertain_patient_vars[scenario][patient_id]["sessions_intervals"][0][0],
                    # the start of the first session.
                    # - ready_dates[patient_id]
                    denom=1440,
                )
            # Create "earliest" objective for each scenario, as sum of first_day_of_treatment over all uncertain
            # patients.
            scenario_earliest = cp_model.LinearExpr.Sum(
                [
                    (4 - patient.location.urgency.value)
                    * self.dict_uncertain_patient_vars[scenario][patient.id][
                        "first_day_of_treatment"
                    ]
                    for patient in problem.patients_queue
                    if not patient.is_certain
                ]
            )
            scenarios_earliest.append(scenario_earliest)

        # We need to create a variable to use AddMaxEquality over all scenarios (uncertain patients).
        self.second_stage_earliest_var = self.model.NewIntVar(
            0, 10000, name="second_stage_earliest_var"
        )
        if aggregation_criterion == "MAX":
            self.model.AddMaxEquality(self.second_stage_earliest_var, scenarios_earliest)
        elif aggregation_criterion == "AVG":
            self.second_stage_earliest_var = (1/self.nb_scenarios)*cp_model.LinearExpr.Sum(
                scenarios_earliest
            )

    def add_objective_first_stage_period_enforcement(self, problem: Problem):
        """
        This function allows to add period enforcement as an objective for the first stage, i.e. certain patients only.
        :param problem: Problem
        :return: cp_model.LinearExpr
        """
        for patient_id, pvars in self.dict_certain_patient_vars.items():
            # Add the variable "last_day_of_treatment" to the dict "dict_certain_patient_vars"
            pvars["last_day_of_treatment"] = (
                self.model.NewIntVar(
                    self.ready_dates[patient_id] % 1440,  # last day of treatment cannot be earlier than release date
                    problem.horizon,
                    name=f"endday_{patient_id}",
                )
            )
            self.model.AddDivisionEquality(
                pvars["last_day_of_treatment"],
                pvars["sessions_intervals"][-1][0],  # the start of the last session.
                denom=1440,
            )
            # Variable "first_day_of_treatment" should always be defined.
            self.model.Add(
                pvars["last_day_of_treatment"] - pvars["first_day_of_treatment"] == pvars["treatment_range"]
            )
        treatment_range = cp_model.LinearExpr.Sum(
            [pvars["treatment_range"] for pvars in self.dict_certain_patient_vars.values()]
        )

        self.first_stage_treatment_range_var = self.model.NewIntVar(
            0,
            # self.ready_dates_uncertain[patient_id][scenario] % 1440,  # first day of treatment cannot be earlier than release date
            100000,
            name=f"first_stage_treatment_range_var")
        self.model.Add(
            self.first_stage_treatment_range_var == treatment_range,
        )

    def add_objective_second_stage_period_enforcement(self, problem: Problem, aggregation_criterion: str = "MAX"):
        """
       This function allows to add period enforcement as an objective for the first stage, i.e. certain patients only.
       :param problem: Problem
       :param aggregation_criterion: str
       :return: cp_model.LinearExpr
        """
        scenario_treatment_ranges = []
        for scenario in range(self.nb_scenarios):
            for patient_id, s_pvars in self.dict_uncertain_patient_vars[scenario].items():
                # Add the variable "last_day_of_treatment" to the dict "dict_certain_patient_vars"
                s_pvars["last_day_of_treatment"] = (
                    self.model.NewIntVar(
                        0,  # last day of treatment cannot be earlier than release date
                        problem.horizon,
                        name=f"endday_{patient_id}",
                    )
                )
                self.model.AddDivisionEquality(
                    s_pvars["last_day_of_treatment"],
                    s_pvars["sessions_intervals"][-1][0],  # the start of the last session.
                    denom=1440,
                )
                # Variable "first_day_of_treatment" should always be defined.
                self.model.Add(
                    s_pvars["last_day_of_treatment"] - s_pvars["first_day_of_treatment"] == s_pvars["treatment_range"]
                )
            scenario_treatment_ranges.append(cp_model.LinearExpr.Sum(
                [s_pvars["treatment_range"] for s_pvars in self.dict_uncertain_patient_vars[scenario].values()]
            ))

        self.second_stage_treatment_range_var = self.model.NewIntVar(
            0, 10000, name="second_stage__treatment_range_var"
        )
        if aggregation_criterion == "MAX":
            self.model.AddMaxEquality(self.second_stage_treatment_range_var, scenario_treatment_ranges)
        elif aggregation_criterion == "AVG":
            self.second_stage_treatment_range_var = (1/self.nb_scenarios)*cp_model.LinearExpr.Sum(
                scenario_treatment_ranges
            )

    def add_objective_first_stage_machine_preferences(self):
        """
        This function allows to add machine preferences as an objective for the first stage, i.e. certain patients only.
        :return: cp_model.LinearExpr
        """
        machine_preference = cp_model.LinearExpr.Sum(
            [
                pvars["sessions"][machine_id][session_id].is_used
                * (2 - self.preferences_by_patient[patient_id][machine_id])
                for patient_id, pvars in self.dict_certain_patient_vars.items()
                for machine_id, _ in pvars["machines_authorized"].items()
                for session_id, session_schedules in enumerate(pvars["sessions"][machine_id])
            ]
        )

        return machine_preference

    def add_objective_first_stage_time_preferences(self, problem: Problem):
        sum_deltas_to_time_preference = []
        for patient in problem.patients_queue:
            if patient.is_certain:
                pvars = self.dict_certain_patient_vars[patient.id]
                sessions = pvars["sessions_intervals"]
                # Create
                pvars["delta_to_time_preference"] = [
                    self.model.NewIntVar(
                        0, 24 * 60, name=f"delta_to_time_preference_{patient.id}_{i}"
                    )
                    for i in range(len(sessions))
                ]
                # Why this variable and the following constraint :
                pvars["sessions_timing"] = [
                    self.model.NewIntVar(
                        0, 24 * 60, name=f"patient_{patient.id}_session_{i}"
                    )
                    for i in range(len(sessions))
                ]
                for i in range(len(sessions)):
                    self.model.AddModuloEquality(
                        pvars["sessions_timing"][i], sessions[i][0], mod=1440
                    )
                nb_sessions = len(pvars["sessions_timing"])

                if patient.preferred_timerange == PreferredTimerange.EARLY_MORNING:
                    for i in range(nb_sessions):
                        self.model.Add(
                            pvars["sessions_timing"][i]
                            <= 540 + pvars["delta_to_time_preference"][i]
                        )
                elif patient.preferred_timerange == PreferredTimerange.MORNING:
                    for i in range(nb_sessions):
                        self.model.Add(
                            pvars["sessions_timing"][i]
                            >= 540 - pvars["delta_to_time_preference"][i]
                        )
                        self.model.Add(
                            pvars["sessions_timing"][i]
                            <= 780 + pvars["delta_to_time_preference"][i]
                        )
                elif patient.preferred_timerange == PreferredTimerange.AFTERNOON:
                    for i in range(nb_sessions):
                        self.model.Add(
                            pvars["sessions_timing"][i]
                            >= 780 - pvars["delta_to_time_preference"][i]
                        )
                        self.model.Add(
                            pvars["sessions_timing"][i]
                            <= 1080 + pvars["delta_to_time_preference"][i]
                        )
                elif patient.preferred_timerange == PreferredTimerange.EVENING:
                    for i in range(nb_sessions):
                        self.model.Add(
                            pvars["sessions_timing"][i]
                            >= 1080 - pvars["delta_to_time_preference"][i]
                        )

                sum_local_deltas = cp_model.LinearExpr.Sum(
                    pvars["delta_to_time_preference"]
                )
                sum_deltas_to_time_preference.append(sum_local_deltas)
        return sum_deltas_to_time_preference

    def add_objective_second_stage_due_date_violations(self, problem: Problem, aggregation_criterion: str = "MAX"):
        """

        """
        due_dates_penalties = []
        for scenario in range(self.nb_scenarios):
            for patient_id, s_pvars in self.dict_uncertain_patient_vars[scenario].items():
                s_pvars["delta_to_due_date"] = (
                    self.model.NewIntVar(
                        0,
                        # first day of treatment cannot be earlier than release date
                        problem.horizon,
                        name=f"delta_to_due_date_s_{scenario}_pid_{patient_id}",
                    )
                )
                self.model.AddMaxEquality(
                    s_pvars["delta_to_due_date"],
                    [0, s_pvars["first_day_of_treatment"]-self.due_dates_uncertain[patient_id][scenario]])

            due_dates_penalties_scenario = cp_model.LinearExpr.Sum(
                [s_pvars["delta_to_due_date"]
                for s_pvars in self.dict_uncertain_patient_vars[scenario].values()]
            )
            due_dates_penalties.append(due_dates_penalties_scenario)
        self.max_due_dates_penalties = self.model.NewIntVar(
            0, 1000, name=f"max_due_dates_penalties"
        )
        if aggregation_criterion == "MAX":
            self.model.AddMaxEquality(self.max_due_dates_penalties, due_dates_penalties)
        elif aggregation_criterion == "AVG":
            self.max_due_dates_penalties = (1/self.nb_scenarios)*cp_model.LinearExpr.Sum(
                due_dates_penalties
            )

    def add_objective_machine_balance(self, problem: Problem):
        """
        First and second stage machine balance.
        """
        # Machine tasks assignment count, this variable is used to count the number of tasks allocated on a given
        # machine.
        task_assigned_linac_by_scenario = {s:
            {
                linac.id: self.model.NewIntVar(
                    # the lower bound cannot be less than the occupancy count on the given linac.
                    self.occupancy_count_by_linac[linac.id],
                    # the upper bound is up to the occupancy count on the given linac + the count of all new tasks.
                    self.occupancy_count_by_linac[linac.id] + self.new_tasks_count,
                    f"tasks_assigned_{linac.id}",
                )
                for linac in problem.linacs
            }
            for s in range(self.nb_scenarios)
        }
        #
        epsilon_scenario = {scenario:
            self.model.NewIntVar(
                0, self.total_occupancy_count + self.new_tasks_count, f"epsilon_{scenario}"
            )
            for scenario in range(self.nb_scenarios)
        }
        # CONSTRAINTS : Machine balance count.
        for linac in problem.linacs:
            for scenario in range(self.nb_scenarios):
                self.model.Add(
                    sum(
                        sum(
                            sum(
                                sessions[i].is_used
                                for i in range(len(sessions))
                                if machine == linac.id
                            )
                            for machine, sessions in pvars["sessions"].items()
                        )
                        for patient, pvars in self.dict_certain_patient_vars.items()
                    )
                    +
                    sum(
                        sum(
                            sum(
                                sessions[i].is_used
                                for i in range(len(sessions))
                                if machine == linac.id
                            )
                            for machine, sessions in pvars["sessions"].items()
                        )
                        for patient, pvars in self.dict_uncertain_patient_vars[scenario].items()
                    )
                    + self.occupancy_count_by_linac[linac.id]
                    == task_assigned_linac_by_scenario[scenario][linac.id]
                )
                self.model.Add(
                    task_assigned_linac_by_scenario[scenario][linac.id]
                    <= self.average_occupancy + epsilon_scenario[scenario]
                )
                self.model.Add(
                    task_assigned_linac_by_scenario[scenario][linac.id]
                    >= self.average_occupancy - epsilon_scenario[scenario]
                )
        return epsilon_scenario

    def add_objectives(self, problem: Problem):

        self.add_objective_first_stage_earliest(problem)
        self.add_objective_second_stage_earliest(problem, aggregation_criterion=self.aggregation_criterion)
        self.add_objective_first_stage_period_enforcement(problem)
        self.add_objective_second_stage_period_enforcement(problem, aggregation_criterion=self.aggregation_criterion)
        self.add_objective_second_stage_due_date_violations(problem, aggregation_criterion=self.aggregation_criterion)

        # TODO: add this in the method directly.
        # machine_balance_by_scenario = self.add_objective_machine_balance(problem)
        # max_machine_balance = self.model.NewIntVar(
        #     0, 10000, name="max_machine_balance"
        # )
        # self.model.AddMaxEquality(max_machine_balance, machine_balance_by_scenario)


        # machine_preference = self.add_objective_first_stage_machine_preferences()
        # time_preference = self.add_objective_first_stage_time_preferences(problem)
        # sum_time_preference = cp_model.LinearExpr.Sum(
        #     time_preference
        # )

        objectives = {
            "earliest": (
                self.first_stage_earliest_var,
                problem.problem_parameters.objective_weights["earliest"],
            ),
            "max_uncertain_earliest": (
                self.second_stage_earliest_var,
                math.ceil(problem.problem_parameters.objective_weights["earliest"]*(1-self.first_stage_weight_proportion)),
            ),
            "treatment_range": (
                self.first_stage_treatment_range_var,
                problem.problem_parameters.objective_weights["treatment_range"],
            ),
            "max_uncertain_treatment_range": (
                self.second_stage_treatment_range_var,
                math.ceil(problem.problem_parameters.objective_weights["treatment_range"]*(1-self.first_stage_weight_proportion)),
            ),
            "max_due_dates_penalties": (
                self.max_due_dates_penalties,
                problem.problem_parameters.objective_weights["due_dates_penalties"],
            )
            # "time_regularity": (
            #     sum_time_preference,
            #     problem.problem_parameters.objective_weights["time_regularity"],
            # ),
            # "machine_preference": (
            #     machine_preference,
            #     problem.problem_parameters.objective_weights["machine_preference"],
            # ),
            # "max_machine_balance": (
            #     max_machine_balance,
            #     problem.problem_parameters.objective_weights["machine_balance"],
            # ),
            # "machine_change": (
            #     machine_changes,
            #     problem.problem_parameters.objective_weights["machine_change"],
            # ),
        }

        print(f"Objective weights : \n"
              f"\t Earliest first stage : {problem.problem_parameters.objective_weights["earliest"]}\n"
              f"\t Treatment range first stage : {problem.problem_parameters.objective_weights["treatment_range"]}\n"
              f"\t Earliest second stage : {math.ceil(problem.problem_parameters.objective_weights["earliest"]*(1-self.first_stage_weight_proportion))}\n"
              f"\t Treatment range second stage : {math.ceil(problem.problem_parameters.objective_weights["treatment_range"]*(1-self.first_stage_weight_proportion))}\n"
              f"\t Due dates penalties second stage : {1.0}\n")

        self.model.Minimize(
            cp_model.LinearExpr.Sum(
                [objective * weight for objective, weight in objectives.values()]
            )
        )

    def solve(self, problem: Problem, fix_seed: bool = False):
        super().solve(problem, fix_seed)

    def retrieve_solution(self, problem: Problem) -> MultiStageSolution:
        return super().retrieve_solution(problem)
