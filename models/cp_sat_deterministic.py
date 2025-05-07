from abc import ABC
from typing import Dict

import pandas as pd
from ortools.sat.python import cp_model
from datetime import timedelta, datetime, time

from API.Problem import Problem
from API.Solution import Solution
from data_structures.Location import Priority

from globals import *
from time import perf_counter

from models.Solver import Solver


INFINITY = 1000000


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """
    Print intermediate solutions.
    """

    def __init__(
        self,
        patient_vars: Dict,
        problem: Problem,
        kpi_data: pd.DataFrame,
        retrieve_intermediate_sols: bool = False,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__dict_patients_vars = patient_vars
        self.__solution_count = 0
        self.problem = problem
        self.kpi_data = kpi_data
        self.start_time = perf_counter()
        self.retrieve_intermediate_sols = retrieve_intermediate_sols

    def on_solution_callback(self) -> None:
        """
        Print intermediate solution when solver finds one.
        :return: None
        """

        print(
            f"========== Solution found @ t = {perf_counter() - self.start_time} "
            f"/ Value : {self.ObjectiveValue():.2f} "
            f"- Best bound : {self.BestObjectiveBound():.2f} "
            f"- Gap : {100*(self.ObjectiveValue()/self.BestObjectiveBound() - 1 if self.BestObjectiveBound() > 0 else 0):.2f} %"
        )
        # Display the value of each objective and its weight.
        # print(
        #     f" Earliest (w={self.problem.problem_parameters.objective_weights["earliest"]:.2f}) : {
        #     sum(
        #         [
        #             self.Value(self.__dict_patients_vars[patient.id]["first_day_of_treatment"])
        #             for patient in self.problem.patients_queue
        #         ]
        #     )
        #     }\t"
        #     f"Treatment Range (w={self.problem.problem_parameters.objective_weights["treatment_range"]:.2f}) : {sum([self.Value(pvars["treatment_range"]) for pvars in self.__dict_patients_vars.values()])}\t"
        #     f"Time regularity (w={self.problem.problem_parameters.objective_weights["time_regularity"]:.2f}) : \t"
        #     f"Machine balance (w={self.problem.problem_parameters.objective_weights["machine_balance"]:.2f}) : \t"
        # )
        self.__solution_count += 1
        if self.retrieve_intermediate_sols:
            # Retrieve current solution.
            schedule = []
            local_date = self.problem.horizon_start
            for patient in self.problem.patients_queue:
                pvars = self.__dict_patients_vars[patient.id]
                # We need to sort the sessions by start date.
                sessions = sorted(
                    [
                        (
                            self.Value(session.start),
                            self.Value(session.end),
                            m,
                        )
                        for m in pvars["machines_authorized"].keys()
                        for s_id, session in enumerate(pvars["sessions"][m])
                        if self.Value(session.is_used)
                    ],
                    key=lambda x: x[0],
                )
                for session in sessions:
                    resource = linacs_ids[session[2]]
                    start = local_date + timedelta(minutes=session[0])
                    end = local_date + timedelta(minutes=session[1])
                    schedule.append((patient, resource, (start, end)))
            # Creating a temporary Solution Object.
            solution = Solution(self.problem)
            solution.patients = self.problem.patients_queue
            solution.schedule = schedule

            # Measuring quality indicators.
            delta_treatment_release = solution.qi_treatment_dates()
            treatment_ranges = solution.qi_treatment_ranges()
            count_violated, days_count = solution.qi_periods_respected()
            time_regularity, _ = solution.qi_time_regularity()
            dict_preferred_machines, dict_accepted_machines = (
                solution.qi_machines_preference()
            )

            lst = [
                self.__solution_count,
                delta_treatment_release,
                treatment_ranges,
                count_violated,
                days_count,
                time_regularity,
                sum(dict_accepted_machines.values()),
                perf_counter() - self.start_time,
            ]
            # Concatenate kpi data of the current solution into the overall dataframe kpi_data.
            self.kpi_data = pd.concat(
                [
                    self.kpi_data,
                    pd.DataFrame(
                        [lst],
                        columns=self.kpi_data.columns.values,
                    ),
                ],
                ignore_index=True,
            )
            # print(self.kpi_data)

    @property
    def solution_count(self) -> int:
        return self.__solution_count


class SolverCPSAT(Solver, ABC):

    def __init__(self):
        super().__init__(cp_model.CpModel(), "CP-SAT", {})
        # Variables related attributes
        self.dict_patient_vars = None
        self.dict_machine_vars = None
        # Data preprocessing related attributes
        self.new_tasks_count = None
        self.minutes_horizon = None
        self.occupancy_count = None
        self.total_occupancy_count = None
        self.average_occupancy = None
        self.ready_dates = None
        self.due_dates = None
        self.task_processing_time_data = None
        self.preferences_by_patient = None
        # Solver related attributes
        self.solver = None
        self.status = None
        self.solution_callback = None

    def preprocess_problem_data(self, problem: Problem):
        # =======================================================================
        # Compute the occupancy of the already existing timeplanner.
        # =======================================================================
        occupancy_count = timeplanner_count(
            dataset_1.timeplanner_filepath, count_type="tasks"
        )
        total_occupancy_count = sum(occupancy_count.values())
        average_occupancy = total_occupancy_count // len(occupancy_count)

        # =======================================================================
        # Compute the ready dates and due dates in minutes for each patient.
        # =======================================================================
        ready_dates = {}
        due_dates = {}

        for patient in problem.patients_queue:
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
                        durations_per_task[linac.id] = time_linac
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

        self.new_tasks_count = sum(
            patient.nb_fractions for patient in problem.patients_queue
        )
        self.minutes_horizon = 1440 * problem.horizon
        self.occupancy_count = occupancy_count
        self.total_occupancy_count = total_occupancy_count
        self.average_occupancy = average_occupancy
        self.ready_dates = ready_dates
        self.due_dates = due_dates
        self.task_processing_time_data = task_processing_time_data
        self.preferences_by_patient = preferences_by_patient

    def init_model(self, problem: Problem):
        session = collections.namedtuple("session", "is_used start end")

        # VARIABLES
        self.dict_patient_vars = {
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
                # Boolean variable that indicates if the machine changes between two sessions of one given patient.
                "machine_change": [
                    self.model.NewBoolVar(f"{patient.id}_changes_machine_{t}_{t + 1}")
                    for t in range(patient.nb_fractions - 1)
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
                "machine_task_count": self.model.NewIntVar(
                    0, self.new_tasks_count, f"machine_task_count_{linac.id}"
                ),
            }
            for linac in problem.linacs
        }

        # Interval variable creation for patients.
        for p, pvars in self.dict_patient_vars.items():
            # REAL SESSIONS INTERVALS
            for s_id, session in enumerate(pvars["sessions_intervals"]):
                session_interval = self.model.NewIntervalVar(
                    start=session[0],
                    size=session[2],
                    end=session[1],
                    name=f"session_interval_{s_id}_of_patient_{p}",
                )
                # noinspection PyTypeChecker
                self.dict_patient_vars[p]["sessions_intervals"][s_id] = session + (
                    session_interval,
                )

    # add_constraints abstract

    # add_objectives abstract

    def solve(self, problem: Problem, fix_seed: bool = False, **kwargs):
        self.preprocess_problem_data(problem)
        self.init_model(problem)
        self.add_constraints(problem)
        self.add_objectives(problem)

        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = problem.solver_time_limit
        # Fix seed to make the solver deterministic.
        if fix_seed:
            self.solver.parameters.random_seed = 42
        self.solution_callback = SolutionCallback(
            patient_vars=self.dict_patient_vars,
            problem=problem,
            kpi_data=self.kpi_df,
            retrieve_intermediate_sols=kwargs.get("retrieve_intermediate_sols", False),
        )

        self.status = self.solver.Solve(self.model, self.solution_callback)
        self.kpi_df = self.solution_callback.kpi_data

        # print(f'Status solver : ", {self.solver.StatusName(self.status)}')
        # print(
        #     f'Status solver : ", {self.solver.ObjectiveValue()} best obj bound={self.solver.BestObjectiveBound()}'
        # )

    def retrieve_solution(self, problem: Problem) -> Solution:
        # Return the solution.
        schedule = []

        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            local_date = problem.horizon_start
            for patient in problem.patients_queue:
                pvars = self.dict_patient_vars[patient.id]
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
                # print(solver.Value(dict_patient_vars[patient.id]["first_day_of_treatment"]))
                # print(solver.Value(dict_patient_vars[patient.id]["last_day_of_treatment"]))
                # print(solver.Value(dict_patient_vars[patient.id]["treatment_range"]))
                for session in sessions:
                    resource = linacs_ids[session[2]]
                    # print(f"Machine id {resource}"
                    #       f" - start = {session[0]} and end = {session[1]}")
                    start = local_date + timedelta(minutes=session[0])
                    end = local_date + timedelta(minutes=session[1])
                    schedule.append((patient, resource, (start, end)))

            print(f"================== Done ==================")
            print(f"Status : {self.solver.StatusName(self.status)}")
            print(f"Objective value : {self.solver.ObjectiveValue()}")
            print(f"Best bound : {self.solver.BestObjectiveBound()}")
            print(
                f"Gap : {100*(self.solver.ObjectiveValue()/self.solver.BestObjectiveBound() - 1 if self.solver.BestObjectiveBound() > 0 else 0):.2f} %"
            )
            print(
                f"Number of solutions found : {self.solution_callback.solution_count}"
            )

            print(f"Number of tasks : {self.new_tasks_count}")
            print(
                f"Number of machine changes : "
                f"""{sum(sum(self.solver.Value(pvars['machine_change'][k])
                             for k in range(len(pvars['machine_change'])))
                         for pvars in self.dict_patient_vars.values())}"""
            )
        elif self.status == cp_model.MODEL_INVALID:
            print("Invalid model.")
        else:
            print(f"Status = {self.solver.StatusName(self.status)}")
        print("Statistics")
        print(f"  - conflicts : {self.solver.NumConflicts()}")
        print(f"  - branches  : {self.solver.NumBranches()}")
        print(f"  - wall time : {self.solver.WallTime()}s")

        solution = Solution(problem=problem)
        solution.patients = problem.patients_queue
        solution.schedule = schedule
        solution.existing_schedule = merged_schedule_to_list(
            linacs=problem.linacs,
            horizon_start=problem.horizon_start,
            existing_schedule=problem.existing_schedule,
        )
        solution.statistics["objective_value"] = self.solver.ObjectiveValue()
        solution.statistics["solver_status"] = str(self.solver.StatusName(self.status))
        solution.statistics["solver_time"] = self.solver.WallTime()
        return solution


class SolverCP(SolverCPSAT):

    def __init__(self):
        super().__init__()

    def preprocess_problem_data(self, problem: Problem):
        super().preprocess_problem_data(problem)

    def init_model(self, problem: Problem):
        super().init_model(problem)

    def add_constraints(self, problem: Problem):
        interval_session = collections.namedtuple(
            "interval_session", "is_used start end interval"
        )

        # OPTIONAL INTERVALS
        for p, pvars in self.dict_patient_vars.items():
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
                        self.dict_patient_vars[p]["sessions_intervals"][s_id][0]
                        == session_schedules.start
                    ).OnlyEnforceIf(session_schedules.is_used)
                    self.model.Add(
                        self.dict_patient_vars[p]["sessions_intervals"][s_id][1]
                        == session_schedules.end
                    ).OnlyEnforceIf(session_schedules.is_used)
                    self.model.Add(
                        self.dict_patient_vars[p]["sessions_intervals"][s_id][2]
                        == slot_duration
                    ).OnlyEnforceIf(session_schedules.is_used)
                    # The comma after interval is to create a tuple of only one element and return a tuple of n+1 elems.
                    # We add the optional interval to the "sessions" in patients_vars.
                    session_schedules = interval_session(
                        *session_schedules + (interval,)
                    )
                    # Don't forget to update the list of schedules.
                    self.dict_patient_vars[p]["sessions"][m][s_id] = session_schedules

        # The following dict allows to retrieve the boolean variable indicating if the slot is used for a given machine,
        # for each authorized machine, for each slot and for each patient.
        machines_by_session = {}
        for patient, pvars in self.dict_patient_vars.items():
            for m in pvars["machines_authorized"].keys():
                # nb_fractions = len(pvars['sessions'][m])
                machines_by_session[patient] = [
                    [
                        pvars["sessions"][m][i].is_used
                        for m in pvars["machines_authorized"].keys()
                    ]
                    for i in range(len(pvars["sessions"][m]))
                ]

                # CONSTRAINTS : no overlapping between tasks of the same patient.
                # != machine overlapping, redundant if all patient's tasks are on the same machine.
                if problem.problem_parameters.allow_change_for_mirrors_linacs:
                    for s_id in range(1, len(pvars["sessions"][m])):
                        session_s = pvars["sessions"][m][s_id]
                        for m1 in pvars["machines_authorized"].keys():
                            session_sm1 = pvars["sessions"][m1][s_id - 1]
                            self.model.Add(
                                session_sm1.end <= session_s.start
                            ).OnlyEnforceIf(session_sm1.is_used)

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
            # Case allow_change_for_mirrors_linacs is True.
            if problem.problem_parameters.allow_change_for_mirrors_linacs:
                # len(machines_by_session[patient]) = nb_fractions of this patient.
                for k in range(len(machines_by_session[patient])):
                    self.model.AddExactlyOne(machines_by_session[patient][k])
            else:
                self.model.AddExactlyOne(machines_by_session[patient][0])
                for k in range(1, len(machines_by_session[patient])):
                    for m in range(len(pvars["machines_authorized"])):
                        # print(machines_by_session[patient][k])
                        self.model.Add(
                            machines_by_session[patient][k][m] == 1
                        ).OnlyEnforceIf(machines_by_session[patient][0][m])
                        self.model.Add(
                            machines_by_session[patient][k][m] == 0
                        ).OnlyEnforceIf(machines_by_session[patient][0][m].Not())

            # CONSTRAINTS: Define the makespan for each patient, i.e. the end time of the last task.
            # makespan == max(end of last task on machine m, for each machine m)
            self.model.AddMaxEquality(
                pvars["makespan"],
                [sessions[-1].end for sessions in pvars["sessions"].values()],
            )

            # b is a temporary variable
            b = [
                [
                    self.model.NewBoolVar(f"b_{k}_{m}")
                    for m in range(len(pvars["machines_authorized"]))
                ]
                for k in range(len(machines_by_session[patient]) - 1)
            ]
            # CONSTRAINTS: Machine change variables, the variable is set to true if the machine of the previous session
            # and the current differ.
            # TODO : the changes must be between the TSU machine and the current
            for k in range(len(machines_by_session[patient]) - 1):
                for m in range(
                    len(pvars["machines_authorized"])
                ):  # TODO : should change this to use a dict key
                    self.model.Add(
                        machines_by_session[patient][k + 1][m]
                        == machines_by_session[patient][k][m]
                    ).OnlyEnforceIf(b[k][m])
                    self.model.Add(
                        machines_by_session[patient][k + 1][m]
                        != machines_by_session[patient][k][m]
                    ).OnlyEnforceIf(b[k][m].Not())
                    # Warning might be some erratic behavior with that one [0,1,0] -> [1,0,0] == two changes & one still
                    self.model.Add(pvars["machine_change"][k] == 0).OnlyEnforceIf(
                        b[k][m]
                    )
                    self.model.Add(pvars["machine_change"][k] == 1).OnlyEnforceIf(
                        b[k][m].Not()
                    )

        # CONSTRAINTS : no overlapping between tasks on same machine.
        for linac in problem.linacs:
            machine_intervals = []
            for pvars in self.dict_patient_vars.values():
                if linac.id in pvars["machines_authorized"]:
                    for session in pvars["sessions"][linac.id]:
                        machine_intervals.append(session.interval)
            self.model.AddNoOverlap(
                machine_intervals
                + self.dict_machine_vars[linac.id]["unavailability_slots"][0]
            )
            self.model.AddNoOverlap(
                machine_intervals
                + self.dict_machine_vars[linac.id]["unavailability_slots"][1]
            )

        # CONSTRAINTS : periods enforcement
        for patient in problem.patients_queue:
            pvars = self.dict_patient_vars[patient.id]
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
                self.model.Add(
                    pvars["delta_period"][i] == sessions[i + 1][0] - sessions[i][1]
                )
                # if period <= 1:
                #     self.model.Add(sessions[i + 1][0] - sessions[i][1] <= 1440 * 3)
                # else:
                #     self.model.Add(sessions[i + 1][0] - sessions[i][1] <= 1440 * 4)

    def add_objectives(self, problem: Problem):
        # Machine tasks assignment count, this variable is used to count the number of tasks allocated on a given
        # machine.
        task_assigned_linac = {
            linac.id: self.model.NewIntVar(
                # the lower bound cannot be less than the occupancy count on the given linac.
                self.occupancy_count[linac.id],
                # the upper bound is up to the occupancy count on the given linac + the count of all new tasks.
                self.occupancy_count[linac.id] + self.new_tasks_count,
                f"tasks_assigned_{linac.id}",
            )
            for linac in problem.linacs
        }
        #
        epsilon = self.model.NewIntVar(
            0, self.total_occupancy_count + self.new_tasks_count, "epsilon"
        )
        # CONSTRAINTS : Machine balance count.
        for linac in problem.linacs:
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
                    for patient, pvars in self.dict_patient_vars.items()
                )
                + self.occupancy_count[linac.id]
                == task_assigned_linac[linac.id]
            )
            self.model.Add(
                task_assigned_linac[linac.id] <= self.average_occupancy + epsilon
            )
            self.model.Add(
                task_assigned_linac[linac.id] >= self.average_occupancy - epsilon
            )

        # OBJECTIVE
        for patient_id in self.dict_patient_vars:
            self.dict_patient_vars[patient_id]["first_day_of_treatment"] = (
                self.model.NewIntVar(
                    self.ready_dates[patient_id]
                    % 1440,  # first day of treatment cannot be earlier than release date
                    problem.horizon,
                    name=f"startday_{patient_id}",
                )
            )
            self.dict_patient_vars[patient_id]["last_day_of_treatment"] = (
                self.model.NewIntVar(
                    self.ready_dates[patient_id]
                    % 1440,  # last day of treatment cannot be earlier than release date
                    problem.horizon,
                    name=f"endday_{patient_id}",
                )
            )
            self.model.AddDivisionEquality(
                self.dict_patient_vars[patient_id]["first_day_of_treatment"],
                self.dict_patient_vars[patient_id]["sessions_intervals"][0][
                    0
                ],  # the start of the first session.
                # - ready_dates[patient_id]
                denom=1440,
            )
            self.model.AddDivisionEquality(
                self.dict_patient_vars[patient_id]["last_day_of_treatment"],
                self.dict_patient_vars[patient_id]["sessions_intervals"][-1][
                    0
                ],  # the start of the last session.
                denom=1440,
            )
            # No need for this variable in practice.
            self.model.Add(
                self.dict_patient_vars[patient_id]["last_day_of_treatment"]
                - self.dict_patient_vars[patient_id]["first_day_of_treatment"]
                == self.dict_patient_vars[patient_id]["treatment_range"]
            )
        earliest = cp_model.LinearExpr.Sum(
            [
                (4 - patient.location.urgency.value)
                * self.dict_patient_vars[patient.id]["first_day_of_treatment"]
                for patient in problem.patients_queue
            ]
        )
        treatment_range = cp_model.LinearExpr.Sum(
            [pvars["treatment_range"] for pvars in self.dict_patient_vars.values()]
        )
        machine_preferrence = cp_model.LinearExpr.Sum(
            [
                pvars["sessions"][m][s_id].is_used
                * (2 - self.preferences_by_patient[p][m])
                for p, pvars in self.dict_patient_vars.items()
                for m, _ in pvars["machines_authorized"].items()
                for s_id, session_schedules in enumerate(pvars["sessions"][m])
            ]
        )
        machine_changes = cp_model.LinearExpr.Sum(
            [
                cp_model.LinearExpr.Sum(
                    [
                        pvars["machine_change"][k]
                        for k in range(len(pvars["machine_change"]))
                    ]
                )
                for pvars in self.dict_patient_vars.values()
            ]
        )
        delta_periods = cp_model.LinearExpr.Sum(
            [
                cp_model.LinearExpr.Sum(
                    [
                        pvars["delta_period"][k]
                        for k in range(len(pvars["delta_period"]))
                    ]
                )
                for pvars in self.dict_patient_vars.values()
            ]
        )
        # This is not properly a makespan.
        makespan = cp_model.LinearExpr.Sum(
            [pvars["makespan"] for pvars in self.dict_patient_vars.values()]
        )

        time_regularity_objectives = self.add_objective_time_regularity(problem=problem)
        time_regularity = sum(time_regularity_objectives)

        objectives = {
            "earliest": (
                earliest,
                problem.problem_parameters.objective_weights["earliest"],
            ),
            # "machine_change": (
            #     machine_changes,
            #     problem.problem_parameters.objective_weights["machine_change"],
            # ),
            "treatment_range": (
                treatment_range,
                problem.problem_parameters.objective_weights["treatment_range"],
            ),
            # "epsilon": (
            #     epsilon,
            #     problem.problem_parameters.objective_weights["machine_balance"],
            # ),
            "time_regularity": (
                time_regularity,
                problem.problem_parameters.objective_weights["time_regularity"],
            ),
            # "machine_preferrence": (
            #     machine_preferrence,
            #     problem.problem_parameters.objective_weights["machine_preferrence"],
            # ),
        }

        self.model.Minimize(
            cp_model.LinearExpr.Sum(
                [objective * weight for objective, weight in objectives.values()]
            )
        )

    def add_objective_time_regularity(self, problem: Problem):
        objectives = []
        for patient in problem.patients_queue:
            pvars = self.dict_patient_vars[patient.id]
            sessions = pvars["sessions_intervals"]
            pvars["sessions_timing"] = [
                self.model.NewIntVar(
                    0, 24 * 60, name=f"patient_{patient.id}_session_{i}"
                )
                for i in range(len(sessions))
            ]
            nb_session = len(pvars["sessions_timing"])
            pvars["delta_to_0_session"] = self.model.NewIntVar(
                0, 24 * 60, name=f"epsilon_patient_{patient.id}"
            )

            pvars["delta_consecutive_session"] = [
                self.model.NewIntVar(
                    0, 24 * 60, name=f"epsilon_patient_consecutive_{patient.id}"
                )
                for i in range(nb_session - 1)
            ]
            for i in range(nb_session):
                self.model.AddModuloEquality(
                    pvars["sessions_timing"][i], sessions[i][0], mod=1440
                )
            for i in range(nb_session - 1):
                self.model.Add(
                    pvars["sessions_timing"][i + 1]
                    <= pvars["sessions_timing"][0] + pvars["delta_to_0_session"]
                )
                self.model.Add(
                    pvars["sessions_timing"][i + 1]
                    >= pvars["sessions_timing"][0] - pvars["delta_to_0_session"]
                )
                self.model.Add(
                    pvars["sessions_timing"][i + 1]
                    <= pvars["sessions_timing"][i]
                    + pvars["delta_consecutive_session"][i]
                )
                self.model.Add(
                    pvars["sessions_timing"][i + 1]
                    >= pvars["sessions_timing"][i]
                    - pvars["delta_consecutive_session"][i]
                )
            objectives.append(pvars["delta_to_0_session"])
            objectives.extend(pvars["delta_consecutive_session"])
        return objectives

    def solve(self, problem: Problem, fix_seed: bool = False):
        super().solve(problem, fix_seed)

    def retrieve_solution(self, problem: Problem) -> Solution:
        return super().retrieve_solution(problem)
