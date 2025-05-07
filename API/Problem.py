from collections import defaultdict
from typing import List, Dict, Tuple
import datetime
from enum import Enum
from collections import namedtuple

from data_structures.Location import Location
from data_structures.Patient import Patient
from data_structures.Linac import Linac
from globals import linacs_ids


problem_parameters_tuple = namedtuple(
    "problem_parameters", ["allow_change_for_mirrors_linacs", "objective_weights"]
)


class Problem(object):
    def __init__(
        self,
        linacs: List[Linac],
        locations: Dict[str, Location],
        # proportions: pd.DataFrame,
        patients_queue: List[Patient],
        existing_schedule: List[Patient],
        horizon_start: datetime.datetime,  # datetime.date
        horizon: int,
        problem_parameters: namedtuple,
        # allow_change_for_mirrors_linacs: bool = False,
        solver_time_limit: int = 20,
    ) -> None:
        self.linacs = linacs
        self.locations = locations
        # self.proportions = proportions
        self.patients_queue = patients_queue
        self.nb_patients = len(patients_queue)
        self.existing_schedule = existing_schedule
        self.horizon_start = horizon_start
        self.horizon = horizon
        self.problem_parameters = problem_parameters
        # self.allow_change_for_mirrors_linacs = allow_change_for_mirrors_linacs
        self.solver_time_limit = solver_time_limit

    def update_existing_schedule(
        self,
        updates: List[
            Tuple[Patient, int, Tuple[datetime.datetime, datetime.datetime]]
        ] = None,
    ):
        patients_schedules = defaultdict(list)
        for k in range(len(updates)):
            p_k, resource_k, (start_k, end_k) = updates[k]
            patients_schedules[p_k].append((start_k, end_k, resource_k))
        for p_k, patient_sessions in patients_schedules.items():
            sorted_sessions = sorted(patient_sessions, key=lambda x: x[0])
            patients_schedules[p_k] = sorted_sessions
        for patient, patient_sessions in patients_schedules.items():
            schedule = []
            for start, end, resource in patient_sessions:
                s = (self.linacs[linacs_ids[resource]], (start, end))
                schedule.append(s)
            patient.schedule = schedule
            # patient.id = 0
            self.existing_schedule.append(patient)
