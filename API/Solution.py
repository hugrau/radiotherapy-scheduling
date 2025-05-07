import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from datetime import datetime, time, timedelta
from copy import deepcopy

from data_structures.Patient import Patient
from data_structures.Location import Priority
from data_structures.Linac import get_linac_by_name
from API.Problem import Problem
from globals import linacs_ids
from data_treatments.calendars import (
    generate_business_day_array,
    merged_schedule_to_list,
)


def same_day(dt1, dt2):
    """
    Function to compute if two datetimes are the same day.
    """
    return dt1.year == dt2.year and dt1.month == dt2.month and dt1.day == dt2.day


class Solution(object):
    """
    Solution class that encapsulates the useful data for solving the problem.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.patients: List[Patient] = problem.patients_queue
        self.schedule: List[Tuple[Patient, int, Tuple[datetime, datetime]]] = []
        # The schedule is for the list of patients above.
        self.is_valid_solution = True
        self.existing_schedule: List[Tuple[Patient, int, Tuple[datetime, datetime]]] = (
            []
        )
        self.statistics: Dict = {
            "objective_value": 0,
            "solver_status": "None",
            "solver_time": 0,
        }
        self.horizon_start: datetime = problem.horizon_start
        self.horizon: int = problem.horizon

    def schedule_to_patient_schedule(self):
        """

        :return:
        """
        patients_schedule = defaultdict(list)
        for k in range(len(self.schedule)):
            p_k, resource_k, (start_k, end_k) = self.schedule[k]
            patients_schedule[p_k].append(
                (start_k, end_k, self.problem.linacs[resource_k].name)
            )
        for p_k, patient_sessions in patients_schedule.items():
            sorted_sessions = sorted(patient_sessions, key=lambda x: x[0])
            patients_schedule[p_k] = sorted_sessions
        return patients_schedule

    def to_patient_objects(self):
        patient_objects = []
        patients_schedule = self.schedule_to_patient_schedule()
        for patient, schedule in patients_schedule.items():
            patient_schedule = []  # List of type (Linac, (datetime, datetime))
            for session in schedule:
                # Format session to format (Linac, (datetime, datetime))
                formatted_session = (
                    get_linac_by_name(session[2], self.problem.linacs),
                    (session[0], session[1]),
                )
            p = deepcopy(patient)
            p.schedule = patient_schedule
            patient_objects.append(p)
        return patient_objects

    def is_not_overlapping(
        self,
        verbose: bool = False,
    ) -> bool:
        overlaps = True
        # Check if there are sessions at the same time on the same machine.
        for k in range(len(self.schedule)):
            for l in range(k + 1, len(self.schedule)):
                p_k, resource_k, (start_k, end_k) = self.schedule[k]
                p_l, resource_l, (start_l, end_l) = self.schedule[l]
                # Checks for partial overlapping.
                if resource_k == resource_l and (
                    (start_k < start_l < end_k < end_l)
                    or (start_l < start_k < end_l < end_k)
                ):
                    # print(f"{start_k=} - {end_k=} - {start_l=} - {end_l=}")
                    if verbose:
                        print(
                            f"Partial overlapping conflict {p_k.id} and {p_l.id} on resource {resource_k} "
                            f"at datetime : {(start_k, end_k)} and {(start_l, end_l)}."
                        )
                    overlaps = False
                # Checks for full overlapping.
                elif resource_k == resource_l and (
                    (start_k <= start_l and end_l <= end_k)
                    or (start_l <= start_k and end_k <= end_l)
                ):
                    # print(f"{start_k=} - {end_k=} - {start_l=} - {end_l=}")
                    if verbose:
                        print(
                            f"Full overlapping conflict {p_k.id} and {p_l.id} on resource {resource_k} "
                            f"at datetime : {(start_k, end_k)} and {(start_l, end_l)}."
                        )
                    overlaps = False
        return overlaps

    def has_all_fractions_scheduled(self, verbose: bool = False) -> bool:
        all_fractions_scheduled = True

        # Create a dictionary with sessions count for each patient in the resulting schedule, the keys are patient.id
        patient_session_count = {patient.id: 0 for patient in self.patients}
        for k in range(len(self.schedule)):
            p_k, resource_k, (start_k, end_k) = self.schedule[k]
            patient_session_count[p_k.id] += 1

        # Create a dictionary with the required fractions number for each patient, the keys are patient.id
        keys = [p.id for p in self.patients]
        values = [p.nb_fractions for p in self.patients]
        required_fractions = dict(zip(keys, values))
        # print(required_fractions)

        if verbose and patient_session_count != required_fractions:
            print(f"Patient sessions count : {patient_session_count}")
            print(f"Patient sessions required : {required_fractions}")
        return patient_session_count == required_fractions

    def no_forbidden_linac(self, verbose: bool = False) -> bool:
        no_forbidden_linac = True

        for k in range(len(self.schedule)):
            p_k, resource_k, (start_k, end_k) = self.schedule[k]
            forbidden_linacs = p_k.location.linacs_by_priority[Priority.FORBIDDEN]
            # print([forbidden_linac.id for forbidden_linac in p_k.location.linacs_by_priority[Priority.FORBIDDEN]])
            for forbidden_linac in forbidden_linacs:
                # From a linac id we want its position in oncopole linacs list
                if linacs_ids[forbidden_linac.id] == resource_k:
                    if verbose:
                        print(
                            f"Patient {p_k.id} with location {p_k.location.name} "
                            f"scheduled on forbidden linac {forbidden_linac.name} == {resource_k}."
                        )
                    no_forbidden_linac = False

        return no_forbidden_linac

    def no_linac_changes(self, verbose: bool = False) -> bool:
        """
        The linac of treatment initiation is the linac we want to treat most of the time with exceptions for mirrors.
        :param verbose:
        :return: bool
        """
        # We need this to track what is the session
        count = 0
        patient_session_count = {patient.id: 0 for patient in self.patients}
        # count and in particular the first session.
        patient_previous_linac = (
            {}
        )  # We need the dict to store the value of first linac.
        patient_changes_linac = {patient.id: False for patient in self.patients}
        for session in self.schedule:
            p_k, resource_k, (start_k, end_k) = session
            t = patient_session_count[p_k.id]
            if t == 0:
                patient_previous_linac[p_k.id] = resource_k
            if t > 0 and resource_k != patient_previous_linac[p_k.id]:
                count += 1
                if verbose:
                    print(
                        f"Patient {p_k.id} has changed linac "
                        f"from {self.problem.linacs[patient_previous_linac[p_k.id]].name} "
                        f"to {self.problem.linacs[resource_k].name} at {start_k} to {end_k}."
                    )
                patient_changes_linac[p_k.id] = True
                patient_previous_linac[p_k.id] = resource_k
            patient_session_count[p_k.id] += 1
        for value in patient_changes_linac.values():
            if value:
                print(f"Number of linac changes in checker : {count}")
                return False
        return True

    def is_not_overlapping_with_schedule(self, verbose: bool = False):
        # Update existing schedule.
        self.existing_schedule = merged_schedule_to_list(
            linacs=self.problem.linacs,
            horizon_start=self.problem.horizon_start,
            existing_schedule=self.problem.existing_schedule,
        )
        if not self.existing_schedule:
            return True
        overlaps = True
        conflict_count = 0
        # Check if there are sessions at the same time on the same machine.
        for k in range(len(self.schedule)):
            for l in range(len(self.existing_schedule)):
                p_k, resource_k, (start_k, end_k) = self.schedule[k]
                p_l, resource_l, (start_l, end_l) = self.existing_schedule[l]
                # Checks for partial overlapping.
                if resource_k == resource_l and (
                    (start_k < start_l < end_k < end_l)
                    or (start_l < start_k < end_l < end_k)
                ):
                    # print(f"{start_k=} - {end_k=} - {start_l=} - {end_l=}")
                    if verbose:
                        print(
                            f"Partial overlapping conflict {p_k.id} and {p_l.id} on resource {resource_k} "
                            f"at datetime : {(start_k, end_k)} and {(start_l, end_l)}."
                        )
                    overlaps = False
                    conflict_count += 1
                # Checks for full overlapping.
                elif resource_k == resource_l and (
                    (start_k <= start_l and end_l <= end_k)
                    or (start_l <= start_k and end_k <= end_l)
                ):
                    # print(f"{start_k=} - {end_k=} - {start_l=} - {end_l=}")
                    if verbose:
                        print(
                            f"Full overlapping conflict {p_k.id} and {p_l.id} on resource {resource_k} "
                            f"at datetime : {(start_k, end_k)} and {(start_l, end_l)}."
                        )
                    overlaps = False
                    conflict_count += 1
        if conflict_count > 0:
            print(f"Number of conflicts : {conflict_count}")
        return overlaps

    def is_period_respected(self, verbose: bool = False):
        horizon_array = generate_business_day_array(self.horizon_start, self.horizon)

        periods_respected = True
        patients = self.schedule_to_patient_schedule()
        for patient, patient_sessions in patients.items():
            # If only one session, we don't need to compute the difference.
            if len(patient_sessions) == 1:
                continue
            for k in range(len(patient_sessions) - 1):
                days_kp1 = int((patient_sessions[k + 1][0] - self.horizon_start).days)
                days_k = int((patient_sessions[k][0] - self.horizon_start).days)
                d = [1 for _ in range(days_kp1 - days_k + 1)]
                # print(d, horizon_array[days_k : days_kp1 + 1])
                res = np.dot(d, horizon_array[days_k : days_kp1 + 1])
                # print(res)
                if (
                    same_day(patient_sessions[k + 1][0], patient_sessions[k][0])
                    or res > patient.location.period + 1
                ):
                    periods_respected = False
                    if verbose:
                        print(
                            f"{patient.id} period not respected on session {k+1} to {k+2}, {res} days between sessions."
                            # f"{(patient_sessions[k+1][0] - patient_sessions[k][0]).total_seconds()//60//1440} days delta"
                        )
        return periods_respected

    def checker(self, verbose=False):
        """
        Checks if the solution returned by the solver is feasible according to the data_structures's criteria.
        :param self: a Solution object.
        :param verbose: boolean.
        :return: boolean, true if the solution is valid false.
        """
        validity_conditions = {
            "Not overlapping sessions": self.is_not_overlapping,
            "All fractions scheduled": self.has_all_fractions_scheduled,
            "No forbidden linacs": self.no_forbidden_linac,
            "Periods respected": self.is_period_respected,
            "No linac changes": self.no_linac_changes,
            "Not overlapping existing schedule": self.is_not_overlapping_with_schedule,
        }

        for condition in validity_conditions:
            if not validity_conditions[condition](verbose):
                print(f"\033[31m- Condition '{condition}' is not satisfied.\033[0m")
                self.is_valid_solution = False
            else:
                print(f"\033[32m- Condition '{condition}' is satisfied.\033[0m")

        # overlaps = self.check_non_overlapping()
        # if overlaps:
        #     self.is_valid_solution = False

        if self.is_valid_solution:
            print(f"\033[32mSolution is valid.\033[0m")
        else:
            print(f"\033[31mSolution is not valid.\033[0m")

    def qi_treatment_dates(self, verbose=False):
        horizon_array = generate_business_day_array(self.horizon_start, self.horizon)

        patient_count_session = {patient.id: 0 for patient in self.patients}
        patient_count_deltas = {
            patient.id: [0, patient.location.urgency.value] for patient in self.patients
        }
        time_count = 0
        time_count_business = 0
        print("QI : Treatment dates : ")
        for session in self.schedule:
            patient, resource, (start, end) = session
            ready_datetime = datetime.combine(patient.ready_date, time(0, 0, 0))
            due_datetime = datetime.combine(patient.due_date, time(0, 0, 0))
            if patient_count_session[patient.id] == 0:
                ready_day = int((ready_datetime - self.horizon_start).days)
                first_session_day = int((start - self.horizon_start).days)
                d = [1 for _ in range(first_session_day - ready_day)]
                # print(d, horizon_array[days_k : days_kp1 + 1])
                res = int(np.dot(d, horizon_array[ready_day:first_session_day]))
                # print(d, horizon_array[ready_day:first_session_day])
                # If the date of readiness is a saturday then the scheduling starts at least on Monday
                # with a delta of two days, so we subtract two to delta in that case.
                if ready_datetime.weekday() < 5:
                    delta_ready = (start - ready_datetime).days
                else:
                    delta_ready = (start - ready_datetime).days - 2
                if verbose:
                    print(
                        f"\tStart of the first session for patient {patient.id} is : {start} when ready date "
                        f"is {patient.ready_date} - delta= {delta_ready} days or {res} business days "
                        f"and due is {patient.due_date} - delta = {(due_datetime-start).days + 1} days."
                        # the "+ 1" here comes from the fact that
                        # datetime.datetime(2024, 1, 1, 0, 0, 0) - datetime.datetime(2024, 1, 1, 15, 30, 0) = -1d,30600s
                    )
                time_count += delta_ready
                time_count_business += res
                patient_count_deltas[patient.id][0] += res
            elif patient_count_session[patient.id] == patient.nb_fractions - 1:
                # This step is possible only if all fractions are scheduled.
                # print(f"Start of the last for patient {patient.id} session is : {end}")
                pass
            patient_count_session[patient.id] += 1
        print(
            f"\tSum of time deltas: {time_count} days (includes week-end) or {time_count_business} business days."
        )
        return time_count_business, patient_count_deltas

    def qi_periods_respected(self, verbose=False):
        """
        Count the number of periods not respected and the associated number of days.
        :param verbose: bool
        :return: (int, int)
        """
        count_violated = 0
        days_count = 0
        patients = self.schedule_to_patient_schedule()
        patient_count_periods = {
            patient.id: [0, patient.location.urgency.value] for patient in self.patients
        }
        horizon_array = generate_business_day_array(self.horizon_start, self.horizon)
        for patient, patient_sessions in patients.items():
            # If only one session, we don't need to compute the difference.
            if len(patient_sessions) == 1:
                continue
            for k in range(len(patient_sessions) - 1):
                days_kp1 = int((patient_sessions[k + 1][0] - self.horizon_start).days)
                days_k = int((patient_sessions[k][0] - self.horizon_start).days)
                d = [1 for _ in range(days_kp1 - days_k + 1)]
                # print(d, horizon_array[days_k : days_kp1 + 1])
                res = np.dot(d, horizon_array[days_k : days_kp1 + 1])
                if (
                    same_day(patient_sessions[k + 1][0], patient_sessions[k][0])
                    or res > patient.location.period + 1
                ):
                    count_violated += 1
                    patient_count_periods[patient.id][0] += 1
                    days_count += res
        print(
            f"QI : Periods enforcement : \n\tThere are {count_violated} period(s) violated for a total of "
            f"{days_count} day(s)."
        )
        return count_violated, patient_count_periods

    def qi_treatment_ranges(self, verbose=False):
        """

        :param verbose:
        :return:
        """
        patient_count_session = {patient.id: 0 for patient in self.patients}
        time_count = 0
        print("QI : Treatment ranges : ")
        patient_start_date = {patient.id: None for patient in self.patients}
        patient_end_date = {patient.id: None for patient in self.patients}
        for session in self.schedule:
            patient, resource, (start, end) = session
            if patient_count_session[patient.id] == 0:
                patient_start_date[patient.id] = end
            if patient_count_session[patient.id] == patient.nb_fractions - 1:
                # This step is possible only if all fractions are scheduled.
                # print(f"Start of the last for patient {patient.id} session is : {end}")
                patient_end_date[patient.id] = end
            patient_count_session[patient.id] += 1
        # print(
        #     [
        #         (patient_end_date[patient.id] - patient_start_date[patient.id]).days
        #         for patient in self.patients
        #     ]
        # )
        time_count += sum(
            [
                (patient_end_date[patient.id] - patient_start_date[patient.id]).days
                for patient in self.patients
            ]
        )
        print(f"Sum of time ranges: {time_count} ")
        return time_count

    def qi_machine_balance(self, verbose=False):
        """
        Function to count the number of tasks associated to each machine.
        :param verbose:
        :return:
        """
        machine_count = {i: 0 for i in range(len(self.problem.linacs))}
        for session in self.schedule:
            patient, resource, (start, end) = session
            machine_count[resource] += 1
        print("QI : Machines balance : ")
        for resource, count in machine_count.items():
            print(f"\t{self.problem.linacs[resource].name}: {count}")

    def qi_time_regularity(self, verbose=False):
        # horizon_array = generate_business_day_array(self.horizon_start, self.horizon)

        print("QI : Time regularity : ")
        patients = self.schedule_to_patient_schedule()
        overall_sum = 0
        patient_sum_irregularities = {
            patient.id: [0, patient.location.urgency.value]
            for patient in patients.keys()
        }
        for patient, patient_sessions in patients.items():
            for k in range(len(patient_sessions) - 1):
                days_kp1 = int((patient_sessions[k + 1][0] - self.horizon_start).days)
                days_k = int((patient_sessions[k][0] - self.horizon_start).days)
                start_kp1 = patient_sessions[k + 1][0]
                start_k = patient_sessions[k][0]
                days_between = days_kp1 - days_k
                time_regularity = (
                    abs((days_between * 86400) - (start_kp1 - start_k).total_seconds())
                    / 60
                )
                if time_regularity != 0:
                    print(
                        f"\tPatient {patient.id} has local time slot irregularity of {time_regularity} "
                        f"minutes between session {k+1} and {k+2}."
                    )
                    overall_sum += time_regularity
                    patient_sum_irregularities[patient.id][0] += time_regularity
        for patient_id, value in patient_sum_irregularities.items():
            print(f"\tPatient {patient_id}: {value} minutes.")
        print(f"\tOverall sum of irregularities: {overall_sum}")
        return overall_sum, patient_sum_irregularities

    def qi_outside_time_preference(self, verbose=False):
        pass

    def qi_machines_preference(self, verbose=False):
        patients = self.schedule_to_patient_schedule()
        nb_preferred_dict = {
            patient.id: [0, patient.location.urgency.value]
            for patient in patients.keys()
        }
        nb_accepted_dict = {
            patient.id: [0, patient.location.urgency.value]
            for patient in patients.keys()
        }

        print("QI : Machines preference : ")
        for patient, patient_sessions in patients.items():
            linacs_preferred = patient.location.linacs_by_priority[Priority.PREFERRED]
            linacs_accepted = patient.location.linacs_by_priority[Priority.ACCEPTED]
            for session_id, session in enumerate(patient_sessions):
                current_linac = get_linac_by_name(session[2], self.problem.linacs)
                if current_linac in linacs_preferred:
                    # print(
                    #     f"Patient {patient.id} on preferred linac for session {session_id}."
                    # )
                    nb_preferred_dict[patient.id][0] += 1
                if current_linac in linacs_accepted:
                    if verbose:
                        print(
                            f"\tPatient {patient.id} on accepted linac for session {session_id}, location is {patient.location.name} and linac is {current_linac.name}."
                        )
                    nb_accepted_dict[patient.id][0] += 1

            print(
                f"\t Patient {patient.id}: {nb_preferred_dict[patient.id]} sessions on preferred machines out of {patient.nb_fractions}."
            )
        return nb_preferred_dict, nb_accepted_dict

    def quality_indicators(self, verbose=False):

        quality_indicators = {
            "Treatment dates": self.qi_treatment_dates,
            "Periods respected": self.qi_periods_respected,
            # "Treatment ranges": self.qi_treatment_ranges,
            "Machine balance": self.qi_machine_balance,
            "Time regularity": self.qi_time_regularity,
            "Time windows preferences": self.qi_outside_time_preference,
            "Machines preference": self.qi_machines_preference,
        }
        returned_indicators = {}
        for indicator, _ in quality_indicators.items():
            returned_indicators[indicator] = quality_indicators[indicator](verbose)

        return returned_indicators

    def display_scheduling(self):
        """
        Display the planning of already scheduled patients contained in a list of patient structure
        :param schedule:
        :param existing_schedule:
        :param self: list of patients to plot.
        """
        full_schedule = self.schedule + self.existing_schedule
        df = []
        location = None
        for patient, machine, dates in full_schedule:
            if patient.id == 0:
                location = "unknown"
            else:
                location = patient.location.name
            df.append(
                dict(
                    patient_id=patient.id,
                    period_preference=patient.preferred_timerange.name,
                    location=location,
                    linac=self.problem.linacs[machine].name,
                    start=dates[0],
                    end=(
                        dates[1]
                        # dates[1] + timedelta(hours=5)
                        # if not patient.id == 0
                        # else dates[1]
                    ),
                )
            )
        df = pd.DataFrame(df)
        fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="linac",
            color="patient_id",
            category_orders={"patient_id": df["patient_id"].tolist()},
            hover_data=["location", "patient_id", "start", "end", "period_preference"],
        )
        fig.update_yaxes(type="category", autorange="reversed")
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1 day", step="day", stepmode="todate"),
                        dict(count=7, label="1 week", step="day", stepmode="todate"),
                        dict(count=14, label="2 weeks", step="day", stepmode="todate"),
                        dict(count=1, label="1m", step="month", stepmode="todate"),
                        dict(count=2, label="2m", step="month", stepmode="todate"),
                        dict(step="all"),
                    ]
                )
            ),
        )
        fig.update_layout(
            hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell")
        )
        fig.show()
        fig.write_html(f"schedule_by_linac.html")

        df = df.sort_values(by=["patient_id", "start"], ascending=False)
        fig2 = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="patient_id",
            color="linac",
            category_orders={"patient_id": df["patient_id"].tolist()},
        )
        fig2.update_yaxes(type="category", autorange="reversed")
        # fig2.update_yaxes(categoryorder = 'category ascending')
        fig2.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1 day", step="day", stepmode="todate"),
                        dict(count=7, label="1 week", step="day", stepmode="todate"),
                        dict(count=14, label="2 weeks", step="day", stepmode="todate"),
                        dict(count=1, label="1m", step="month", stepmode="todate"),
                        dict(count=2, label="2m", step="month", stepmode="todate"),
                        dict(step="all"),
                    ]
                )
            ),
        )
        fig2.update_layout(
            hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell")
        )
        fig2.show()
        fig2.write_html(f"schedule.html")

        df = df.sort_values(by=["patient_id", "start"], ascending=True)
        df.to_csv("schedule.csv")
