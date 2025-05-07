from ortools.sat.python import cp_model

from API.Problem import Problem
import numpy as np
import pandas as pd
from globals import *
import datetime
from datetime import timedelta
from workalendar.europe import France
from typing import Dict, Tuple, Any


def is_business_day(date):
    """
    Check if the given date is a business day in France i.e. not a weekend day or a public holiday.
    :param date: the considered date.
    :return: boolean indicating whether the given date is a business day.
    """
    # Check if the date is a weekend (Saturday or Sunday)
    if date.dayofweek >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check if the date is a public holiday in France
    cal = France()
    if cal.is_working_day(date):
        return True
    else:
        return False


def generate_business_day_array(start_date: datetime, num_days: int) -> list[bool]:
    """

    :param start_date: datetime
    :param num_days: int
    :return: list[bool]
    """
    date_range = pd.date_range(start=start_date, periods=num_days)
    # Generate an array of booleans indicating if each date is a business day
    business_day_array = [is_business_day(date) for date in date_range]

    return business_day_array


def existing_schedule_to_list(problem: Problem):
    """
    Existing schedule is a list of Patient objects where we can find the already existing schedule,
    we transform it into a list of tuples.
    :param problem: Problem, the problem object.
    :return: List[patient_session], list of patient_sessions namedtuples.
    """
    existing_schedule = []
    for patient in problem.existing_schedule:
        # Check if no conflict with prescription file.
        for session in patient.existing_schedule:
            session_start = session[1][0]
            session_end = session[1][1]
            delta_start = (session_start - problem.horizon_start).total_seconds() / 60
            delta_end = (session_end - problem.horizon_start).total_seconds() / 60
            resource = linacs_ids[session[0].id]
            if session_start >= problem.horizon_start:
                # print(f"Linac : {session[0].name} - date : {session[1][0]} "
                #      f"- minutes from horizon start : {delta_end-delta_start} - linac : {linacs_ids[session[0].id]}")
                existing_schedule.append(
                    patient_session(patient, resource, (session_start, session_end))
                )
    return existing_schedule


def get_unavailable_slots_by_linac(linac_id, horizon_start, existing_schedule):
    """
    Given a linac id and horizon start and existing schedule, return a list of unavailable
    :param linac_id:
    :param horizon_start:
    :param existing_schedule:
    :return:
    """
    unavailable_slots = []
    for patient in existing_schedule:
        for session in patient.existing_schedule:
            session_start = session[1][0]
            session_end = session[1][1]
            # delta_start = (session_start - problem.horizon_start).total_seconds() / 60
            # delta_end = ((session_end - problem.horizon_start).total_seconds() / 60)
            resource = linacs_ids[session[0].id]
            if (
                linac_id == oncopole_linacs[resource].id
                and session_start >= horizon_start
            ):
                unavailable_slots.append(
                    patient_session(patient, resource, (session_start, session_end))
                )
    # We sort the schedule by dates of sessions start, important for merging.
    unavailable_slots = sorted(unavailable_slots, key=lambda x: x.dates[0])
    return unavailable_slots


def merge_unavailable_slots_by_linac(linac_id: int, horizon_start, existing_schedule):
    """
    Given a linac id and horizon start and existing schedule, merge unavailable slots and returns them as a list of
    namedtuples patient_session.
    :param linac_id: int, the id of the linac.
    :param horizon_start: datetime, the start of the horizon interval.
    :param existing_schedule: List[Patient], the existing schedule as Patient objects.
    :return: List[patient_session], the merged schedule as a list of patient_session.
    """
    linac_unavailable_schedule = get_unavailable_slots_by_linac(
        linac_id=linac_id,
        horizon_start=horizon_start,
        existing_schedule=existing_schedule,
    )
    # concatenated_schedule = [
    #     session for session in linac_unavailable_schedule
    #     if session.dates[0] >= problem.horizon_start
    # ]
    if len(linac_unavailable_schedule) > 0:
        merged_schedule = [linac_unavailable_schedule[0]]
        for s in range(len(linac_unavailable_schedule) - 1):
            if (
                linac_unavailable_schedule[s + 1].dates[0] - timedelta(minutes=1)
                <= merged_schedule[-1].dates[1]
            ):
                merged_schedule[-1] = patient_session(
                    merged_schedule[
                        -1
                    ].patient,  # patient id doesn't matter in that case they both worth 0
                    # by definition in read_timeplanner
                    merged_schedule[-1].machine,  # machine is the same
                    (
                        merged_schedule[-1].dates[0],
                        max(
                            merged_schedule[-1].dates[1],
                            linac_unavailable_schedule[s + 1].dates[1],
                        ),
                    ),
                )
            else:
                merged_schedule.append(linac_unavailable_schedule[s + 1])
        # print(merged_schedule[-1].patient)
        # print(merged_schedule[-1].machine)
        # print(merged_schedule[-1].dates)
        return merged_schedule
    else:
        return []


def merged_schedule_to_list(
        linacs: List[Linac],
        horizon_start: datetime,
        existing_schedule
):
    """

    """
    schedule = []
    for linac in linacs:
        schedule = schedule + merge_unavailable_slots_by_linac(
            linac_id=linac.id,
            horizon_start=horizon_start,
            existing_schedule=existing_schedule,
        )
    return schedule


def compute_unavailable_slots_by_machine(
    existing_schedule,
    start_horizon: datetime,
    days_horizon: int,
    model: cp_model.CpModel,
    machine_id: int,
) -> Tuple[List[Any], List[Any]]:
    """
    :param existing_schedule:
    :param start_horizon:
    :param days_horizon:
    :param model:
    :param machine_id:
    :return: Tuple[List[Any], List[Any]]
    """
    machine_index = linacs_ids[machine_id]
    hrs_start, mns_start = (
        oncopole_linacs[machine_index].start_time.hour * 60,
        oncopole_linacs[machine_index].start_time.minute,
    )
    hrs_end, mns_end = (
        oncopole_linacs[machine_index].finish_time.hour * 60,
        oncopole_linacs[machine_index].finish_time.minute,
    )
    business_days = generate_business_day_array(start_horizon, days_horizon)
    unavailable_intervals_weeks = []
    for i, is_day_worked in enumerate(business_days):
        # Weekends unavailabilities
        if not is_day_worked:
            unavailable_intervals_weeks.append(
                model.NewIntervalVar(
                    start=i * 1440,
                    size=1440,
                    end=(i + 1) * 1440,
                    name=f"week_end_{i}_for_machine{machine_id}",
                )
            )
        # Weekdays unavailabilities
        else:
            unavailable_intervals_weeks.append(
                model.NewIntervalVar(
                    start=i * 1440,
                    size=hrs_start + mns_start,
                    end=i * 1440 + hrs_start + mns_start,
                    name=f"Start_day_{i}_for_machine{machine_id}",
                )
            )
            unavailable_intervals_weeks.append(
                model.NewIntervalVar(
                    start=i * 1440 + hrs_end + mns_end,
                    size=1440 - (hrs_end + mns_end),
                    end=(i + 1) * 1440,
                    name=f"End_day_{i}_for_machine{machine_id}",
                )
            )
    unavailable_intervals = []
    merged_schedule_linac = merge_unavailable_slots_by_linac(
        machine_id, start_horizon, existing_schedule
    )
    for patient, machine, dates in merged_schedule_linac:
        session_start = dates[0]
        session_end = dates[1]
        delta_start = int((session_start - start_horizon).total_seconds() / 60)
        delta_end = int((session_end - start_horizon).total_seconds() / 60)
        # session_duration = delta_end - delta_start
        unavailable_intervals.append(
            model.NewIntervalVar(
                start=delta_start,
                size=delta_end - delta_start,
                end=delta_end,
                name=f"Unavailable_slot_for_machine{machine_id}",
            )
        )
    return unavailable_intervals_weeks, unavailable_intervals


def compute_unavailable_slots_by_machine_list_tuple(
    existing_schedule,
    start_horizon,
    days_horizon,
    machine_id,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Same as compute_unavailable_slots_by_machine but not returning list of interval var variable,
    only list of tuple (start, duration, end) being minutes values where the machine is used or closed.
    """
    machine_index = linacs_ids[machine_id]
    hrs_start, mns_start = (
        oncopole_linacs[machine_index].start_time.hour * 60,
        oncopole_linacs[machine_index].start_time.minute,
    )
    hrs_end, mns_end = (
        oncopole_linacs[machine_index].finish_time.hour * 60,
        oncopole_linacs[machine_index].finish_time.minute,
    )
    business_days = generate_business_day_array(start_horizon, days_horizon)
    unavailable_intervals_weeks = []
    for i, is_day_worked in enumerate(business_days):
        # Weekends
        if not is_day_worked:
            unavailable_intervals_weeks.append((i * 1440, 1440, (i + 1) * 1440))
        # Weekdays
        else:
            # closed between midnight and opening time
            unavailable_intervals_weeks.append(
                (i * 1440, hrs_start + mns_start, i * 1440 + hrs_start + mns_start)
            )
            # closed between closing time and midnight
            unavailable_intervals_weeks.append(
                (
                    i * 1440 + hrs_end + mns_end,
                    1440 - (hrs_end + mns_end),
                    (i + 1) * 1440,
                )
            )
    unavailable_intervals = []
    # Compute merged unavailability slots
    merged_schedule_linac = merge_unavailable_slots_by_linac(
        machine_id, start_horizon, existing_schedule
    )
    for patient, machine, dates in merged_schedule_linac:
        session_start = dates[0]
        session_end = dates[1]
        delta_start = int((session_start - start_horizon).total_seconds() / 60)
        delta_end = int((session_end - start_horizon).total_seconds() / 60)
        # session_duration = delta_end - delta_start
        unavailable_intervals.append((delta_start, delta_end - delta_start, delta_end))
    return unavailable_intervals_weeks, unavailable_intervals
