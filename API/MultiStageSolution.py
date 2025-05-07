from copy import deepcopy
import json
import numpy as np
import random
from typing import List, Tuple, Dict
from datetime import datetime, timedelta, time
from collections import defaultdict
import pandas as pd
import plotly.express as px
from globals import machine_ids

from API.Problem import Problem
from data_structures.Linac import Linac, get_linac_by_name
from data_structures.Patient import Patient
from data_treatments.calendars import generate_business_day_array

def same_day(dt1, dt2):
    """
    Function to compute if two datetimes are the same day.
    """
    return dt1.year == dt2.year and dt1.month == dt2.month and dt1.day == dt2.day


class MultiStageSolution(object):
    """
    Solution class that encapsulates the useful data for solving the problem.
    This multi-stage solution is specifically designed to take into account different scenarios.
    """

    def __init__(
            self,
            problem: Problem,
            nb_scenarios: int
    ):
        self.problem = problem
        self.nb_scenarios = nb_scenarios
        self.certain_patients: List[Patient] = []
        self.uncertain_patients: List[Patient] = []
        self.certain_patients_schedule: List[Tuple[Patient, int, Tuple[datetime, datetime]]] = []
        self.dict_schedules: Dict[int: List[Tuple[Patient, int, Tuple[datetime, datetime]]]] = (
            {s: [] for s in range(self.nb_scenarios)}
        )
        # self.dict_is_valid_solution: Dict[int: bool] = {s: True for s in range(self.nb_scenarios)}
        self.existing_schedule: List[Tuple[Patient, int, Tuple[datetime, datetime]]] = (
            []
        )
        self.machine_occupancy: Dict[int: int] = {},
        self.statistics: Dict = {
            "objective_value": 0,
            "objective_first_stage_earliest": 0,
            "objective_first_stage_treatment_range": 0,
            "objective_second_stage_earliest": 0,
            "objective_second_stage_treatment_range": 0,
            "objective_second_stage_due_dates_penalties": 0,
            "gap": 0,
            "solver_status": "None",
            "solver_time": 0,
        }

    def schedule_to_patient_schedule(self):
        """
        :return:
        """
        patients_schedule = defaultdict(list)
        for p_k, resource_k, (start_k, end_k) in self.certain_patients_schedule:
            patients_schedule[p_k].append(
                (start_k, end_k, self.problem.linacs[resource_k].name)
            )
            # print(f"Patients_schedule: {patients_schedule[p_k]})")
        for p_k, patient_sessions in patients_schedule.items():
            sorted_sessions = sorted(patient_sessions, key=lambda x: x[0])
            patients_schedule[p_k] = sorted_sessions
        return patients_schedule

    def to_patient_objects(self) -> List[Patient]:
        """
        :return:
        """
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
                patient_schedule.append(formatted_session)
            p = deepcopy(patient)
            p.existing_schedule = patient_schedule
            patient_objects.append(p)
        return patient_objects

    def qi_treatment_dates(self, verbose=False):
        """
        :param verbose: Bool
        :return:
        """
        horizon_array = generate_business_day_array(self.problem.horizon_start, self.problem.horizon)

        patient_count_session = {patient.id: 0 for patient in self.certain_patients}
        # The following variable allows to retrieve the delay in days by patients and also give the patient's urgency.
        patient_count_deltas = {
            patient.id: [0, patient.location.urgency.value] for patient in self.certain_patients
        }
        time_count = 0
        time_count_business = 0
        print("QI : Treatment dates : ")
        for session in self.certain_patients_schedule:
            patient, resource, (start, end) = session
            ready_datetime = patient.ready_date
            due_datetime = patient.due_date
            if patient_count_session[patient.id] == 0:
                ready_day = int((ready_datetime - self.problem.horizon_start).days)
                first_session_day = int((start - self.problem.horizon_start).days)
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
        certain_patients = self.schedule_to_patient_schedule()
        certain_patient_count_periods = {
            patient.id: [0, patient.location.urgency.value] for patient in self.certain_patients
        }
        horizon_array = generate_business_day_array(self.problem.horizon_start, self.problem.horizon)
        for patient, patient_sessions in certain_patients.items():
            # If only one session, we don't need to compute the difference.
            if len(patient_sessions) == 1:
                continue
            for k in range(len(patient_sessions) - 1):
                days_kp1 = int((patient_sessions[k + 1][0] - self.problem.horizon_start).days)
                days_k = int((patient_sessions[k][0] - self.problem.horizon_start).days)
                d = [1 for _ in range(days_kp1 - days_k + 1)]
                # print(d, horizon_array[days_k : days_kp1 + 1])
                res = np.dot(d, horizon_array[days_k: days_kp1 + 1])
                if (
                        same_day(patient_sessions[k + 1][0], patient_sessions[k][0])
                        or res > patient.location.period + 1
                ):
                    count_violated += 1
                    certain_patient_count_periods[patient.id][0] += 1
                    days_count += res
        print(
            f"QI : Periods enforcement : \n\tThere are {count_violated} period(s) violated for a total of "
            f"{days_count} day(s)."
        )
        return count_violated, certain_patient_count_periods

    def qi_treatment_ranges(self, verbose=False):
        """

        """
        print("QI : Treatment ranges : ")
        patients_schedule = self.schedule_to_patient_schedule()
        overall_count = 0
        patient_treatment_range_count = {patient.id: 0 for patient in self.certain_patients}
        for patient, schedule in patients_schedule.items():
            rounded_start = schedule[0][0].replace(hour = 0, minute = 0, second = 0, microsecond = 0)
            rounded_end = schedule[-1][0].replace(hour = 0, minute = 0, second = 0, microsecond = 0)
            print(f"\tPatient {patient.id} starts at {schedule[0][0]} and ends at {schedule[-1][0]}"
                  f" Delta = {schedule[-1][0] - schedule[0][0]} and {(rounded_end - rounded_start).days}")
            overall_count += (rounded_end - rounded_start).days
            patient_treatment_range_count[patient.id] = (rounded_end - rounded_start).days
        print(f"\tSum of treatment ranges : {overall_count}")
        return overall_count, patient_treatment_range_count

    def display_scheduling_by_scenario(self, scenario: int, show_in_html: bool = False):
        """
        Display the scheduling of both existing and new patients (certain and uncertain), given a scenario number.
        The option show_in_html can be set to True to display the schedule in an HTML page.
        :param scenario: int
        :param show_in_html: bool
        :param self: list of patients to plot.
        """
        full_schedule = self.dict_schedules[scenario] + self.existing_schedule
        df = []
        for patient, machine, dates in full_schedule:
            if patient.id == 0:
                location = "unknown"
            else:
                location = patient.location.name
            df.append(
                dict(
                    patient_id=patient.id,
                    location=location,
                    linac=self.problem.linacs[machine].name,
                    start=dates[0],
                    end=(
                        # dates[1]
                        dates[1] + timedelta(hours=5)
                        if not patient.id == 0
                        else dates[1]
                    ),
                    is_certain=patient.is_certain,
                    preferred_timerange=patient.preferred_timerange.name,
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
            hover_data=["location", "patient_id", "start", "end", "is_certain", "preferred_timerange"],
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
        if show_in_html:
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
            hover_data=["location", "patient_id", "start", "end", "is_certain", "preferred_timerange"],
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
        if show_in_html:
            fig2.show()
            fig2.write_html(f"schedule.html")

        return fig, fig2


def display_scheduling_fullcalendar(schedule) -> None:
    events = []
    colors = ["#{:02x}{:02x}{:02x}".format(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)) for _ in range(7)]
    for patient, resource, (start, end) in schedule:
        formatted_event = {
            "id": f"{patient.id}",
            "title": f"{patient.id}",
            "start": f"{start.isoformat()}",
            "end": f"{end.isoformat()}",
            "backgroundColor": colors[resource],
            "editable": "true",
            "eventDurationEditable": "true",
        }
        events.append(formatted_event)
    events_json = json.dumps(events)
    # print(events_json)
    html = """
    <!DOCTYPE html>
        <html>
        <head>
            <title>FullCalendar Example</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"
                  integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
        
            <script src='https://cdn.jsdelivr.net/npm/@fullcalendar/core@6.1.15/index.global.min.js'></script>
            <script src='https://cdn.jsdelivr.net/npm/@fullcalendar/daygrid@6.1.15/index.global.min.js'></script>
            <script src="https://cdn.jsdelivr.net/npm/@fullcalendar/timegrid@6.1.15/index.global.min.js"></script>
            <script>
        
              document.addEventListener('DOMContentLoaded', function() {
                var calendarEl = document.getElementById('calendar');
                var calendar = new FullCalendar.Calendar(calendarEl, {
                    height: 1000,
                    // plugins: [timeGridPlugin],
                    initialView: 'timeGridWeek', // or your preferred initial view
                    locale: 'fr',
                    // timeZone: 'UTC',
                    themeSystem: 'bootstrap5',
                    nowIndicator: true,
                    slotMinTime: '06:00:00',   // Start displaying at 6:00 AM
                    slotMaxTime: '22:00:00',    // Stop displaying at 10:00 PM
                    slotDuration: "00:06:00", // X-minute slots
                    snapDuration: "00:06:00",
        
                    events: """ + events_json + """
                });
                calendar.render();
              });
        
            </script>
          </head>
        <body>
        <div class="card">
            <div class="card-header">
        
            </div>
            <div class="card-body">
                <div id="calendar"></div>
            </div>
            <div class="card-footer">
        
            </div>
        </div>
        </body>
        </html>"""
    with open("schedule_fullcalendar.html", "w") as f:
        f.write(html)


def display_scheduling_fullcalendar_by_resource(schedule, linac_id: int) -> None:
    events = []
    color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for patient, resource, (start, end) in schedule:
        if resource == linac_id:
            formatted_event = {
                "id": f"{patient.id}",
                "title": f"{patient.id}",
                "start": f"{start.isoformat()}",
                "end": f"{end.isoformat()}",
                "backgroundColor": color,
                "editable": "true",
                "eventDurationEditable": "true",
            }
            events.append(formatted_event)
    events_json = json.dumps(events)
    # print(events_json)
    html = """
    <!DOCTYPE html>
        <html>
        <head>
            <title>FullCalendar Example</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"
                  integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">

            <script src='https://cdn.jsdelivr.net/npm/@fullcalendar/core@6.1.15/index.global.min.js'></script>
            <script src='https://cdn.jsdelivr.net/npm/@fullcalendar/daygrid@6.1.15/index.global.min.js'></script>
            <script src="https://cdn.jsdelivr.net/npm/@fullcalendar/timegrid@6.1.15/index.global.min.js"></script>
            <script>

              document.addEventListener('DOMContentLoaded', function() {
                var calendarEl = document.getElementById('calendar');
                var calendar = new FullCalendar.Calendar(calendarEl, {
                    height: 1000,
                    // plugins: [timeGridPlugin],
                    initialView: 'timeGridWeek', // or your preferred initial view
                    locale: 'fr',
                    // timeZone: 'UTC',
                    themeSystem: 'bootstrap5',
                    nowIndicator: true,
                    slotMinTime: '06:00:00',   // Start displaying at 6:00 AM
                    slotMaxTime: '22:00:00',    // Stop displaying at 10:00 PM
                    slotDuration: "00:06:00", // X-minute slots
                    snapDuration: "00:06:00",

                    events: """ + events_json + """
                });
                calendar.render();
              });

            </script>
          </head>
        <body>
        <div class="card">
            <div class="card-header">
            """ + machine_ids[linac_id] + """
            </div>
            <div class="card-body">
                <div id="calendar"></div>
            </div>
            <div class="card-footer">

            </div>
        </div>
        </body>
        </html>"""
    with open(f"schedule_fullcalendar_{machine_ids[linac_id]}.html", "w") as f:
        f.write(html)
