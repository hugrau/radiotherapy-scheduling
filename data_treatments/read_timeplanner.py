import os
import pandas as pd
from typing import List
from pathlib import Path

from data_structures.Linac import get_linac_by_name
from data_structures.Patient import Patient
from data_treatments.parsing_utils import insert_space_before_numbers
from globals import oncopole_linacs


def clean_timeplanner(filepath: str) -> pd.DataFrame:
    """
    Clean the timeplanner file we retrieve from the OIS. Cleaning essentially consists in proper formatting of machine
    names and sessions datetimes.
    :param filepath: str, path to file
    :return: pd.DataFrame
    """
    df = pd.read_csv(filepath)
    for index, row in df.iterrows():
        if row["linac"] == "0210462":
            row["linac"] = "TOMO 4"
        elif row["linac"] == "0210471":
            row["linac"] = "TOMO 7"
        df.loc[index, "linac"] = insert_space_before_numbers(row["linac"])

        df.loc[index, "start"] = (
            pd.to_datetime(row["start"], format="ISO8601")
            .tz_localize(None)
            .to_pydatetime()
        )
        # df.loc[index, "start"] = row["start"].tz_localize(None)
        # df.loc[index, "start"] = row["start"].to_pydatetime()

        df.loc[index, "end"] = (
            pd.to_datetime(row["end"], format="ISO8601")
            .tz_localize(None)
            .to_pydatetime()
        )
        # df.loc[index, "end"] = row["end"].tz_localize(None)
        # df.loc[index, "end"] = row["end"].to_pydatetime()
    # basename = os.path.basename(filepath)  # Get the base name (file + extension)
    # filename, ext = os.path.splitext(basename)  # Split into filename and extension
    # df.to_csv(f"cleaned_{filename}.csv", index=False)
    return df


def parse_timeplanner(filepath: str) -> List[Patient]:
    """
    Cleans the timeplanner file by calling the clean_timeplanner function and truns sessions into Patient objects.
    :param filepath: str, path to file
    :return: List[Patient], a list of Patient objects.
    """
    df = clean_timeplanner(filepath)
    # print(df)
    patients = []
    sessions = []
    for _, row in df.iterrows():
        # patient_id = row["patient_id"] if row["patient_id"] else 0
        linac = get_linac_by_name(row["linac"], oncopole_linacs)
        sessions.append((linac, (row["start"], row["end"])))
        # print(row)
    schedule = sorted(sessions, key=lambda x: x[1][0])
    patients.append(
        Patient(
            id=0,  # int(row["PATIENT_ID"])//2,
            location=None,
            ready_date=None,
            due_date=None,
            nb_fractions=len(schedule),
            is_new=False,
            has_started=None,
            existing_schedule=schedule,
        )
    )
    return patients


def format_timeplanner_for_fullcalendar(filepath: str, machine_name: str):
    df = clean_timeplanner(filepath)

    formatted_events = []
    for _, event in df.iterrows():
        if event["linac"] == machine_name:
            formatted_event = {
                "title": f"busy-slot",
                "start": event["start"].strftime("%Y-%m-%d %H:%M:%S"),
                "end": event["end"].strftime("%Y-%m-%d %H:%M:%S"),
                # "backgroundColor": "blue",
            }
            # print(formatted_event)

            formatted_events.append(formatted_event)

    return formatted_events


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / Path("data")
    filepath = str(DATA_DIR / "schedule_080724.csv")

    patients = parse_timeplanner(filepath)
    print(format_timeplanner_for_fullcalendar(filepath, "TOMO 4"))
    # print(patients)
