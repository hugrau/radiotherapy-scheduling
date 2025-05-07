from typing import List
import datetime
import collections
from pathlib import Path
from collections import namedtuple

from data_treatments.read_locations import file_to_locations
from data_structures.Linac import Linac
from data_structures.Patient import Patient

patient_session = collections.namedtuple(
    "patient_session", "patient machine dates"
)  # dates=tuple


# Data files paths and metadata.
DATA_DIR = Path(__file__).resolve().parent / Path("data")
locations_file_name = Path("Fichier_Repartition_Hackathon_070723.xlsx")
proportions_file_name = str(DATA_DIR / "proportion_data.csv")


oncopole_linacs = [
    Linac(
        id=0,
        name="TOMO 2",
        type="TOMO",
        slot_duration=15,
        start_time=datetime.time(8, 30, 0),
        finish_time=datetime.time(19, 54, 0),
        nb_slot=52,
    ),
    Linac(
        id=1,
        name="NOVA 3",
        type="NOVA",
        slot_duration=12,
        start_time=datetime.time(7, 48, 0),
        finish_time=datetime.time(19, 48, 0),
        nb_slot=60,
    ),
    Linac(
        id=2,
        name="TOMO 4",
        type="TOMO",
        slot_duration=20,
        start_time=datetime.time(7, 40, 0),
        finish_time=datetime.time(20, 0, 0),
        nb_slot=39,
    ),
    Linac(
        id=3,
        name="NOVA 5",
        type="NOVA",
        slot_duration=12,
        start_time=datetime.time(7, 48, 0),
        finish_time=datetime.time(19, 48, 0),
        nb_slot=60,
    ),
    Linac(
        id=4,
        name="HALCYON 6",
        type="HALCYON",
        slot_duration=12,
        start_time=datetime.time(7, 48, 0),
        finish_time=datetime.time(19, 48, 0),
        nb_slot=60,
    ),
    Linac(
        id=5,
        name="TOMO 7",
        type="TOMO",
        slot_duration=20,
        start_time=datetime.time(8, 0, 0),
        finish_time=datetime.time(15, 40, 0),
        nb_slot=26,
    ),
    Linac(
        id=6,
        name="HALCYON 8",
        type="HALCYON",
        slot_duration=12,
        start_time=datetime.time(7, 48, 0),
        finish_time=datetime.time(19, 48, 0),
        nb_slot=60,
    ),
]

# Dict with location name as key and Location object as value
# allows to retrieve a location by its name.
LOCATIONS = file_to_locations(str(DATA_DIR / locations_file_name), oncopole_linacs)

# To get the index of linac with id linac_id in problem_linacs list.
linacs_ids = {oncopole_linacs[i].id: i for i in range(len(oncopole_linacs))}

# to get machine name by id.
machine_ids = {oncopole_linacs[i].id: oncopole_linacs[i].name for i in range(len(oncopole_linacs))}
