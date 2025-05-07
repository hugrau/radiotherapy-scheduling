import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

from data_structures.Location import Linac
from data_structures.Location import Location


class PreferredTimerange(Enum):
    """
    Enumeration of the preferred time range for a radiotherapy session.
    """

    EARLY_MORNING = 1
    MORNING = 2
    AFTERNOON = 3
    EVENING = 4
    NO_PREFERENCE = 0


@dataclass
class Patient:
    """
    dataclass representing a unique patient. If is_new is False and has_started is False it means that the patient have
    already been scheduled but haven't started yet his sessions
    :param id: id of the patient
    :param location: Location object of the patient's location
    :param ready_date: ready date of the patient, as a datetime object
    :param due_date: due date of the patient, as a datetime object
    :param nb_fractions: number of fraction needed for the patient
    :param is_new: if True it corresponds to the patient coming in the planning WF2 box
    :param has_started: if True, the patient already have a schedule

    :param preferred_timerange: PreferredTimerange enum representing the preferred time range
    :param existing_schedule: sessions of the patient if he has already been scheduled, else this is an empty list
        e.g. for one session scheduled : [( Linac object,(datetime.time(8,0,0), datetime.time(8,12,0)))]
    :param user_excluded_machines: List of machines excluded by the user as list of Linac objects
    :param creation_datetime: Creation date of the patient as a datetime object
    :param has_tsu: boolean that is True if the patient has a Treatment Start U...
    """

    id: int
    location: Location
    ready_date: datetime.datetime
    due_date: datetime.datetime
    nb_fractions: int
    is_new: bool
    has_started: bool
    is_certain: bool = False

    scanner_date: Optional[datetime] = None
    medical_validation_date: Optional[datetime] = None
    preferred_timerange: Optional[PreferredTimerange] = PreferredTimerange.NO_PREFERENCE
    existing_schedule: Optional[
        List[Tuple[Linac, Tuple[datetime.datetime, datetime.datetime]]]
    ] = None
    user_excluded_machines: Optional[List[Linac]] = None
    creation_datetime: Optional[datetime] = None
    has_tsu: Optional[bool] = False

    # We need the two following functions below to make sure our object is immutable to use it as a key to a Dict.
    def __hash__(self):
        return hash(self.id)

    # See previous comment.
    def __eq__(self, other):
        return self.id == other.id

    def to_dict(self):
        return {
            "id": self.id,
            "location": self.location.name,
            "ready_date": self.ready_date,
            "due_date": self.due_date,
            "nb_fractions": self.nb_fractions,
        }
