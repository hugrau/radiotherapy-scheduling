from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set

from data_structures.Linac import Linac


class Urgency(Enum):
    """
    Enumeration of the urgency level of a patient according to location.
    """

    VERY_URGENT = 1
    URGENT = 2
    NORMAL = 3
    NOT_SPECIFIED = 4


class Priority(Enum):
    """
    Enumeration of the priority level of a linac according to location.
    """

    FORBIDDEN = 0
    ACCEPTED = 1
    PREFERRED = 2


@dataclass
class Location:
    """
    :param name: technical name of the location (str)
    :param urgency: urgency level of that location
    :param period: number of days between consecutive session (int)
    :param linacs_by_priority: list of linac associated with each priority level
    :param duration_by_linac: duration of the session (in minutes) associated with each compatible linac
    """

    # id: int
    name: str
    urgency: Urgency
    period: int
    linacs_by_priority: Dict[Priority, List[Linac]]
    duration_by_linac: Dict[Linac, int]

    def get_nb_alternative_linac(self) -> int:
        s = set()
        for prio in [Priority.PREFERRED, Priority.ACCEPTED]:
            s.update(self.linacs_by_priority.get(prio, []))
        return len(s)

    def get_potential_linacs(self) -> Set[Linac]:
        s = set()
        for prio in [Priority.PREFERRED, Priority.ACCEPTED]:
            s.update(self.linacs_by_priority.get(prio, []))
        return s
