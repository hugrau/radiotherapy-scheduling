import datetime
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Linac:
    """
    Class to represent a Linac, i.e. treatment unit for radiotherapy.
    :param id: id of the linac
    :param name: name of the Linac (e.g. "NOVA 3")
    :param type: type of the Linac (e.g "NOVA")
    :param slot_duration: unit duration of a slot (in minutes) on that linac
    :param start_time: start time of the day
    :param finish_time : finish time of the day
    :param nb_slot: number of slots available for this linac
    """

    id: int
    name: str
    type: str
    slot_duration: int
    start_time: datetime.time
    finish_time: datetime.time
    nb_slot: int

    # We need the two functions below to make sure our object is immutable to use it as a key to a Dict.
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Linac) and self.id == other.id

# TODO = some erratic behavior when this function can also return None: TOMO 4 and TOMO 7 not detected
def get_linac_by_name(name: str, linacs: List[Linac]):
    for linac in linacs:
        if linac.name == name:
            return linac

def get_linac_by_id(linac_id: int, linacs: List[Linac]):
    for linac in linacs:
        if linac.id == linac_id:
            return linac