import pandas as pd
from typing import List, Dict, Optional
from data_structures.Location import Location, Urgency, Priority
from data_structures.Linac import Linac

# from Oncopole.constants import linac_list as default_linacs

from data_treatments.parsing_utils import *


def read_locations_file(
    filename: str,
    sheet_name_1: str = "Repartition et priorité",
    sheet_name_2: str = "Temps des séances en min",
):
    df1 = pd.read_excel(filename, sheet_name=sheet_name_1)
    df2 = pd.read_excel(filename, sheet_name=sheet_name_2)
    # Careful the first row of df2 is not location related data.

    # Formatting the strings in the data frames.
    df1 = df1.map(apply_lower)
    df1 = df1.map(apply_unidecode)
    df1 = df1.map(apply_replace)

    df2 = df2.map(apply_lower)
    df2 = df2.map(apply_unidecode)
    df2 = df2.map(apply_replace)

    colnames_df1 = list(df1.columns)
    colnames_df2 = list(df2.columns)

    # Issue between location file and prescription file
    df1[colnames_df1[0]] = df1[colnames_df1[0]].apply(insert_space_before_numbers)
    df2[colnames_df2[0]] = df2[colnames_df2[0]].apply(insert_space_before_numbers)

    # Treatment necessary to homogenize linac names from colnames and given by user
    # issue : NOVA3 and NOVA5 in columns instead of NOVA 3 and NOVA 5
    for k in range(3, len(colnames_df1)):
        colnames_df1[k] = separate_letters_and_digits(colnames_df1[k])
        colnames_df2[k - 1] = separate_letters_and_digits(colnames_df2[k - 1])

    # print(colnames_df1, colnames_df2)

    df1.columns = colnames_df1
    df2.columns = colnames_df2

    return df1, df2


# TODO = put them as methods in Location class
def int_to_urgency(value: int) -> Urgency:
    """
    Converts an integer value into an Urgency object.
    :param value: integer value
    :return: Urgency object
    """
    if value == 1:
        return Urgency.VERY_URGENT
    elif value == 2:
        return Urgency.URGENT
    elif value == 3:
        return Urgency.NORMAL
    else:
        return Urgency.NOT_SPECIFIED


def int_to_priority(value: int) -> Priority:
    """
    Converts an integer value into a Priority object.
    :param value: integer value
    :return: Priority object
    """
    if value == 0:
        return Priority.FORBIDDEN
    elif value == 1:
        return Priority.ACCEPTED
    elif value == 2:
        return Priority.PREFERRED


def file_to_locations(
    filename: str, linacs: Optional[List[Linac]] = None
) -> Dict[str, Location]:

    priorities_df, time_df = read_locations_file(filename)
    locations = []
    dict_locations = {}

    colnames_priorities = list(priorities_df.columns)
    colnames_time = list(time_df.columns)

    for i in range(priorities_df.shape[0]):

        name = priorities_df.loc[i, colnames_priorities[0]]
        urgency = int_to_urgency(priorities_df.loc[i, colnames_priorities[1]])
        period = priorities_df.loc[i, colnames_priorities[2]]
        # print(f"{name}, {urgency}, {period}")

        linacs_by_priority = {
            Priority.PREFERRED: [],
            Priority.ACCEPTED: [],
            Priority.FORBIDDEN: [],
        }
        for linac in linacs:
            priority = int_to_priority(priorities_df.loc[i, linac.name])
            linacs_by_priority[priority].append(linac)
        # print(linacs_by_priority)

        duration_by_linac = {}
        for linac in linacs:
            if linac not in linacs_by_priority[Priority.FORBIDDEN]:
                duration_by_linac[linac] = time_df.loc[i + 1, linac.name]

        # print(duration_by_linac)
        location = Location(
            name=name,
            urgency=urgency,
            period=period,
            linacs_by_priority=linacs_by_priority,
            duration_by_linac=duration_by_linac,
        )
        locations.append(location)
        dict_locations[name] = location
    return dict_locations
