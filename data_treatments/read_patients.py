import datetime
import pandas as pd
from typing import List, Dict

from globals import oncopole_linacs
from data_treatments.parsing_utils import apply_lower, apply_unidecode, apply_replace
from data_treatments.read_locations import file_to_locations
from data_structures.Patient import Patient
from data_structures.Location import Location


def read_prescription_file(
    filename: str, column_name: str = "location"
) -> pd.DataFrame:
    """
    Function to clean the prescription file and drop Na values of column 'SITE'.
    It means if the prescription file has a Na value in column SITE then the row is dropped.
    :param filename: name of the prescription file.
    :param column_name: column name to drop Na values from.
    :return:
    """
    df = pd.read_csv(filename)
    df = df.map(apply_lower)
    df = df.map(apply_unidecode)
    df = df.map(apply_replace)
    # Duplicates happen in the prescription file.
    df.drop_duplicates(inplace=True)

    null_counts = df[column_name].isna().sum()
    df_cleaned = df.dropna(subset=[column_name])
    df_cleaned.loc[:, "scanner_date"] = pd.to_datetime(
        df_cleaned["scanner_date"], dayfirst=False
    )
    # The following line is to remove entries with same 'IPP' but with different scanner dates, we keep the latest.
    df_cleaned = df_cleaned.sort_values(by="scanner_date").groupby("ipp").tail(1)
    df_cleaned.loc[:, "prescription_date"] = pd.to_datetime(
        df_cleaned["prescription_date"], dayfirst=False
    )

    df_cleaned.loc[:, "scanner_date"] = df_cleaned["scanner_date"].apply(
        lambda x: x.strftime("%Y-%m-%d")
    )
    df_cleaned.loc[:, "prescription_date"] = df_cleaned["prescription_date"].apply(
        lambda x: x.strftime("%Y-%m-%d")
    )

    # df_cleaned.to_csv('../data/prescription_cleaned.csv', index=False)
    print(
        f"\033[33mWarning::Dropping {null_counts} entries from prescription file "
        f"for missing {column_name} values.\033[0m\n"
    )

    return df_cleaned


def prescription_to_patients(
    filename: str, locations: Dict[str, Location]
) -> List[Patient]:
    """

    :param filename:
    :param locations:
    :return: list of Patient
    """
    new_patients = []

    df = read_prescription_file(filename, column_name="location")

    # plt.show()
    # print(df.columns)
    for index, row in df.iterrows():
        # print(row['SITE'])
        try:
            location = locations[row["location"]]
        except KeyError:
            print(
                f"\033[31mLocation \"{row['location']}\" of "
                f"patient {row['ipp']} cannot be found in locations file.\033[0m"
            )
            continue

        patient = Patient(
            id=row["ipp"],
            location=location,
            ready_date=datetime.datetime.strptime(row["prescription_date"], "%Y-%m-%d")
            + datetime.timedelta(days=8),
            due_date=datetime.datetime.strptime(row["prescription_date"], "%Y-%m-%d")
            + datetime.timedelta(days=20),
            nb_fractions=int(row["nb_fractions"]),
            is_new=True,
            has_started=False,
            existing_schedule=[],
        )
        if patient.ready_date.weekday() >= 5:
            patient.ready_date += datetime.timedelta(
                days=7 - patient.ready_date.weekday()
            )
        if patient.due_date.weekday() >= 5:
            patient.due_date += datetime.timedelta(days=7 - patient.due_date.weekday())

        new_patients.append(patient)
    return new_patients


if __name__ == "__main__":
    # TESTS
    locs = file_to_locations(
        "../data/Fichier_Repartition_Hackathon_070723.xlsx", oncopole_linacs
    )
    # print(locs)
    patients = prescription_to_patients("../data/pflow08072024_13072024.csv", locs)
    for patient in patients:
        print(patient)
