import numpy as np
import pandas as pd
import sys
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

from API.Problem import Problem
from data_structures.Location import Location
from data_structures.Patient import Patient, PreferredTimerange
from datetime import datetime, timedelta

from data_treatments.read_proportions import simulate_discrete_law
from data_treatments.calendars import generate_business_day_array


def generate_random_patients(
        proportions_df: pd.DataFrame,
        dict_locations: Dict[str, Location],
        date: datetime,
        nb_patients: int = 15,
) -> list[Patient]:
    # TODO : generate over a time horizon more than over a patient number.
    patients = []

    # On tire aléatoirement sans remise pour choisir les id, comme ça on s'assure de ne pas programmer
    # plusieurs fois le même patient.
    patients_id = list(np.random.choice(np.arange(10000), nb_patients, replace=False) + 10000)

    # On tire aléatoirement avec remise.
    df = proportions_df  # read_proportions_file('../data/proportion_data.csv')
    patients_locations = [simulate_discrete_law(df) for _ in range(nb_patients)]

    for i in range(nb_patients):
        patient_id = int(patients_id[i])
        patient_location = dict_locations[patients_locations[i]]
        patient_location.period = int(patient_location.period)
        # num = np.random.poisson(10, 1)
        # TODO : better way to generate nb_fractions but no information !
        # TODO : proportion of TSU ?
        patient = Patient(
            id=patient_id,
            location=patient_location,
            ready_date=date,
            due_date=date,
            nb_fractions=15,
            is_new=True,
            has_started=False,
            existing_schedule=[]
        )
        patients.append(patient)

    return patients


def generate_hidden_truth(
        start_date: datetime,
        number_days_simulated: int,
        average_patients_per_day: int,
        proportions_df: pd.DataFrame,
        dict_locations: Dict[str, Location],
) -> pd.DataFrame:
    """
    :param start_date: datetime
    :param number_days_simulated: int
    :param average_patients_per_day: int
    :param proportions_df: pd.DataFrame
    :param dict_locations: Dict[str, Location]
    :return: pd.DataFrame
    """
    scenario = []
    rng = np.random.default_rng()
    business_days = generate_business_day_array(
        start_date=start_date,
        num_days=number_days_simulated,
    )
    df = proportions_df

    k = 0
    for n, b in enumerate(business_days):
        # Draw the random number of daily patients (Poisson's law).
        nb_daily_patients = b*rng.poisson(lam=average_patients_per_day)
        patients_ids = [n*100 + i for i in range(nb_daily_patients)]
        # We draw the locations according to proportions
        patients_locations = [simulate_discrete_law(df) for _ in range(nb_daily_patients)]  # is a list of Location objects.
        for i in range(nb_daily_patients):
            patient_id = patients_ids[i]
            location = patients_locations[i]
            urgency = dict_locations[patients_locations[i]].urgency.value
            start_datetime_patient = start_date + timedelta(
                days=n
            )
            validation_variation = int(rng.normal(loc=4.0, scale=2.0))
            validation_datetime_patient = start_datetime_patient + timedelta(
                days=validation_variation
            )
            # Draw a half-day arrival for that patient 1 = morning, 2 = afternoon.
            half_day = rng.integers(low=1, high=3)
            # Check if the start datetime is not a weekend day (ideally should be made for public holidays).
            if start_datetime_patient.weekday() > 5:
                start_datetime_patient = start_datetime_patient + timedelta(days=2)
            # Check if the validation datetime is not a weekend day (ideally should be made for public holidays).
            if validation_datetime_patient.weekday() > 5:
                validation_datetime_patient = validation_datetime_patient + timedelta(days=2)
            k += 1
            scenario.append({"id_day": n, "scanner_date": start_datetime_patient, "validation_date": validation_datetime_patient, "half_day": half_day, "patient_id": patient_id, "patient_location": location, "urgency": urgency})
    df = pd.DataFrame.from_records(scenario)
    # df.to_csv("instance.csv", index=False)

    # Statistics of the generated scenario.
    arrivals = df.groupby(["id_day"]).count()
    f, ax = plt.subplots(figsize=(12, 8))
    sns.despine(f)

    sns.histplot(
        df,
        x="id_day", hue="urgency",
        bins=number_days_simulated//7,
        multiple="stack",
        palette="light:b",
        edgecolor=".3",
        linewidth=.5,
    )
    sns.lineplot(
        arrivals, x="id_day", y="scanner_date", color="red"
    )
    # ax.set_xticks([k for k in range(number_days_simulated//30)])
    # ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    # plt.axhline(y=arrivals["scanner_date"].mean(), color='red', label=f'Moyenne {arrivals["scanner_date"].mean():.2f}')
    # sns.barplot(arrivals, x="id_day", y="scanner_date")
    plt.show()

    # Create a hash for the instance file, the base string for the hash function is the concatenation of validation dates' counts over
    # the horizon. Scanner dates' counts might be the same and shouldn't be used for hash.
    hash_string = ""
    for index, row in arrivals.iterrows():
        hash_string += str(row["validation_date"])
    hash_instance = hash(hash_string) % ((sys.maxsize + 1) * 2)

    # Record the file.
    df.to_csv(f"./instance_{hash_instance}.csv", index=False)
    return df


def generate_patients_from_instance(
        instance_file: str,
        dict_locations: Dict[str, Location]
) -> list[Patient]:
    """
    Generate fake patients from an instance file where scanner and medical validation dates are given.
    :param instance_file:
    :param dict_locations:
    :return: list[Patient]
    """
    rng = np.random.default_rng()
    generated_patients = []

    patients_flux = pd.read_csv(instance_file)
    nb_patients = patients_flux.shape[0]

    for index, row in patients_flux.iterrows():
        patient_id = int(row["patient_id"])
        patient_location = dict_locations[row["patient_location"]]
        scanner_date = datetime.strptime(row["scanner_date"], "%Y-%m-%d")
        medical_validation_date = datetime.strptime(row["validation_date"], "%Y-%m-%d")

        # TODO: generate an appropriate number of locations.
        # If the period is <= 1 then a patient has on average 15 fractions, but if period is 2 then it is 4 on average.
        nb_fractions = 15 if patient_location.period <= 1 else 4
        # The ready date is here chosen as 48h after the medical validation date.
        # Caution : .weekday = 0 to 6 but .isoweekday = 1 to 7
        ready_date = medical_validation_date+timedelta(days=2) if (medical_validation_date+timedelta(days=2)).weekday() < 5 \
            else medical_validation_date+timedelta(days=4)
        # The due date is ready date + 12 days (or scanner date + 20 days ?)
        # Checking if the date is not on a weekend.
        due_date = scanner_date + timedelta(days=20) if (scanner_date + timedelta(days=20)).weekday() < 5 \
            else scanner_date + timedelta(days=22)
        patient = Patient(
            id=patient_id,
            location=patient_location,
            ready_date=ready_date,
            due_date=due_date,
            nb_fractions=nb_fractions,
            is_new=True,
            has_started=False,
            scanner_date=scanner_date,
            medical_validation_date=medical_validation_date,
            preferred_timerange=PreferredTimerange(rng.integers(low=0, high=5)),
        )
        generated_patients.append(patient)
    return generated_patients