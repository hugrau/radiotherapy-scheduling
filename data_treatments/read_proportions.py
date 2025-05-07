import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_treatments.parsing_utils import apply_lower, apply_unidecode, apply_replace, insert_space_before_numbers


def read_proportions_file(filename: str):
    """
    Reads a csv file containing all proportions for locations.
    :param filename:
    :return:
    """
    df = pd.read_csv(filename)
    df = df.map(apply_lower)
    df = df.map(apply_unidecode)
    df = df.map(apply_replace)
    df['location'] = df['location'].apply(insert_space_before_numbers)
    return df


def simulate_discrete_law(proportions_frequencies: pd.DataFrame):
    """
    Simulate a discrete law of probability where the probabilities are given by the frequency of locations.
    F^-1(U) = X
    Returns a random event (a location).
    :param proportions_frequencies: pandas data frame of the different frequencies for different locations.
    :return: string that represents a random event.
    """
    random_variable = np.random.rand(1)
    # print(df.shape[0])
    s = np.array(proportions_frequencies['proportion'].cumsum())
    t = s <= random_variable
    i = np.argmax(t == False)
    return proportions_frequencies.loc[i].location


def test_simulation(n: int, df: pd.DataFrame):
    """
    Function to compare the real frequencies to the simulations of random variable made by simulate_discrete_law.
    :param n: number of simulations to make
    :param df: the data frame with proportions/frequencies for each location
    :return: None
    """
    y1 = np.array(df['proportion'])
    # print(y1)
    locations = list(df['location'])
    random_variable = [simulate_discrete_law(df) for _ in range(n)]
    # print(random_variable)
    y2 = np.zeros(len(locations))
    i = 0
    # TODO = this part could be improved somehow to be faster.
    for loc in locations:
        k = random_variable.count(loc)
        y2[i] = k/n
        i += 1
    # print(y2)
    lx = np.linspace(0, 137, 137)
    plt.plot(lx, y1, 'rx', label='Real frequencies')
    plt.plot(lx, y2, 'bx', label='Simulation')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    filepath = '../data/proportion_data.csv'
    dft = read_proportions_file(filepath)
    location = simulate_discrete_law(dft)
    # print(f"{location=}")
    test_simulation(2000, dft)
    # print(df.loc[])
    # print(df['proportion'].sum())
