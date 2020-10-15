import os, sys 

import pandas as pd
import numpy  as np

class SolarSystem:
    def __init__(self, names):
        """
        Initialize the solar system with the masses of the sun and planets.
        names: A list of strings with names of the planets to be considered.
        """

        # Setting index_col=0 to easier work with the DataFrame
        filename = '/Data/planet_data.csv'
        cwd      = os.getcwd()
        fn       = cwd + filename
        nanDict  = {}
        Data     = pd.read_csv(fn, header=0, skiprows=0, index_col=0, na_values=nanDict)

        # Creating a new DataFrame only containing input planets: 'names'
        Planets = Data.loc[names].reset_index()

        print('\nOur Solar System now has the following planets:\n')
        print(Planets)

        # Create lists of Mass of Dist values
        Mass = Planets.eval(Planets['Mass']).tolist()
        Dist = Planets['Distance to the Sun'].values.astype(float).tolist()

        our_system      = np.swapaxes(np.array([Mass, Dist]), 0, 1)
        self.our_system = our_system

    def __call__(self):
        """
        Returns masses and initial conditions of the sun and planets.
        """
        return self.our_system


if __name__ == '__main__':
    ex3b = SolarSystem(["Earth", "Jupiter", "Mercury", "Saturn"])
    print('\nrow: planets, columns: mass, distance:\n'); print(ex3b())
