import os
import pandas as pd
import numpy as np

class SolarSystem:
    def __init__(self, names):
        """
        Initialize the solar system with the masses of the sun and planets.
        names: A list of strings with names of the planets to be considered.
        """

        filename = '\Data\planet_data.csv'
        cwd      = os.getcwd()
        fn       = cwd + filename
        nanDict  = {}
        Data     = pd.read_csv(fn, header=0, skiprows=0, index_col=False, na_values=nanDict)

        Planet   = Data["Planet"].values  #names of planets
        Mass     = Data["Mass"].values    #[kg], masses
        Dist     = Data["Distance to the Sun"].values  #[AU], distance from Sun

        N = len(names) #nr of planets
        our_system = np.zeros([N, 2])  #row: planets, columns: name, mass, distance

        index = [] #indexes of planets
        for i in range(len(names)):
            for j in range(len(Planet)):
                if names[i] == Planet[j]:
                    index.append(j)
                    our_system[i,:] = [eval(Mass[j]), float(Dist[j])]
        self.our_system = our_system
        print("Our Solar System now has the following planets:")
        for k in range(len(index)):
            print("%s" % Planet[index[k]])

    def __call__(self):
        """
        Returns masses and initial conditions of the sun and planets.
        """
        return self.our_system




if __name__ == '__main__':
    ex3b = SolarSystem(["Earth", "Jupiter", "Mercury", "Saturn"])
    print(ex3b())
