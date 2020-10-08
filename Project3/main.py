
"""
Main program for FYS4150 - Project 3: Solar system
"""
import sys, os
import argparse
import numpy                as np
import matplotlib.pyplot    as plt

# Import python programs
import functions            as func
#------------------------------------------------------------------------------

Data = func.GetData(filename='\planet_data.csv')
#print(Data)

Planet   = Data["Planet"].values
# Mass blir feil... Maa endres i .csv
Mass     = Data["Mass"].values
Dist     = Data["Distance to the Sun"].values

#print(Data.loc[:, Data.columns == 'Mass'].values)
"""
print(Planet)
print(Mass)
print(Dist)
"""

yr      = 365*24*60*60
M_Sun   = 1.989*10**30          # [kg]
#M_E     = Mass[0]               # Sting ikke tall... Maa endre noe i .csv
M_E     = 6.0*10**24
print(M_E)

AU      = 149597870691          # AU [m]
#G       = 6.67430*10**(-11)     # [m^3/kgs^2]
GM      = 4*np.pi**2            # G*M_sun, Astro units, [AU^3/yr^2]


# Bare begynte aa sette opp noe til senere..

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Solar system")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--a3', action="store_true", help="Project 3, a)")
    group.add_argument('-2', '--b3', action="store_true", help="Project 3, b)")

    # Optional argument for habitable zone calculations
    #parser.add_argument('-X', '--hab', action='store_true', help="Habitable zone calculations", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()
    ex_3a = args.a3
    ex_3b = args.b3

    if ex_3a == True:

        print("Earth-Sun system in 2D. Not object oriented")
        print("--"*55)

        total_time  = 10                         # [yr]
        dt          = 1e-3
        ts          = int(total_time/dt)        # Time step

        pos         = np.zeros((ts, 2))
        vel         = np.zeros((ts, 2))
        acc         = np.zeros((ts, 2))

        pos[0, :]   = [1, 0]
        vel[0, :]   = [0, 2*np.pi]
        acc[0, :]   = func.get_acceleration(GM, 0, pos)

        ### OBS! Noe blir overkjort. Kan ikke flytte plottet etter verlet...
        pos_E, vel_E = func.ForwardEuler(GM, ts, pos, vel, dt)
        func.Plot_Sun_Earth_system(pos_E, label="ForwardEuler")

        pos_V, vel_V = func.Verlet(GM, ts, pos, vel, acc, dt)
        func.Plot_Sun_Earth_system(pos_V, label="Verlet")

        # Burde ogsaa plotte solen i midten...

        plt.title("Earth-Sun system. Over %g years" %total_time)
        plt.savefig("Results/3a_Earth_Sun_system.png")
        plt.show()


    elif ex_3b == True:

        print("Earth-Sun system. Object oriented")
        print("--"*55)
