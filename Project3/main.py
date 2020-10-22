"""
Main program for FYS4150 - Project 3: Solar system
"""
import sys, os
import argparse
import numpy                as np
import matplotlib.pyplot    as plt

# Import python programs
import functions            as func

from Solver                 import Solver
from SolarSystem            import SolarSystem
#------------------------------------------------------------------------------

planets = SolarSystem(["Earth", "Jupiter", "Saturn"])

M_E = planets.mass[0]
print('Mass Earth:', M_E)

M_J = planets.mass[1]*1000
print('Mass Jupiter:', M_J)

yr      = 365*24*60*60          # [s]
M_Sun   = 1.989*10**30          # [kg]

AU      = 149597870691          # AU [m]
GMJ     = 4*np.pi*(M_J/M_Sun)   # [AU^3/yr^2]
GM      = 4*np.pi**2            # G*M_sun, Astro units, [AU^3/yr^2]
G = GM/M_Sun


# Bare begynte aa sette opp noe til senere..

if __name__ == '__main__':

    '''
    parser = argparse.ArgumentParser(description="Solar system")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--b3', action="store_true", help="Project 3, b)")
    group.add_argument('-2', '--c3', action="store_true", help="Project 3, c)")

    # Optional argument for habitable zone calculations
    #parser.add_argument('-X', '--hab', action='store_true', help="Habitable zone calculations", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()
    ex_3a = args.b3
    ex_3b = args.c3
    '''

    ex_3c = True
    ex_3b = False

    ex_test = False

    if ex_3b == True:

        print("Earth-Sun system in 2D. Not object oriented")
        print("--"*55)

        # Kan denne bare importeres og kjores?
        # 3b.py

    elif ex_3c == True:
        print("Earth-Sun system in 2D. Object oriented (we think...)")

        # Denne maa inn i Solver...?
        def a(r, t):
            """
            Right-hand-side of the ODE dv/dt = -GM/r^2.
            GM = 4*pi^2, gravitational constant in solar system units (?)
            r = radial distance from the Sun to a planet.
            kan kanskje vaere i func?
            """
            unit_r = r/np.linalg.norm(r, axis=0)
            acc    = -GM/np.linalg.norm(r, axis=0)**2*unit_r
            return acc


        T  = 10  #[yr]
        n  = int(10e3)
        Np = 1  #nr of planets
        M   = planets.mass[0]


        init_pos = np.array([[1,0]])            # [AU]
        init_vel = np.array([[0,2*np.pi]])      # [AU/yr]

        init_pos = np.transpose(init_pos)
        init_vel = np.transpose(init_vel)

        #using the class
        solver1 = Solver(a, M, init_pos, init_vel, Np, T, n)
        solver2 = Solver(a, M, init_pos, init_vel, Np, T, n)
        pos_E, vel_E, t_E = solver1.solve(method = "Euler")
        pos_V, vel_V, t_V = solver2.solve(method = "Verlet")

        plt.plot(pos_E[0,:,0], pos_E[1,:,0], label="Forward Euler")
        plt.plot(pos_E[0,0,0], pos_E[1,0,0], "x", label="Init. pos.")
        plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet")
        plt.plot(pos_V[0,0,0], pos_V[1,0,0], "x", label="Init. pos.")

        plt.title("Earth-Sun system. Over %g years \n Object oriented" %T, fontsize=15)
        plt.plot(0,0,'yo', label='The Sun') # Plotte radius til solen kanskje..?
        plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.axis('equal')
        plt.savefig("Results/3b_Earth_Sun_system_object.png"); plt.show()

        # Funker ikke helt...

        func.Energy(M, GM, vel_V, pos_V, t_V)
        plt.savefig("Results/3c_Earth_Sun_system_energy_object.png"); plt.show()

    elif ex_test == True:

        def a(r, t):
            """
            Right-hand-side of the ODE dv/dt = -GM/r^2.
            GM = 4*pi^2, gravitational constant in solar system units (?)
            r = radial distance from the Sun to a planet.
            kan kanskje vaere i func?
            """
            #unit_r = r/np.linalg.norm(r, axis=0)
            #acc    = -GM/np.linalg.norm(r, axis=0)**2*unit_r
            unit_r = r/np.linalg.norm(r, axis=0)
            acc    = -GM/np.linalg.norm(r, axis=0)**2*unit_r
            return acc


        T  = 100  #[yr]
        n  = int(10e3)
        Np = len(planets.mass)  #nr of planets

        #init_pos = np.array([[1,0]])            # [AU]
        #init_vel = np.array([[0,2*np.pi]])      # [AU/yr]

        init_pos = planets.initPos
        init_vel = planets.initVel
        masses   = planets.mass

        #using the class
        solver2 = Solver(a, masses, init_pos, init_vel, Np, T, n)
        pos_V, vel_V, t_V = solver2.solve(method = "Verlet")

        plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet")
        plt.plot(pos_V[0,0,0], pos_V[1,0,0], "x", label="Init. pos.")

        plt.plot(pos_V[0,:,1], pos_V[1,:,1], label="Verlet")
        plt.plot(pos_V[0,0,1], pos_V[1,0,1], "x", label="Init. pos.")

        plt.plot(pos_V[0,:,2], pos_V[1,:,2], label="Verlet")
        plt.plot(pos_V[0,0,2], pos_V[1,0,2], "x", label="Init. pos.")

        plt.title("Earth-Sun system. Over %g years \n Object oriented" %T, fontsize=15)
        plt.plot(0,0,'yo', label='The Sun') # Plotte radius til solen kanskje..?
        plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.axis('equal')
        plt.show()
