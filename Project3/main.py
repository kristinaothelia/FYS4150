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

planet_names = ["Earth", "Jupiter", "Saturn", "Pluto"]
planets = SolarSystem(planet_names)

yr      = 365*24*60*60          # [s]
M_Sun   = 1.989*10**30          # [kg]
M_E     = planets.mass[0]       # [kg]
M_J     = planets.mass[1]       # [kg]

AU      = 149597870691          # AU [m]

# GMJ blir feil..? GM*(M_J/M_Sun)..?
GMJ     = 4*np.pi*(M_J/M_Sun)   # G*M_J, Astro units, [AU^3/yr^2]
GM      = 4*np.pi**2            # G*M_sun, Astro units, [AU^3/yr^2]


def Energy(vel, pos, time):
    K = 0.5*M_E*np.linalg.norm(vel, axis=0)**2
    U = -(GM*M_E)/np.linalg.norm(pos, axis=0)

    K = np.ravel(K)
    U = np.ravel(U)
    time = time[:-1]

    plt.figure(1)
    plt.plot(time, U, label="Potential")
    plt.plot(time, K, label="Kinetic")
    plt.plot(time, U+K, label="Total energy")
    plt.title("Energy", fontsize=15)
    plt.xlabel("Time [yr]", fontsize=15); plt.ylabel("Energy [J] ??", fontsize=15)
    plt.legend()

def angular_momentum(vel, pos, time):

    L = np.cross(pos, vel, axis=0)
    L = np.linalg.norm(L, axis=1)
    time = time[:-1]
    plt.plot(time, L)

def Figure(title=''):
    plt.title(title, fontsize=15)
    plt.plot(0,0,'yo', label='The Sun') # Plotte radius til solen kanskje..?
    plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.axis('equal')



if __name__ == '__main__':

    '''
    parser = argparse.ArgumentParser(description="Solar system")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--c3', action="store_true", help="Project 3, c)")
    group.add_argument('-2', '--d3', action="store_true", help="Project 3, d)")
    group.add_argument('-3', '--e3', action="store_true", help="Project 3, e)")
    group.add_argument('-4', '--f3', action="store_true", help="Project 3, f)")
    group.add_argument('-5', '--g3', action="store_true", help="Project 3, g)")
    group.add_argument('-6', '--h3', action="store_true", help="Project 3, h)")
    group.add_argument('-7', '--i3', action="store_true", help="Project 3, i)")

    # Optional argument for habitable zone calculations
    #parser.add_argument('-X', '--hab', action='store_true', help="Habitable zone calculations", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()
    ex_3c = args.c3
    ex_3d = args.d3
    ex_3e = args.e3
    ex_3f = args.f3
    ex_3g = args.g3
    ex_3h = args.h3
    ex_3i = args.i3
    '''

    ex_3c = True
    ex_3d = False
    ex_3e = False
    ex_3f = False
    ex_3g = False
    ex_3h = False
    ex_3i = False

    ex_test = False


    if ex_3c == True:
        print("Earth-Sun system in 2D. Object oriented")

        T  = 10                                 # [yr]
        Np = 1                                  # Nr. of planets. Only Earth
        n  = int(10e3)

        init_pos = np.array([[1,0]])            # [AU]
        init_vel = np.array([[0,2*np.pi]])      # [AU/yr]

        init_pos = np.transpose(init_pos)
        init_vel = np.transpose(init_vel)

        # Using the class
        solver1 = Solver(M_E, init_pos, init_vel, Np, T, n)
        solver2 = Solver(M_E, init_pos, init_vel, Np, T, n)
        pos_E, vel_E, t_E = solver1.solve(method = "Euler")
        pos_V, vel_V, t_V = solver2.solve(method = "Verlet")

        # Plot for Euler and Verlet
        plt.plot(pos_E[0,:,0], pos_E[1,:,0], label="Forward Euler")
        plt.plot(pos_E[0,0,0], pos_E[1,0,0], "x", label="Init. pos.")
        plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet")
        plt.plot(pos_V[0,0,0], pos_V[1,0,0], "x", label="Init. pos.")

        # Make figure
        Figure( title="Earth-Sun system. Over %g years \n Object oriented" %T)
        plt.savefig("Results/3b_Earth_Sun_system_object_.png"); plt.show()

        # Ogsaa gjore for Euler...!

        Energy(vel_V, pos_V, t_V)
        plt.savefig("Results/3c_Earth_Sun_system_energy_object.png"); plt.show()
        angular_momentum(vel_V, pos_V, t_V) # OBS! NOE GALT
        plt.savefig("Results/3c_Earth_Sun_system_momentum_object.png"); plt.show()

    elif ex_3f == True:
        """
        Escape velocity.
        Consider then a planet (Earth) which begins at a distance of 1 AU
        from the sun. Find out by trial and error what the initial velocity
        must be in order for the planet to escape from the sun
        """

        T  = 10                        # [yr]
        Np = 1                                  # Nr. of planets. Only Earth
        n  = int(10e3)

        # Trial and error to find the initial velocity for escaping
        test = np.array([0.9, 1.1, 1.3, 1.35, 1.4, 1.415])
        # Known escape velocity equation
        eq   = np.sqrt(2*GM/1)
        # All velocities to be tested
        vel  = np.append(2*np.pi*test, eq)

        for i in range(len(vel)):

            v_esc       = vel[i]

            init_pos = np.array([[1,0]])            # [AU]
            init_vel = np.array([[0,v_esc]])        # [AU/yr]

            init_pos = np.transpose(init_pos)
            init_vel = np.transpose(init_vel)

            solver1 = Solver(M_E, init_pos, init_vel, Np, T, n)
            pos_V, vel_V, t_V = solver1.solve(method = "Verlet")

            if i < 6:
                plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet. v=2pi*%.3f" %test[i])
                plt.plot(pos_V[0,0,0], pos_V[1,0,0], "kx", label="Init. pos.")
            else:
                plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet. v=sqrt(2GM/r)")
                plt.plot(pos_V[0,0,0], pos_V[1,0,0], "kx", label="Init. pos.")


        # Make figure
        Figure(title="Earth-Sun system. Over %g years \n Escape velocity" %T)
        plt.savefig("Results/3f_v_esc_Earth_Sun_system.png"); plt.show()

    elif ex_test == True:
        """
        Test for 3h). But need to add center of mass
        """
        T  = 100                        # [yr]
        Np = len(planets.mass)          # Nr. of planets
        n  = int(10e3)

        init_pos = planets.initPos
        init_vel = planets.initVel
        masses   = planets.mass

        # Using the class
        solver2 = Solver(masses, init_pos, init_vel, Np, T, n)
        pos_V, vel_V, t_V = solver2.solve(method = "Verlet")

        # Plot orbits
        for i in range(Np):

            plt.plot(pos_V[0,:,i], pos_V[1,:,i], label="%s" %planet_names[i])
            plt.plot(pos_V[0,0,i], pos_V[1,0,i], "kx", label="Init. pos. %s" %planet_names[i])

        # Make figure
        Figure(title="Solar system. Over %g years \n Object oriented, Verlet method" %T)
        plt.savefig("Results/test_orbits.png"); plt.show()
