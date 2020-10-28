"""
Main program for FYS4150 - Project 3: Solar system
"""
import sys, os, time
import argparse
import numpy                as np
import matplotlib.pyplot    as plt

# Import python classes
from Solver                 import Solver
from SolarSystem            import SolarSystem
#------------------------------------------------------------------------------

def Energy(vel, pos, time, title=''):

    K    = 0.5*np.linalg.norm(vel, axis=0)**2
    U    = -(GM)/np.linalg.norm(pos, axis=0)
    K    = np.ravel(K)
    U    = np.ravel(U)
    time = time[:-1]

    plt.plot(time, U, label="Potential")
    plt.plot(time, K, label="Kinetic")
    plt.plot(time, U+K, label="Total energy")
    plt.xticks(fontsize=13); plt.yticks(fontsize=13)
    plt.title(title, fontsize=15)
    plt.xlabel('Time [yr]', fontsize=15)
    plt.ylabel('Energy/mass [AU^3/yr^2]', fontsize=15)
    plt.legend(fontsize=13)
    return K, U

def angular_momentum(vel, pos, time, title=''):

    L = np.cross(pos, vel, axis=0)
    L = np.linalg.norm(L, axis=1)
    time = time[:-1]

    plt.plot(time, L) # :100
    plt.yticks(fontsize=13); plt.xticks(fontsize=13)
    plt.axis([np.min(time),np.max(time), 0,10])
    plt.title(title, fontsize=15)
    plt.xlabel('Time [yr]', fontsize=15)
    plt.ylabel('L/mass [AU^2/yr]', fontsize=15)
    return L

def Figure(title=''):

    plt.title(title, fontsize=15)
    plt.plot(0,0,'yo', label='The Sun')
    plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.xticks(fontsize=13); plt.yticks(fontsize=13)
    plt.axis('equal'); plt.tight_layout()

def Figure_noSunPlot(title=''):

    plt.title(title, fontsize=15)
    plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.xticks(fontsize=13); plt.yticks(fontsize=13)
    plt.axis('equal'); plt.tight_layout()


def find_last_min(distances):
    """looping backwards to find the last minimum"""

    index = int(len(distances)-2)
    while distances[index] < distances[index + 1]:
        #print('index', index);print(distances[index]);print(distances[index+1])
        index -= 1
    index_minimum = index+1
    return index_minimum

def Ex3cd(n, T=10, Np=1, test_stability=False, save_plot=False):
    """
    n       : Integration points
    T       : Time to run the simulation. Default 10 years. [yr]
    Np      : Number of planets. Earth only for 3c)

    Test stability : Test n-values needed for a stable orbit
    """

    init_pos = np.array([[1,0]])            # [AU]
    init_vel = np.array([[0,2*np.pi]])      # [AU/yr]

    init_pos = np.transpose(init_pos)
    init_vel = np.transpose(init_vel)

    if test_stability == True:

        n = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]   # Different n values

        for i in range(len(n)):

            # Using the class
            solver1 = Solver(M_E, init_pos, init_vel, Np, T, int(n[i]))
            solver2 = Solver(M_E, init_pos, init_vel, Np, T, int(n[i]))

            start_time = time.time()
            pos_E, vel_E, t_E = solver1.solve(method = "Euler")
            print("Forward Euler time: ", time.time()-start_time)
            start_time_V = time.time()
            pos_V, vel_V, t_V = solver2.solve(method = "Verlet")
            print("Vel. Verlet time:   ", time.time()-start_time_V)

            # Plot for Euler and Verlet
            plt.plot(pos_E[0,:,0], pos_E[1,:,0], label="Forward Euler")
            plt.plot(pos_E[0,0,0], pos_E[1,0,0], "kx", label="Init. pos.")
            plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet")
            plt.plot(pos_V[0,0,0], pos_V[1,0,0], "kx", label="Init. pos.")

            Figure(title="Earth-Sun system. Over %g years \n Object oriented. n=$10^%g$" %(T, i+1))
            if save_plot==True:
                plt.savefig("Results/Integration_points/3b_stability_%g.png" %(i+1))
            plt.show()
        sys.exit()

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
    Figure(title="Earth-Sun system. Over %g years \n Object oriented" %T)
    if save_plot==True:
        plt.savefig("Results/3b_Earth_Sun_system_object_.png")
    plt.show()

    # Energy and momentum, Forward Euler:
    Energy(vel_E, pos_E, t_E, "Earth-Sun system. Energy conservation \n Forward Euler")
    if save_plot==True:
        plt.savefig("Results/3c_Earth_Sun_system_energy_object_E.png")
    plt.show()
    angular_momentum(vel_E, pos_E, t_E, "Earth-Sun system. Angular momentum per mass \n Forward Euler")
    if save_plot==True:
        plt.savefig("Results/3c_Earth_Sun_system_momentum_object_E.png")
    plt.show()

    # Energy and momentum, Verlet:
    Energy(vel_V, pos_V, t_V, "Earth-Sun system. Energy conservation \n Verlet")
    if save_plot==True:
         plt.savefig("Results/3c_Earth_Sun_system_energy_object_V.png")
    plt.show()
    angular_momentum(vel_V, pos_V, t_V, "Earth-Sun system. Angular momentum per mass \n Verlet")
    if save_plot==True:
        plt.savefig("Results/3c_Earth_Sun_system_momentum_object_V.png")
    plt.show()


def Ex3e(n, T=10, Np=1, beta=2, v0=2*np.pi, save_plot=False):
    """
    n       : Integration points
    T       : Time to run the simulation. Default 10 years. [yr]
    Np      : Number of planets. Earth only for 3e)
    Beta    : Beta values. beta=2 as default

    v0
    """

    if type(beta) == int:

        init_pos = np.array([[1,0]])                # [AU]
        init_vel = np.array([[0,v0]])               # [AU/yr]

        init_pos = np.transpose(init_pos)
        init_vel = np.transpose(init_vel)

        solver1 = Solver(M_E, init_pos, init_vel, Np, T, n)
        pos_V, vel_V, t_V = solver1.solve(method = "Verlet", beta=beta)

        plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet. beta=%.1f" % beta)
        plt.plot(pos_V[0,0,0], pos_V[1,0,0], "kx", label="Init. pos.")

        # Make figure
        Figure(title="Earth-Sun system. Over %g years \n v0=%g AU/yr" %(T, v0))
        if save_plot==True:
            plt.savefig("Results/3e_beta_Earth_Sun_system_v0%g.png" %v0)
        plt.show()

        # Energy and momentum, Verlet:
        Energy(vel_V, pos_V, t_V, "Earth-Sun system, v0=%g AU/yr \n Energy conservation with  Verlet" %v0)
        plt.show()
        angular_momentum(vel_V, pos_V, t_V, "Earth-Sun system, v0=%g AU/yr \n Angular momentum with Verlet" %v0) # OBS! NOE GALT
        plt.show()


    else:
        for i in range(len(beta)):

            init_pos = np.array([[1,0]])                # [AU]
            init_vel = np.array([[0,v0]])               # [AU/yr]

            init_pos = np.transpose(init_pos)
            init_vel = np.transpose(init_vel)

            solver1 = Solver(M_E, init_pos, init_vel, Np, T, n)
            pos_V, vel_V, t_V = solver1.solve(method = "Verlet", beta=beta[i])

            plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet. beta=%.1f" % beta[i])
            plt.plot(pos_V[0,0,0], pos_V[1,0,0], "kx", label="Init. pos.")

        # Make figure
        Figure(title="Earth-Sun system. Over %g years \n Different $\\beta$ values. v0=2$\\pi$" %T)
        if save_plot==True:
            plt.savefig("Results/3e_beta_Earth_Sun_system.png")
        plt.show()


def Ex3f(n, T=10, Np=1, v_esc_test=[0.9, 1.1, 1.3, 1.35, 1.4, 1.415], save_plot=False):
    """
    n       : Integration points
    T       : Time to run the simulation. Default 10 years. [yr]
    Np      : Number of planets. Earth only for 3f)

    v_esc_test : List of initial velocities to test as escape velocity
                 Default: [0.9, 1.1, 1.3, 1.35, 1.4, 1.415]
    """

    # Trial and error to find the initial velocity for escaping
    test = np.array(v_esc_test)
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
            plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet. v0=2pi*%.3f" %test[i])
            plt.plot(pos_V[0,0,0], pos_V[1,0,0], "kx", label="Init. pos.")
        else:
            plt.plot(pos_V[0,:,0], pos_V[1,:,0], label="Verlet. v0=sqrt(2GM/r)")
            plt.plot(pos_V[0,0,0], pos_V[1,0,0], "kx", label="Init. pos.")

    # Make figure
    Figure(title="Earth-Sun system. Over %g years \n Escape velocity" %T)
    if save_plot==True:
        plt.savefig("Results/3f_v_esc_Earth_Sun_system_yr%g.png" %T)
    plt.show()


def Ex3g(n, T, m=1, save_plot=False):
    """
    n       : Integration points
    T       : Time to run the simulation.
    m       : Factor to change Jupiter mass
    """

    planet_names = ["Earth", "Jupiter"]
    planets = SolarSystem(planet_names)

    Np = len(planets.mass)          # Nr. of planets

    init_pos = planets.initPos
    init_vel = planets.initVel
    masses   = planets.mass

    for M in range(len(m)):

        masses_  = [masses[0], masses[1]*m[M]]

        # Using the class
        solver2 = Solver(masses_, init_pos, init_vel, Np, T, n)
        pos_V, vel_V, t_V = solver2.solve(method = "Verlet")

        # Plot orbits
        for i in range(Np):

            plt.plot(pos_V[0,:,i], pos_V[1,:,i], label="%s" %planet_names[i])
            plt.plot(pos_V[0,0,i], pos_V[1,0,i], "kx", label="Init. pos. %s" %planet_names[i])

        # Make figure
        Figure(title="Solar system. Over %g years \n Verlet method. $M_J$=$M_J$*%g" %(T,m[M]))
        if save_plot==True:
            plt.savefig("Results/3g_E_J_Sun_system_m%g.png" %m[M])
        plt.show()


def Ex3h(n, T, planet_names, save_plot=False):
    """
    n       : Integration points
    T       : Time to run the simulation.

    planet_names : Import of wanted planets (and/or Sun)
    """
    #n = int(1e5)
    planets = SolarSystem(planet_names)
    Np      = len(planets.mass)          # Nr. of planets

    init_pos = planets.initPos
    init_vel = planets.initVel
    masses   = planets.mass

    # Using the class
    solver = Solver(masses, init_pos, init_vel, Np, T, n)
    pos_V, vel_V, t_V = solver.solve(method = "Verlet", SunInMotion=True)

    # Plot orbits
    for i in range(Np):

        plt.plot(pos_V[0,:,i], pos_V[1,:,i], label="%s" %planet_names[i])
        plt.plot(pos_V[0,0,i], pos_V[1,0,i], "kx", label="Init. pos. %s" %planet_names[i])

    # Make figure
    Figure_noSunPlot(title="Solar system. Over %g years \n Verlet method" %(T))
    if save_plot==True:
        plt.savefig("Results/3h_solar_system_nrPlanets_%g.png" %Np)
    plt.show()


def Ex3i(planet_names, n=1e4, T=100, slice_n=3000, save_plot=False):
    """The orbit of Mercury with relativistic precision"""

    n      = int(n)
    last_n = int(slice_n)

    if n < slice_n:
        print('slice_n must <= than n');sys.exit()


    planets  = SolarSystem(planet_names, PrintTable=True)
    M_M      = planets.mass

    Np       = len(planets.mass)        # Nr. of planets

    masses   = planets.mass
    init_pos = np.array([[0.3075,0]])   # [AU]
    init_vel = np.array([[0,12.44]])    # [AU/yr]
    init_pos = np.transpose(init_pos)
    init_vel = np.transpose(init_vel)


    solver            = Solver(masses, init_pos, init_vel, Np, T, n)
    pos_V, vel_V, t_V = solver.solver_relativistic(beta=2)
    distances_all     = np.linalg.norm(pos_V, axis=0)


    min_dist       = np.min(distances_all)
    max_dist       = np.max(distances_all)
    distances      = distances_all[-last_n:]  # last_n distances


    index_minimum = find_last_min(distances)

    time_index    = (len(pos_V[0,:,0])-len(pos_V[0,-last_n:,0]))+index_minimum

    if distances[index_minimum] != distances_all[time_index]:
        print('%f != %f'\
            %(distances[index_minimum], distances_all[time_index]))
        sys.exit()


    angle_t0   = np.arctan2(pos_V[0,0,0],pos_V[1,0,0])                   # angle at t=0 (pi/2)
    angle_t100 = np.arctan(pos_V[0,time_index,0]/pos_V[1,time_index,0])  # angle at t=100 yrs


    angle_t0_per_year   = np.abs(angle_t0)/100
    angle_t100_per_year = np.abs(angle_t100)/100

    delta_rad = (angle_t0_per_year - angle_t100_per_year)

    arc_seconds = np.rad2deg(delta_rad)*60*60


    with open('Results/Mercury/new_T[%g]_n[%g]_.txt' %(T, n), 'w') as f:

        f.write('\nthe perihelion position: (%f, %f) \n'\
              %(pos_V[0,time_index,0],pos_V[1,time_index,0]))

        f.write('\nt = 0   yrs, theta = %f \n' %(angle_t0))
        f.write('t = 100 yrs, theta = %f \n'   %(angle_t100))

        f.write('\ntheta per year = %f \n' %(angle_t0/100))
        f.write('theta per year = %f \n'   %(angle_t100/100))

        f.write('\ndelta rad   = %f \n' %delta_rad)
        f.write('\narc seconds = %f \n' %arc_seconds)


    out_data = open('Results/Mercury/new_T[%g]_n[%g]_.txt' %(T, n)).read()
    print(out_data)


    plt.plot(pos_V[0,:,0], pos_V[1,:,0], label='Mercury')
    plt.plot(pos_V[0,0,0], pos_V[1,0,0], 'rx', label='Init. pos.')
    plt.plot(pos_V[0,-1,0], pos_V[1,-1,0], 'kx', label='Last pos.')
    plt.plot(pos_V[0,time_index,0], pos_V[1,time_index,0], 'mx', label='Last min.')


    plt.title('The orbit of Mercury for 100 years', fontsize=15)

    Figure(title='The orbit of Mercury for 100 years')
    if save_plot==True:
        plt.savefig('Results/Mercury/new_T[%g]_n[%g]_plot.png' %(T, n))

    plt.show()



planet_names_ = ['Earth', 'Jupiter']
planets_      = SolarSystem(planet_names_)


yr      = 365*24*60*60          # [s]
M_Sun   = 1.989*10**30          # [kg]
M_E     = planets_.mass[0]      # [kg]
M_J     = planets_.mass[1]      # [kg]

AU      = 149597870691          # [m]

GMJ     = 4*np.pi*(M_J/M_Sun)   # [AU^3/yr^2] (G*M_J, Astro units,)
GM      = 4*np.pi**2            # [AU^3/yr^2] ( G*M_sun, Astro units,)

n       = int(1e4)              # integration points


if __name__ == '__main__':

    # For ex. 3b) Run >>>3b.py only

    parser = argparse.ArgumentParser(description="Solar system")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--cd3', action="store_true", help="Project 3, c) og d)")
    group.add_argument('-2', '--e3',  action="store_true", help="Project 3, e)")
    group.add_argument('-3', '--f3',  action="store_true", help="Project 3, f)")
    group.add_argument('-4', '--g3',  action="store_true", help="Project 3, g)")
    group.add_argument('-5', '--h3',  action="store_true", help="Project 3, h)")
    group.add_argument('-6', '--i3',  action="store_true", help="Project 3, i)")

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args   = parser.parse_args()
    ex_3cd = args.cd3
    ex_3e  = args.e3
    ex_3f  = args.f3
    ex_3g  = args.g3
    ex_3h  = args.h3
    ex_3i  = args.i3


    if ex_3cd == True:
        print("--------------------------------------------------------------")
        print("Earth-Sun system in 2D. Object oriented. Energy cons and AM")
        print("--------------------------------------------------------------")

        #Ex3cd(n=n, T=10, Np=1, test_stability=True)
        n=int(1e4)
        Ex3cd(n=n, T=10, Np=1, test_stability=False, save_plot=True)

    elif ex_3e == True:
        print("--------------------------------------------------------------")
        print("Beta values")
        print("--------------------------------------------------------------")

        #b  = np.linspace(2, 3, 6)
        b  = [3, 2.9, 2]                        # Beta values
        Ex3e(n=n, T=50, Np=1, beta=b, save_plot=True)

        # Just for beta=2
        Ex3e(n=n, T=10, Np=1, beta=2, v0=5)


    elif ex_3f == True:
        print("--------------------------------------------------------------")
        print("Escape velocity")
        print("--------------------------------------------------------------")

        list = [0.9, 1.1, 1.3, 1.35, 1.4, 1.415]
        Ex3f(n=n, T=10,  Np=1, v_esc_test=list, save_plot=True)
        Ex3f(n=n, T=100, Np=1, v_esc_test=list, save_plot=True)


    elif ex_3g == True:
        print("--------------------------------------------------------------")
        print("The three-body problem. Earth-Jupiter-Sun")
        print("--------------------------------------------------------------")

        n = 5*int(1e5)                  # integration points

        # Factors to change the mass of Jupiter
        m = [1, 10, 1000]
        Ex3g(n=n, T=100, m=m, save_plot=True)


    elif ex_3h == True:
        '''
        Planets and the Sun have to orbit around
        the center of mass, not a stationary Sun.
        '''
        print("--------------------------------------------------------------")
        print("Model for all planets of the solar system. Sun in motion")
        print("--------------------------------------------------------------")

        n       = 5*int(1e5)            # integration points

        SEJ = ["Sun", "Earth", "Jupiter"]
        SS  = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptun', 'Pluto']

        Ex3h(n, T=100, planet_names=SEJ, save_plot=True)
        Ex3h(n, T=250, planet_names=SS, save_plot=True)


    elif ex_3i == True:
        print("--------------------------------------------------------------")
        print("The perihelion precession of Mercury")
        print("--------------------------------------------------------------")

        #n  = 5*int(1e5)                # integration points

        SM = ['Mercury']

        Ex3i(planet_names=SM, n=4.5*1e5, T=100, slice_n=3000, save_plot=True)
