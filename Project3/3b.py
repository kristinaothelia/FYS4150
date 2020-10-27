"""
FYS4150 - Project 3: b)
"""
import time, sys
import numpy                as np
import matplotlib.pyplot    as plt

# Import python programs
import functions            as func

# -----------------------------------------------------------------------------

def get_acceleration(GM, t, pos):
    """Returns the calculated acceleration"""

    r_vec = np.array([0, 0]) - pos[t, :]
    r     = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
    acc   = GM*r_vec / r**3

    return acc # numpy.ndarray


def ForwardEuler(G, ts, pos, vel, dt):
    """Forwrd Euler method


    Parameters
    ----------
    G   : float  [AU^3/yr^2]
        GravitationalConstant*M_sun, Astro units
    ts  : int 
        time steps
    pos : numpy.ndarray 
        positions       (10000, 2)
    vel : numpy.ndarray
        velocites       (10000, 2)
    dt  : float


    Returns
    -------
    position : array  
           the positions of the planet   (x,y)
    velocity : array  
            the velocities of the planet (vx,vy)

    """

    start_time = time.time()

    for t in range(ts-1):
        pos[t+1, :] = pos[t, :] + vel[t, :]*dt
        vel[t+1, :] = vel[t, :] + get_acceleration(G, t, pos)*dt

    print('Forward Euler time: ', time.time()-start_time)
    return pos, vel


def Verlet(G, ts, pos, vel, acc, dt):
    """ The Verlet method

    Parameters
    ----------
    G   : float  [AU^3/yr^2]
        GravitationalConstant*M_sun, Astro units
    ts  : int 
        time steps
    pos : numpy.ndarray
        positions
    vel : numpy.ndarray
        velocites 
    acc : array
        acceleration 
    dt  : float


    Returns
    -------
    position : array   
           the positions of the planet   (x,y)
    velocity : array 
            the velocities of the planet (vx,vy)

    """

    sys.exit()
    start_time = time.time()

    for t in range(ts-1):
        pos[t+1, :] = pos[t, :] + vel[t, :]*dt + 0.5*acc[t, :]*dt**2
        acc[t+1, :] = get_acceleration(G, t+1, pos)
        vel[t+1, :] = vel[t, :] + 0.5*(acc[t, :] + acc[t+1, :])*dt

    print('Verlet time:        ', time.time()-start_time)
    return pos, vel


def Plot_Sun_Earth_system(pos, total_time, label=None, save_fig=False):
    """Plotting the Solar-Earth"""

    if isinstance(label, str) == True:
        print('\nThe label can not be str')
        label = [label]; print(label)

    for i in range(len(label)):
        plt.plot(pos[:, 0], pos[:, 1],       label=label)
        plt.plot(pos[0, 0], pos[0, 1],  'x', label='Init. pos.')

    
    plt.axis('equal')

    plt.title("Earth-Sun system. Over %g years" %total_time, fontsize=15)
    plt.plot(0,0,'yo', label='The Sun')         # Plotte radius kanskje?

    plt.xlabel('x [AU]', fontsize=15)
    plt.ylabel('y [AU]', fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    if save_fig and label==['Verlet','Forward Euler']:
        plt.savefig("Results/3b_Earth_Sun_system.png")
    

def Energy(M_E, GM, vel, pos, time):
    K = 0.5*M_E*np.linalg.norm(vel, axis=0)**2
    U = -(GM*M_E)/np.linalg.norm(pos, axis=0)

    K = np.ravel(K)
    U = np.ravel(U)
    time = time[:-1]

    plt.figure(1)
    plt.plot(time, U, label="potential")
    plt.plot(time, K, label="kinetic")
    plt.plot(time, U+K, label="total energy")
    plt.title("Energy", fontsize=15)
    plt.xlabel("Time [yr]", fontsize=15); plt.ylabel("Energy [J] ??", fontsize=15)
    plt.legend()
    plt.show()


GM          = 4*np.pi**2                # G*M_sun, Astro units, [AU^3/yr^2]
total_time  = 10                        # [yr]
dt          = 1e-3
ts          = int(total_time/dt)        # Time step
v0          = 2*np.pi                   # Equal to np.sqrt(GM/r), where r=1 AU

pos         = np.zeros((ts, 2))
vel         = np.zeros((ts, 2))
acc         = np.zeros((ts, 2))

pos[0, :]   = [1, 0]
vel[0, :]   = [0, v0]
acc[0, :]   = get_acceleration(GM, 0, pos)


if __name__ == '__main__':

    ''' Example usage of functions '''


    pos_E, vel_E = ForwardEuler(GM, ts, pos, vel, dt)
    pos_V, vel_V = Verlet(GM, ts, pos, vel, acc, dt)


    # plotting the Sun-Earth system for Eular and Verlet
    Plot_Sun_Earth_system(pos_E, total_time=10,\
                                 label='Forward Euler',\
                                 save_fig=False);show()
    
    Plot_Sun_Earth_system(pos_V, total_time=10,\
                                 label='Verlet',\
                                 save_fig=False);show()


    # plotting the Sun-Earth system with both methods 
    Plot_Sun_Earth_system([pos_V,pos_E], total_time=10,\
                                         label=['Verlet','Forward Euler'],\
                                         save_fig=False)

    
    energy_calc = False

    if energy_calc: 
        M_earth = 5.972e24  #[kg]
        Energy(M_earth, GM, vel_V, pos_V, )
        savefig("Results/3b_Earth_Sun_system_energy.png")
    
