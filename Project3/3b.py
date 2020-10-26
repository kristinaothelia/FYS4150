"""
FYS4150 - Project 3: b)
"""
import time
import numpy                as np
import matplotlib.pyplot    as plt

# Import python programs
import functions            as func

# -----------------------------------------------------------------------------

def get_acceleration(GM, t, pos):
    """
    Returns the calculated acceleration
    """
    r_vec = np.array([0, 0]) - pos[t, :]
    r     = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
    acc   = GM*r_vec / r**3

    return acc


def ForwardEuler(G, ts, pos, vel, dt):
    """
    Forwrd Euler method. Returns position and velocity
    """
    start_time = time.time()

    for t in range(ts-1):
        pos[t+1, :] = pos[t, :] + vel[t, :]*dt
        vel[t+1, :] = vel[t, :] + get_acceleration(G, t, pos)*dt

    print("Forward Euler time: ", time.time()-start_time)
    # Trenger kanskje ikke return..?
    return pos, vel


def Verlet(G, ts, pos, vel, acc, dt):
    """
    Verlet method. Returns position and velocity
    """
    start_time = time.time()

    for t in range(ts-1):
        pos[t+1, :] = pos[t, :] + vel[t, :]*dt + 0.5*acc[t, :]*dt**2
        acc[t+1, :] = get_acceleration(G, t+1, pos)
        vel[t+1, :] = vel[t, :] + 0.5*(acc[t, :] + acc[t+1, :])*dt

    print("Verlet time: ", time.time()-start_time)
    return pos, vel


def Plot_Sun_Earth_system(pos, total_time, label='', save_fig=False):
    """Plotting the SolarSystwm with Earth

    """

    #if isinstance(name, str) == False:
    #        raise TypeError('name must be str')

    print(label)
    print(type(label))
    #if isinstance(label, str):


    if len(label) >=  2:

        posV = pos[0]
        posE = pos[1]

        plt.plot(posV[:, 0], posV[:, 1],      label=labels[0])
        plt.plot(posV[0, 0], posV[0, 1], 'x', label='Init. pos.')
        plt.plot(posE[:, :], posE[:, :],      label=labels[1])
        plt.plot(posE[0, 0], posE[0, 1], 'x', label='Init. pos.')

    elif len(label) >= 1:
        plt.plot(pos[:, 0], pos[:, 1],        label=label)
        plt.plot(pos[0, 0], pos[0, 1],   'x', label='Init. pos.')
    else:
        print('gg'); sys.exit()

    
    plt.axis('equal')

    plt.title("Earth-Sun system. Over %g years" %total_time, fontsize=15)
    plt.plot(0,0,'yo', label='The Sun')         # Plotte radius kanskje?

    plt.xlabel('x [AU]', fontsize=15)
    plt.ylabel('y [AU]', fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    if save_fig:
        plt.savefig("Results/3b_Earth_Sun_system.png")
    


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


    #Plot_Sun_Earth_system(pos_E, total_time=10, label='Forward Euler', save_fig=False)
    #Plot_Sun_Earth_system(pos_V, total_time=10, label='Verlet', save_fig=False)

    Plot_Sun_Earth_system([pos_V,pos_E], total_time=10, label=['Verlet','Forward Euler'], save_fig=False)


    '''
    plt.title("Earth-Sun system. Over %g years" %total_time, fontsize=15)
    plt.plot(0,0,'yo', label='The Sun') # Plotte radius til solen kanskje..?

    plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("Results/3b_Earth_Sun_system.png"); plt.show()
    '''
    

    '''
    M_earth = 5.972e24  #[kg]
    func.Energy(M_earth, GM, vel_V, pos_V, )
    plt.savefig("Results/3c_Earth_Sun_system_energy.png"); plt.show()
    '''
