"""
FYS4150 - Project 3: b)
"""
import time
import numpy                as np
import matplotlib.pyplot    as plt

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
    # Trenger kanskje ikke return..?
    return pos, vel


def Plot_Sun_Earth_system(pos, label=''):

    plt.plot(pos[:, 0], pos[:, 1], label=label)
    plt.plot(pos[0, 0], pos[0, 1], 'x', label='Init. pos.')
    plt.axis("equal")


GM          = 4*np.pi**2                # G*M_sun, Astro units, [AU^3/yr^2]
total_time  = 10                        # [yr]
dt          = 1e-3
ts          = int(total_time/dt)        # Time step

pos         = np.zeros((ts, 2))
vel         = np.zeros((ts, 2))
acc         = np.zeros((ts, 2))

pos[0, :]   = [1, 0]
vel[0, :]   = [0, 2*np.pi]
acc[0, :]   = get_acceleration(GM, 0, pos)

### OBS! Noe blir overkjort. Kan ikke flytte plottet etter verlet...
pos_E, vel_E = ForwardEuler(GM, ts, pos, vel, dt)
Plot_Sun_Earth_system(pos_E, label="Forward Euler")

pos_V, vel_V = Verlet(GM, ts, pos, vel, acc, dt)
Plot_Sun_Earth_system(pos_V, label="Verlet")


plt.title("Earth-Sun system. Over %g years" %total_time, fontsize=15)
plt.plot(0,0,'yo', label='The Sun') # Plotte radius til solen kanskje..?

plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("Results/3b_Earth_Sun_system.png"); plt.show()

M_earth = 5.972e24  #[kg]
K = 0.5*M_earth*np.linalg.norm(vel_V, axis=1)**2
U = -(GM*M_earth)/np.linalg.norm(pos_V, axis=1)

plt.figure()
plt.plot(U, label="potential")
plt.plot(K, label="kinetic")
plt.plot(U+K, label="energy")

plt.title("Energy", fontsize=15)
plt.xlabel("...", fontsize=15); plt.ylabel("...", fontsize=15)
plt.legend()
plt.savefig("Results/3c_Earth_Sun_system_energy.png"); plt.show()
