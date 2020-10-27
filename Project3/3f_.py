"""
FYS4150 - Project 3: f)
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
    #plt.plot(pos[0, 0], pos[0, 1], 'x', label='Init. pos.')
    plt.axis("equal")

# -----------------------------------------------------------------------------

GM          = 4*np.pi**2                # G*M_sun, Astro units, [AU^3/yr^2]
total_time  = 10                        # [yr]
dt          = 1e-3
ts          = int(total_time/dt)        # Time step

pos         = np.zeros((ts, 2))
vel         = np.zeros((ts, 2))
acc         = np.zeros((ts, 2))

# Trial and error to find the initial velocity for escaping
test = [0.9, 1.1, 1.3, 1.35, 1.4, 1.415]

for i in range(len(test)):
    v_esc       = 2*np.pi*test[i]
    pos[0, :]   = [1, 0]
    vel[0, :]   = [0, v_esc]
    acc[0, :]   = get_acceleration(GM, 0, pos)

    pos_V, vel_V = Verlet(GM, ts, pos, vel, acc, dt)
    Plot_Sun_Earth_system(pos_V, label="Verlet. v=2pi*%.3f"%test[i])

# The formula for escape velocity
v_esc       = np.sqrt(2*GM/1)           # sqrt(2GM/r)

pos[0, :]   = [1, 0]
vel[0, :]   = [0, v_esc]
acc[0, :]   = get_acceleration(GM, 0, pos)

pos_V, vel_V = Verlet(GM, ts, pos, vel, acc, dt)
Plot_Sun_Earth_system(pos_V, label="Verlet. v=sqrt(2GM/r)")
plt.plot(pos[0, 0], pos[0, 1], 'x', label='Init. pos.')

plt.title("Earth-Sun system. Over %g years \n Escape velocity" %total_time, fontsize=15)
plt.plot(0,0,'yo', label='The Sun') # Plotte radius til solen kanskje..?

plt.xlabel("x [AU]", fontsize=15); plt.ylabel("y [AU]", fontsize=15)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("Results/3f_v_esc_Earth_Sun_system.png"); plt.show()