"""
Analythical and numerical solution of the 2x2 Ising model
"""
import numpy             as np
import ising_model       as ising

# -----------------------------------------------------------------------------

def numerical_solution(spin_matrix, n_cycles, temp, L):
    """
    This function calculates the numerical solution of the 2x2 Ising model
    lattice, using the MC function from ising_model.py
    """

    # Compute quantities
    quantities       = ising.MC(spin_matrix, n_cycles, temp)

    norm             = 1.0/float(n_cycles)
    E_avg            = np.sum(quantities[:,0])*norm
    M_avg            = np.sum(quantities[:,1])*norm
    E2_avg           = np.sum(quantities[:,2])*norm
    M2_avg           = np.sum(quantities[:,3])*norm
    M_abs_avg        = np.sum(quantities[:,4])*norm

    E_var            = (E2_avg - E_avg**2)/(L**2)
    M_var            = (M2_avg - M_avg**2)/(L**2)

    Energy           = E_avg    /(L**2)
    Magnetization    = M_avg    /(L**2)
    MagnetizationAbs = M_abs_avg/(L**2)
    SpecificHeat     = E_var    /(temp**2)  # * L**2?
    Susceptibility   = M_var    /(temp)     # * L**2?

    return Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility

def twoXtwo(L, temp, max_cycles):

    # Array of Monte Carlo runs we want to evaluate, logarithmic spacing
    runs        = np.logspace(2, int(np.log10(max_cycles)), (int(np.log10(max_cycles))-1), endpoint=True)
    spin_matrix = np.ones((L, L), np.int8)

    # Looping over the different number of Monte Carlo cycles
    for i, n_cycles in enumerate(runs):
        Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility = \
        numerical_solution(spin_matrix, n_cycles, temp, L)

        # Print for each magnitude of Monte Carlo cycles
        print('\nMonte Carlo cycles:      %s' % n_cycles)
        print('Mean energy:             %f' % Energy)
        print('Specific Heat:           %f' % SpecificHeat)
        print('Mean Magenetization:     %f' % Magnetization)
        print('Susceptibility:          %f' % Susceptibility)
        print('Mean abs. Magnetization: %f' % MagnetizationAbs)


def Analythical_2x2(J, L, temp):

    ang  = 8*J/temp
    Z    = 12 + 2*np.exp(-ang) + 2*np.exp(ang)

    E_avg       = 16*J*    (np.exp(-ang) - np.exp(ang)) / Z
    E2_avg      = 128*J**2*(np.exp(-ang) + np.exp(ang)) / Z
    E_var       = E2_avg - E_avg**2

    M_avg       = 0
    M2_avg      = 32*(1 + np.exp(ang)) / Z
    M_abs_avg   = 8*(2 + np.exp(ang))  / Z
    M_var       = M2_avg - M_avg**2

    A_Energy            = E_avg / L**2
    A_SpecificHeat      = E_var / (temp**2 * L**2)  # C_V
    A_Magnetization     = M_avg / L**2
    A_MagnetizationAbs  = M_abs_avg / L**2
    A_Susceptibility    = M_var / (temp * L**2)     # Chi, (32/(4*Z))*(1+np.exp(ang))

    print("\nAnalytical solutions:")
    print("Mean energy:             %f" %A_Energy)
    print("Specific heat:           %f" %A_SpecificHeat)
    print("Mean Magenetization:     %f" %A_Magnetization)
    print("Susceptibility:          %f" %A_Susceptibility)
    print("Mean abs. Magnetization: %f" %A_MagnetizationAbs)

""" Sample run

max_cycles = 1e7
L          = 2
temp       = 1
J          = 1

twoXtwo(L, temp, max_cycles)
Analythical_2x2(J, L, temp)
"""
