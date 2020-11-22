#from __future__ import division

import math, sys, time, timeit

import matplotlib.pyplot as plt
import numpy             as np

from numba import vectorize, jit, int32, float64, intp, double

#https://stackoverflow.com/questions/57285547/numba-jit-warnings-interpretation-in-python
#https://numba.pydata.org/numba-doc/latest/user/5minguide.html

@jit(int32(intp,intp,intp), nopython=True) # forceobj=True
def periodic(i, limit, add):
    """
    Choose correct matrix index with periodic
    boundary conditions
    Input:
    - i     : int
        Base index
    - limit : int
        Highest \"legal\" index
    - add   : int
        Number to add or subtract from i
    Returns np.int32
    """
    #p = (i+limit+add) % limit
    #print(type(i), type(limit), type(add), p, p.size, type(p))
    #sys.exit()
    return (i+limit+add) % limit

#@jit(double(float64,intp,intp))      # forceobj=True  parallel=True
@jit
def monteCarlo(temp, NSpins, MCcycles):
    """
    Calculate the energy and magnetization
    (\"straight\" and squared) for a given temperature
    Input:
    - temp     : np.float64
        Temperature to calculate for
    - NSpins   : int
        dimension of square matrix
    - MCcycles : int
        Monte-carlo MCcycles (how many times do we flip the matrix?)

    Output:
    - E_av       : np.float64
        Energy of matrix averaged over MCcycles, normalized to spins**2
    - E_variance : np.float64
        Variance of energy, same normalization * temp**2
    - M_av       : np.float64
        Magnetic field of matrix, averaged over MCcycles, normalized to spins**2
    - M_variance : np.float64
        Variance of magnetic field, same normalization * temp
    - Mabs       : np.float64
        Absolute value of magnetic field, averaged over MCcycles
    """

    #Setup spin matrix, initialize to ground state
    spin_matrix = np.zeros((NSpins,NSpins), np.int8) + 1

    #Create and initialize variables
    E    = M     = 0 # np.int32(0)
    E_av = E2_av = M_av = M2_av = Mabs_av = 0 # np.int32(0)

    #Setup array for possible energy changes
    w = np.zeros(17,np.float64)
    for de in range(-8,9,4): #include +8
        w[de+8] = np.exp(-de/temp)

    #Calculate initial magnetization:
    M = spin_matrix.sum()
    #Calculate initial energy
    for j in range(NSpins):
        for i in range(NSpins):
            E -= spin_matrix[i][j]*\
                 (spin_matrix[periodic(i,NSpins,-1)][j] + spin_matrix[i][periodic(j,NSpins,1)])

    #Start metropolis MonteCarlo computation
    for i in range(MCcycles):
        #Metropolis
        #Loop over all spins, pick a random spin each time
        for s in range(NSpins**2):
            x = np.int32(np.random.random()*NSpins)
            y = np.int32(np.random.random()*NSpins)
            deltaE = 2*spin_matrix[x][y]*\
                     (spin_matrix[periodic(x,NSpins,-1)][y] +\
                      spin_matrix[periodic(x,NSpins,1)][y] +\
                      spin_matrix[x][periodic(y,NSpins,-1)] +\
                      spin_matrix[x][periodic(y,NSpins,1)])
            if np.random.random() <= w[deltaE+8]:
                #Accept!
                spin_matrix[x,y] *= -1
                M += 2*spin_matrix[x,y]
                E += deltaE


        #Update expectation values
        E_av    += E
        E2_av   += E**2
        M_av    += M
        M2_av   += M**2
        Mabs_av += np.int32(math.fabs(M))

    #Normalize average values
    E_av       /= np.float64(MCcycles);
    E2_av      /= np.float64(MCcycles);
    M_av       /= np.float64(MCcycles);
    M2_av      /= np.float64(MCcycles);
    Mabs_av    /= np.float64(MCcycles);

    #Calculate variance and normalize to per-point and temp
    E_variance  = (E2_av-E_av*E_av)/np.float64(NSpins*NSpins*temp*temp);
    M_variance  = (M2_av-M_av*M_av)/np.float64(NSpins*NSpins*temp);

    #Normalize returned averages to per-point
    E_av       /= np.float64(NSpins*NSpins);
    M_av       /= np.float64(NSpins*NSpins);
    Mabs_av    /= np.float64(NSpins*NSpins);

    #print(temp, type(temp), NSpins, type(NSpins), MCcycles, type(MCcycles))
    #print(E_av, type(E_av), E_variance, type(E_variance), M_av, type(M_av))
    #print(M_variance, type(M_variance), Mabs_av, type(Mabs_av))
    #sys.exit()
    return E_av, E_variance, M_av, M_variance, Mabs_av


def model(InitialT, FinalT, NumberTsteps, NSpins, MCcycles):

    Tsteps = (FinalT-InitialT)/NumberTsteps
    Temp   = np.zeros(NumberTsteps, np.float64)

    # Declare arrays that hold averages
    Energy           = np.zeros(NumberTsteps, dtype=np.float64)
    Magnetization    = np.zeros(NumberTsteps, dtype=np.float64)
    SpecificHeat     = np.zeros(NumberTsteps, dtype=np.float64)
    Susceptibility   = np.zeros(NumberTsteps, dtype=np.float64)
    MagnetizationAbs = np.zeros(NumberTsteps, dtype=np.float64)

    for T in range(NumberTsteps):
        Temp[T] = InitialT + T*Tsteps
        Energy[T], SpecificHeat[T], Magnetization[T], Susceptibility[T], MagnetizationAbs[T] = monteCarlo(Temp[T],NSpins,MCcycles)
        
        #print(Energy[T], SpecificHeat[T], Magnetization[T], Susceptibility[T], MagnetizationAbs[T])
        #sys.exit()

    return Energy, SpecificHeat, Magnetization, Susceptibility, MagnetizationAbs, Temp

if __name__ == "__main__":

    # temperature steps, initial temperature, final temperature
    NumberTsteps = 1 #20
    InitialT     = 1 #2.4 #1 #1.5
    FinalT       = 1 #2.4 #1 #2.5

    # Define number of spins. Er dette L? Som skal vere 2 i starten..?
    NSpins = 2

    # Define number of Monte Carlo cycles
    MCcycles = 10000000 #10000


    # this is probably bad way to calc time (should use timeit)
    start = time.time()

    Energy, SpecificHeat, Magnetization, Susceptibility, MagnetizationAbs, Temp = model(InitialT, FinalT, NumberTsteps, NSpins, MCcycles)

    end      = time.time()
    run_time = end - start

    print('\nExecution time (with compilation) : %f s\n' %run_time)

    # better way to measure execution time ('without' compilation time)
    #execution_time = timeit.timeit(lambda:model(InitialT, FinalT, NumberTsteps, NSpins, MCcycles), number=3)
    #print(execution_time/3)

    print('Energy:', np.sum(Energy)/len(Energy))                    
    print('SpecificHeat:', np.sum(SpecificHeat)/len(SpecificHeat))         
    print('Magnetization:', np.sum(Magnetization)/len(Magnetization))  
    print('Susceptibility:', np.sum(Susceptibility)/len(Susceptibility))
    print('MagnetizationAbs:', np.sum(MagnetizationAbs)/len(MagnetizationAbs))

    '''
    T=1, NSpins=2, MCcycles=10000000:
    ---------------------------------
    Execution time (with compilation) : 2.870548 s

    Energy: -1.9960468
    SpecificHeat: 0.031563088839039466
    Magnetization: 0.01723985
    Susceptibility: 3.99222105028791
    MagnetizationAbs: 0.99868155

    T=1, NSpins=20, MCcycles=10000000:
    ----------------------------------
    Execution time (with compilation) : 194.809163 s

    Energy: -1.997162759
    SpecificHeat: 0.023367653403256555
    Magnetization: 0.9992764169999999
    Susceptibility: 0.0015686570569232571
    MagnetizationAbs: 0.9992764169999999

    T=2.4, NSpins=20, MCcycles=10000000:
    ------------------------------------
    Execution time (with compilation) : 229.515274 s (after ctrl+c)

    Energy: -1.236161381
    SpecificHeat: 1.4060732704286667
    Magnetization: 0.0029550325
    Susceptibility: 42.92205587090398
    MagnetizationAbs: 0.4521837215
    '''

    #monteCarlo.inspect_types()
    sys.exit()

    # And finally plot
    f = plt.figure(figsize=(15,8)) # 18,10

    f.suptitle('The Main Title Of The Subfigures', fontsize=15)

    sp =  f.add_subplot(2, 2, 1)
    plt.plot(Temp, Energy, 'o', color="green")
    #plt.title('Energy') # better?
    plt.ylabel("Energy ", fontsize=15)
    plt.xticks(visible=False);plt.yticks(fontsize=13)

    sp =  f.add_subplot(2, 2, 2)
    plt.plot(Temp, abs(Magnetization), 'o', color="red")
    plt.ylabel("Magnetization ", fontsize=15)
    plt.xticks(visible=False);plt.yticks(fontsize=11)

    sp =  f.add_subplot(2, 2, 3)
    plt.plot(Temp, SpecificHeat, 'o', color="blue")
    plt.xlabel("Temperature (T)", fontsize=15)
    plt.ylabel("Specific Heat ", fontsize=15)
    plt.xticks(fontsize=11);plt.yticks(fontsize=11)

    sp =  f.add_subplot(2, 2, 4)
    plt.plot(Temp, Susceptibility, 'o', color="black")
    plt.xlabel("Temperature (T)", fontsize=15)
    plt.ylabel("Susceptibility", fontsize=15)
    plt.xticks(fontsize=11);plt.yticks(fontsize=11)

    f.tight_layout()
    f.subplots_adjust(top=0.89)

    plt.show()


    def test_2x2(Initial=1, FinalT=1, NumberTsteps=NumberTsteps, NSpins=2, MCcycles=MCcycles):

        # med MCcycles = 10000000 får jeg basically samme
        # som two_x_two.py og analytisk 

        Energy, SpecificHeat, Magnetization, Susceptibility, MagnetizationAbs, Temp = model(InitialT, FinalT, NumberTsteps, NSpins, MCcycles)
        print(Energy)
        print(SpecificHeat)
        print(Susceptibility)

        # hmmmm stemmer ikke som two_x_two.py
        print('Mean energy:             %f' % np.mean(Energy))
        print('Specific Heat:           %f' % np.mean(SpecificHeat))
        print('Mean Magenetization:     %f' % np.mean(Magnetization))
        print('Susceptibility:          %f' % np.mean(Susceptibility))
        print('Mean abs. Magnetization: %f' % np.mean(MagnetizationAbs))

    test_2x2(Initial=1, FinalT=1, NumberTsteps=NumberTsteps, NSpins=2, MCcycles=MCcycles)
