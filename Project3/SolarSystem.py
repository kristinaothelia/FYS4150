"""
FYS4150 - Project 3: Solar system class
"""
import os, sys
import pandas as pd
import numpy  as np

#------------------------------------------------------------------------------

class SolarSystem:

    def __init__(self, names, PrintTable=False):
        """Initialize the solar system

        Reads a .csv-file with Planet, Mass, x, y, vx, vy as columns,
        where (x,y) is the start position, and (vx,vy) is the start velocity.

        Hardcoded for exact file/usage.
        The file contains the Sun as a planet, and values are 
        according to the Solar System Barycenter (mass center)

        Parameters
        ----------
        names : list
            A list of strings with names of the planets to be considered.
        PrintTable : bool
            When True, a DataFrame with planet (names) information is printed.

        
        I will remove this later (just idea for testing other functions):
        Doctest raises TypeError if __init__ returns other than None
        
        >>> SolarSystem(['Earth', 'Jupiter', 'Mercury', 'Saturn'])
        None
        """
        
        # Setting index_col=0 to easier work with the DataFrame
        filename = '/Data/planet_data.csv'
        cwd      = os.getcwd()
        fn       = cwd + filename
        nanDict  = {}
        Data     = pd.read_csv(fn, header=0, skiprows=0, index_col=0, na_values=nanDict)

        names = names
    
        # New DataFrame only containing the planets given by names:
        self.TableWithPlanets = Data.loc[names].reset_index()

        if PrintTable:
            print('\nThe Solar System contains the following planets:\n')
            print(self.TableWithPlanets)


        x0     = self.TableWithPlanets['x'].values          # [AU]
        y0     = self.TableWithPlanets['y'].values          # [AU]
        vx0    = self.TableWithPlanets['vx'].values*365.25  # [AU/day]->[AU/year]
        vy0    = self.TableWithPlanets['vy'].values*365.25  # [AU/day]->[AU/year]


        # Creating instance attributes; mass, InitPos and InitVel 
        # of type np.ndarray with dtypes float64
        self.mass    = self.TableWithPlanets.eval(\
                       self.TableWithPlanets['Mass'])\
                       .astype('float64')                    # shape (Number_of_Planets,)

        self.initPos = np.array((x0, y0), dtype='float64')   # shape (2, Number_of_Planets)
        self.initVel = np.array((vx0, vy0), dtype='float64') # shape (2, Number_of_Planets)


    '''
    def init_cond(self, x0, y0, vx0, vy0):
        """ Creating arrays for initial conditions

        Returns
        -------
        InitPos : type
            InitPos description

        Use Doctest somehow to check return types?:
        #>>> init_cond()
        type of arrays
        """

        self.initPos = np.array((x0, y0))
        self.initVel = np.array((vx0, vy0))
        return #self.initPos, self.initVel
    '''
    


if __name__ == '__main__':

    ''' Usage of the SolarSystem class '''

    planets_ex = SolarSystem(['Earth', 'Jupiter', 'Mercury', 'Saturn'], PrintTable=True)

    '''
    # Planning to do some prints to illustrate
    masses_ex  = planets_ex.mass
    initPos_ex = planets_ex.initPos
    initVel_ex = planets_ex.initVel

    print('\n Mass of planets')
    print('-- type', type(masses_ex))
    print(masses_ex)

    print('\n Initial position of planets (type: %s):\n'\
    %type(planets_ex.initPos))
    print(planets_ex.initPos)
    print(planets_ex.initPos.shape, planets_ex.initPos.size)

    print('\n Initial velocity of planets (type: %s):\n'\
    %type(planets_ex.initVel))
    print(planets_ex.initVel)
    '''


    '''
    The SolarSystem class could be rewritten 
    to be more general, for instance with an input filename,
    or by using astroquery to fetch Horizons data.
    
    Example for fetching Earth data:
    --------------------------------
    # https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html

    from astroquery.jplhorizons import Horizons

    earth_id = 399
    ref_body = '500@10'   # referred to as 'CENTER' in Horizons
                          # '500@10' = Sun as center of mass
                          # '500@0'  = real mass center..?

    start_date = 'YYYY-MM-DD'
    end_date   = 'YYYY-MM-DD'

    object   = Horizons(id=earth_id,\
                        id_type='id',\
                        epochs={'start':start_date,'stop':end_date,'step':'1d'},\
                        location=ref_body)

    data_vector = object.vectors()
    x  = data_vector.columns['x'][0]  # something like this..
    y  = data_vector.columns['y'][0]  # something like this..
    vx = data_vector.columns['vx'][0] # something like this..
    xy = data_vector.columns['xy'][0] # something like this..
    '''