from math import pi
import numpy as np
import pyenskog


kb = 1.380649e-23  # Boltzmann constant in J/K
avogadro = 6.022e23  # Avogadro's constant in particles/mol

def get_kinetic_diameter(temp: float, molar_mass: float, k: float):
    m = molar_mass/avogadro
    # sigma = (((1.02513*(75*kb/64)*(kb*temp/(m*pi))**0.5)/k)*((kb*temp/m)**0.5)*kb)**(1/4)
    # return sigma

    sigma = (1.02513*75*temp*kb**3/(64*k*m*pi))**0.25
    return sigma



print(get_kinetic_diameter(423.23, 4e-3, 0.20202))

# molar_mass = 0.02018
# m = molar_mass/avogadro
# temp = 600
# v = (1.02513*(75*kb/64)*(kb*temp/(m*pi))**0.5)/(289e-12)**2

# print(v)