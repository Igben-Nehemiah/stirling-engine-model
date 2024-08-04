import numpy as np
import pyenskog
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots

# Sutherland's constants
k0=0.1513
k_t0=273
k_t_suth=79.4

# Enskog 
x2=1
x1 = 1 - x2

kb = 1.380649e-23  # Boltzmann constant in J/K
avogadro = 6.022e23  # Avogadro's constant in particles/mol

# Helium
diameter2 = 208e-12
molar_mass_helium = 4e-3  # Molar mass in kg/mol
mass2 = molar_mass_helium / avogadro  # in kg

# Hydrogen
diameter1 = 228e-12
molar_mass_hydrogen = 2e-3  # Molar mass in kg/mol
mass1 = molar_mass_hydrogen / avogadro  # in kg

kT = 1.0  # This is an "uninteresting" variable for hard spheres, so is set to 1
verbose = False  # Don't output lots of data
use_tables = False
SonineOrder = 10  # Reduced the order of the SonineOrder to 10
EOS = pyenskog.EOStype.BMCSL  # Choose the equation of state used

density = 0.00001  # Not actual density but taken in limit as Density -> 0
e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)

e.addSpecies(diameter2 / diameter1, mass2 / mass1, x2+1e-8)
e.addSpecies(1, 1, x1+1e-8)
e.init()

k_mix_red = e.Luu()  # if params.is_max else e.Luu() - e.Lau(0)**2 / e.Lab(0, 0)

def thermal_conductivity_sutherland(temp: float) -> float:
    k = k0 * ((k_t0 + k_t_suth) / (temp + k_t_suth) * (temp / k_t0)**1.5)
    return k


def thermal_conductivity_enskog(temp: float) -> float:
    k_mix = k_mix_red * (kb * temp / (mass1 * diameter1**2))**0.5 * kb / diameter1
    return k_mix
   



temps = np.arange(250, 1000, 50)

sutherland_thermal_conductivities = [thermal_conductivity_sutherland(temp) for temp in temps]
enskog_thermal_conductivities = [thermal_conductivity_enskog(temp) for temp in temps]

print(sutherland_thermal_conductivities)
print(enskog_thermal_conductivities)

plt.plot(temps, sutherland_thermal_conductivities)
plt.plot(temps, enskog_thermal_conductivities)

plt.savefig("enskog_vs_sutherland.png")




