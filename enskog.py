import pyenskog
import math
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

kb = 1.380649e-23  # Boltzmann constant in J/K
avogadro = 6.022e23  # Avogadro's constant in particles/mol

def convert_to_real_units(k_red, m1, d1, T):
    k = k_red * (kb * T / (m1 * d1**2))**0.5 * kb / d1
    return k

# Set up the solver
kT = 1.0  # This is an "uninteresting" variable for hard spheres, so is set to 1
verbose = False  # Don't output lots of data

use_tables = False
SonineOrder = 20  
EOS = pyenskog.EOStype.BMCSL

# Define density
density_close_pack = 2.0 / (math.sqrt(2))
density = 0.00001  

# Initialize Enskog solver
e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)

# Parameters for species
diameter1 = 208e-12
molar_mass_helium = 4e-3  
mass1 = molar_mass_helium / avogadro 

diameter2 = 228e-12
molar_mass_hydrogen = 2e-3  
mass2 = molar_mass_hydrogen / avogadro 

mole_fractions = np.linspace(0.0001, 0.9999, 10)
temperatures = np.arange(250, 1000, 50)
luu_thermal_conductivities = []
ss_thermal_conductivities = []

for T in temperatures:
    luu_temp = []
    ss_temp = []
    for x in mole_fractions:
        e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)
        e.addSpecies(diameter2 / diameter1, mass2 / mass1, x)
        e.addSpecies(1, 1, 1 - x)
        e.init()

        luu_thermal_conductivity = e.Luu()
        luu_temp.append(convert_to_real_units(
            luu_thermal_conductivity, mass1, diameter1, T))

        ss_thermal_conductivity = (
            e.Luu() - e.Lau(0)**2 / e.Lab(0, 0))
        ss_temp.append(convert_to_real_units(
            ss_thermal_conductivity, mass1, diameter1, T))
    luu_thermal_conductivities.append(luu_temp)
    ss_thermal_conductivities.append(ss_temp)

luu_thermal_conductivities = np.array(luu_thermal_conductivities)
ss_thermal_conductivities = np.array(ss_thermal_conductivities)
X, Y = np.meshgrid(mole_fractions, temperatures)

X_flat = X.ravel()
Y_flat = Y.ravel()
luu_flat = luu_thermal_conductivities.ravel()
ss_flat = ss_thermal_conductivities.ravel()

plt.style.use(["science", "ieee", "no-latex"])

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

fontsize = 17

# General format without units
fmt = '%1.3f'
# Specific levels with units
unit_contour_levels_luu = {0.300: '$L_{uu}=$0.300 W/m$\cdot$K'}
unit_contour_levels_ss = {0.260: '$\lambda$=0.260 W/m$\cdot$K'}

# Contour plot for Phenomenological Thermal Conductivity (Luu)
contour1 = ax[0].contourf(
    X, Y, luu_thermal_conductivities, cmap='coolwarm', levels=20)
ax[0].set_xlabel('Mole Fraction of Hydrogen ($x_{H_2}$)', fontsize=fontsize, fontweight='bold')
ax[0].set_ylabel('Temperature (K)', fontsize=fontsize, fontweight='bold')
contour1_lines = ax[0].contour(
    X, Y, luu_thermal_conductivities, levels=contour1.levels, colors='black', linewidths=0.5)
ax[0].clabel(contour1_lines, inline=True, fontsize=fontsize,
             fmt=lambda val: unit_contour_levels_luu.get(val, fmt % val))  # Apply specific fmt only to certain levels
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[0].set_xlim(0, 1)  # Set x-axis limits from 0 to 1

# Contour plot for Steady State Thermal Conductivity (λ)
contour2 = ax[1].contourf(
    X, Y, ss_thermal_conductivities, cmap='coolwarm', levels=20)
ax[1].set_xlabel('Mole Fraction of Hydrogen ($x_{H_2}$)', fontsize=fontsize, fontweight='bold')
ax[1].set_ylabel('Temperature (K)', fontsize=fontsize, fontweight='bold')
contour2_lines = ax[1].contour(
    X, Y, ss_thermal_conductivities, levels=contour2.levels, colors='black', linewidths=0.5)
ax[1].clabel(contour2_lines, inline=True, fontsize=fontsize,
             fmt=lambda val: unit_contour_levels_ss.get(val, fmt % val))  # Apply specific fmt only to certain levels
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1].set_xlim(0, 1)  # Set x-axis limits from 0 to 1

# Create a secondary x-axis on top
ax2 = ax[0].twiny()
ax2.set_xticks(ax[0].get_xticks())
ax2.set_xbound(ax[0].get_xbound())
ax2.set_xticklabels([f'{1-x:.1f}' for x in ax[0].get_xticks()], fontsize=fontsize)
ax2.set_xlabel('Mole Fraction of Helium ($x_{He}$)', fontsize=fontsize, fontweight='bold')

ax3 = ax[1].twiny()
ax3.set_xticks(ax[1].get_xticks())
ax3.set_xbound(ax[1].get_xbound())
ax3.set_xticklabels([f'{1-x:.1f}' for x in ax[1].get_xticks()], fontsize=fontsize)
ax3.set_xlabel('Mole Fraction of Helium ($x_{He}$)', fontsize=fontsize, fontweight='bold')

plt.tight_layout()
plt.savefig("thermal_conductivity_contour_plots.png", dpi=300)
