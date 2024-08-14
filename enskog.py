import pyenskog
import math
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

kb = 1.380649e-23  # Boltzmann constant in J/K
avogadro = 6.022e23  # Avogadro's constant in particles/mol

# lambda_zero
lambda_0 = 1.02513*(75/64)*(1/np.pi)**0.5


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
diameter_helium = 208e-12
molar_mass_helium = 4e-3
mass_helium = molar_mass_helium / avogadro

diameter_hydrogen = 228e-12
molar_mass_hydrogen = 2e-3
mass_hydrogen = molar_mass_hydrogen / avogadro

mole_fractions_hydrogen = np.linspace(0.000001, 0.999999, 15)
temperatures = np.arange(250, 1300, 100)
luu_thermal_conductivities = []
ss_thermal_conductivities = []


# -------------------------
def sutherland_thermal_conductivity(k0: float, k_t0: float, k_t_suth: float, temp: float) -> float:
    k = k0 * ((k_t0 + k_t_suth) / (temp + k_t_suth)) * (temp / k_t0)**1.5
    return k


# Sutherland' constants for hydrogen
hy_k0 = 0.168
hy_k_t0 = 273
hy_k_t_suth = 72

# Sutherland's constants for helium
he_k0 = 0.1513
he_k_t0 = 273
he_k_t_suth = 79.4
# -------------------------


def get_kinetic_diameter(k: float, m: float, temp: float):
    sigma = (kb*(lambda_0/k)*(kb*temp/m)**0.5)**0.5
    return sigma


for T in temperatures:
    luu_temp = []
    ss_temp = []
    k_hydrogen = sutherland_thermal_conductivity(hy_k0, hy_k_t0, hy_k_t_suth, T)
    k_helium = sutherland_thermal_conductivity(he_k0, he_k_t0, he_k_t_suth, T)
    diameter_helium = get_kinetic_diameter(k_helium, mass_helium, T)
    diameter_hydrogen = get_kinetic_diameter(k_hydrogen, mass_hydrogen, T)
    for x in mole_fractions_hydrogen:
        e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)
        e.addSpecies(diameter_hydrogen / diameter_helium,
                     mass_hydrogen / mass_helium, x)
        e.addSpecies(1, 1, 1 - x)
        e.init()

        luu_thermal_conductivity = e.Luu()
        luu_temp.append(convert_to_real_units(
            luu_thermal_conductivity, mass_helium, diameter_helium, T))

        ss_thermal_conductivity = (
            e.Luu() - e.Lau(0)**2 / e.Lab(0, 0))
        ss_temp.append(convert_to_real_units(
            ss_thermal_conductivity, mass_helium, diameter_helium, T))
    luu_thermal_conductivities.append(luu_temp)
    ss_thermal_conductivities.append(ss_temp)

luu_thermal_conductivities = np.array(luu_thermal_conductivities)
ss_thermal_conductivities = np.array(ss_thermal_conductivities)
X, Y = np.meshgrid(mole_fractions_hydrogen, temperatures)

X_flat = X.ravel()
Y_flat = Y.ravel()
luu_flat = luu_thermal_conductivities.ravel()
ss_flat = ss_thermal_conductivities.ravel()

plt.style.use(["science", "ieee", "no-latex"])

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

fontsize = 24

# General format without units
fmt = '%1.3f'
unit_contour_levels_luu = {0.350: '$L_{uu}=$0.350 W/m$\cdot$K'}
unit_contour_levels_ss = {0.300: '$\lambda$=0.300 W/m$\cdot$K'}

# Contour plot for Phenomenological Thermal Conductivity (Luu)
contour1 = ax[0].contourf(
    X, Y, luu_thermal_conductivities, cmap='coolwarm', levels=15)
ax[0].set_xlabel(
    'Mole Fraction of Hydrogen ($x_{H_2}$)', fontsize=fontsize, fontweight='bold')
ax[0].set_ylabel('Temperature (K)', fontsize=fontsize, fontweight='bold')
contour1_lines = ax[0].contour(
    X, Y, luu_thermal_conductivities, levels=contour1.levels, colors='black', linewidths=0.1)
ax[0].clabel(contour1_lines, inline=True, fontsize=fontsize,
             # Apply specific fmt only to certain levels
             fmt=lambda val: unit_contour_levels_luu.get(val, fmt % val))
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[0].set_xlim(0, 1)  # Set x-axis limits from 0 to 1

# Contour plot for Steady State Thermal Conductivity (λ)
contour2 = ax[1].contourf(
    X, Y, ss_thermal_conductivities, cmap='coolwarm', levels=15)
ax[1].set_xlabel(
    'Mole Fraction of Hydrogen ($x_{H_2}$)', fontsize=fontsize, fontweight='bold')
ax[1].set_ylabel('Temperature (K)', fontsize=fontsize, fontweight='bold')
contour2_lines = ax[1].contour(
    X, Y, ss_thermal_conductivities, levels=contour2.levels, colors='black', linewidths=0.1)
ax[1].clabel(contour2_lines, inline=True, fontsize=fontsize,
             # Apply specific fmt only to certain levels
             fmt=lambda val: unit_contour_levels_ss.get(val, fmt % val))
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1].set_xlim(0, 1)  # Set x-axis limits from 0 to 1

# Create a secondary x-axis on top
ax2 = ax[0].twiny()
ax2.set_xticks(ax[0].get_xticks())
ax2.set_xbound(ax[0].get_xbound())
ax2.set_xticklabels(
    [f'{1-x:.1f}' for x in ax[0].get_xticks()], fontsize=fontsize)
ax2.set_xlabel(
    'Mole Fraction of Helium ($x_{He}$)', fontsize=fontsize, fontweight='bold')

ax3 = ax[1].twiny()
ax3.set_xticks(ax[1].get_xticks())
ax3.set_xbound(ax[1].get_xbound())
ax3.set_xticklabels(
    [f'{1-x:.1f}' for x in ax[1].get_xticks()], fontsize=fontsize)
ax3.set_xlabel(
    'Mole Fraction of Helium ($x_{He}$)', fontsize=fontsize, fontweight='bold')

plt.tight_layout()
plt.savefig("./figures/thermal_conductivity_contour_plots.pdf", dpi=300)
