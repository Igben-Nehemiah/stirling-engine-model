import numpy as np
import matplotlib.pyplot as plt
import pyenskog
import scienceplots

# Experimental data from Shaskov and Kamchatov (1972)
exp_mole_fractions = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1])
exp_thermal_conductivities = np.array([0.20202, 0.20099, 0.19968, 0.20049, 0.19927, 0.20126, 0.21048, 0.23785])

# Parameters for the gas mixtures
kb = 1.3806e-23  # Boltzmann constant
avogadro = 6.022e23  # Avogadro's constant in particles/mol

# Set up the solver
kT = 1.0  # This is an "uninteresting" variable for hard spheres, so is set to 1
verbose = False  # Don't output lots of data
use_tables = False
SonineOrder = 10
# Choose the equation of state used. Can be BOLTZMANN (ideal gas), BMCSL (recommended), VS, or HEYES
EOS = pyenskog.EOStype.BMCSL

# Define density
density = 0.00001  # density_close_pack / 3 # Volume of three times close packed

# Parameters for species
# Hydrogen
d_h2 = 228e-12
molar_mass_hydrogen = 2e-3  # Molar mass in kg/mol
m_h2 = molar_mass_hydrogen / avogadro  # in kg
k_h2 = 0.23785

# Helium
d_he = 208e-12 
molar_mass_helium = 4e-3  # Molar mass in kg/mol
m_he = molar_mass_helium / avogadro  # in kg
k_he = 0.20202

# Temperature at which thermal conductivity is measured
temperature = 423.15  # Kelvin

def series_parallel_bound_prediction(x_h2, is_max):
    x_he = 1 - x_h2
    if is_max:
        return x_h2 * k_h2 + x_he * k_he
   
    return 1 / (x_he / k_he + x_h2 / k_h2)

def enskog_theory_prediction(x_h2, temp, is_max):
    e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)
    e.addSpecies(d_he/d_h2, m_he/m_h2, 1-x_h2+1e-6)
    e.addSpecies(1, 1, x_h2+1e-6)
    e.init()
 
    if is_max:
        k_mix_red = e.Luu()
    else:
        k_mix_red = e.Luu() - e.Lau(0)**2 / e.Lab(0, 0)
    k_mix = k_mix_red * (kb * temp / (m_h2 * d_h2**2))**0.5 * kb / d_h2
    return k_mix

# Calculate predictions
prediction_range = np.linspace(0, 1, 30)
series_predictions = [series_parallel_bound_prediction(x, True) for x in prediction_range]
parallel_predictions = [series_parallel_bound_prediction(x, False) for x in prediction_range]
enskog_predictions = [enskog_theory_prediction(x, temperature, False) for x in prediction_range]
enskog_superconduction = [enskog_theory_prediction(x, temperature, True) for x in prediction_range]

font_size = 20
# Plotting the results
plt.style.use(["science", "ieee", "no-latex"])
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the data on the primary x-axis
ax1.plot(exp_mole_fractions, exp_thermal_conductivities, '^', color='black', label='Experimental data', markersize=8)
ax1.plot(prediction_range, series_predictions, '--', color='#ff7f0e', linewidth=2, label='Series-Parallel Bound')
ax1.plot(prediction_range, parallel_predictions, '--', color='#ff7f0e', linewidth=2)
ax1.plot(prediction_range, enskog_predictions, '-', color='#1f77b4', label='Enskog Steady-State')
ax1.plot(prediction_range, enskog_superconduction, '-', color='#d62728', label='Enskog Transient')

# Set labels for the primary x-axis
ax1.set_xlabel('Mole Fraction of Hydrogen ($x_{H_{2}}$)', fontsize=font_size)
ax1.set_ylabel('Thermal Conductivity (W/mK)', fontsize=font_size)
ax1.grid(True)
ax1.set_xlim(0, 1)

# Create a secondary x-axis on top
ax2 = ax1.twiny()

# Set ticks and labels for the secondary x-axis
ax2.set_xticks(ax1.get_xticks())
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([f'{1-x:.1f}' for x in ax1.get_xticks()])
ax2.set_xlabel('Mole Fraction of Helium ($x_{He}$)', fontsize=font_size)

# Adjust tick label sizes
ax1.tick_params(axis='both', which='major', labelsize=font_size)
ax2.tick_params(axis='x', which='major', labelsize=font_size)

# Add legend
ax1.legend(fontsize=font_size)

# Final adjustments
fig.tight_layout()
plt.savefig("./figures/thermal_conductivity_predictions_validation_with_helium.pdf", dpi=300, bbox_inches='tight')
