import pyenskog
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

kb = 1.380649e-23  # Boltzmann constant in J/K
avogadro = 6.022e23  # Avogadro's constant in particles/mol


def convert_to_real_units(k_red, m1, d1, T):
    k = k_red * (kb * T / (m1 * d1**2))**0.5 * kb / d1
    return k

# Higher-order polynomial surface function for fitting


def poly2d(XY, *coeffs):
    x, y = XY
    terms = []
    order = 4  # Increased polynomial order
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append(coeffs[idx] * x**i * y**j)
            idx += 1
    return np.sum(terms, axis=0)


# Set up the solver
kT = 1.0  # This is an "uninteresting" variable for hard spheres, so is set to 1
verbose = False  # Don't output lots of data

# Now we choose the order of the solution. Higher orders mean slower, but more accurate.
# Generalised N-order solution is available and should be used instead.
use_tables = False
SonineOrder = 20  # Reduced the order of the SonineOrder to 10
# Choose the equation of state used. Can be BOLTZMANN (ideal gas), BMCSL (recommended), VS, or HEYES
EOS = pyenskog.EOStype.BMCSL

# Define density
density_close_pack = 2.0 / (math.sqrt(2))
density = 0.00001  

# Initialize Enskog solver
e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)

# Parameters for species
# Helium
diameter1 = 208e-12
molar_mass_helium = 4e-3  # Molar mass in kg/mol
mass1 = molar_mass_helium / avogadro  # in kg

# Hydrogen
diameter2 = 228e-12
molar_mass_hydrogen = 2e-3  # Molar mass in kg/mol
mass2 = molar_mass_hydrogen / avogadro  # in kg

# Arrays to store results
mole_fractions = np.linspace(0.0001, 0.9999, 10)
# Temperature range of heat exchangers in the GPU-3 Stirling engine
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

# Convert lists to numpy arrays for easier handling
luu_thermal_conductivities = np.array(luu_thermal_conductivities)
ss_thermal_conductivities = np.array(ss_thermal_conductivities)
X, Y = np.meshgrid(mole_fractions, temperatures)

# Flatten the data for curve fitting
X_flat = X.ravel()
Y_flat = Y.ravel()
luu_flat = luu_thermal_conductivities.ravel()
ss_flat = ss_thermal_conductivities.ravel()

# Fit polynomial surface to the data
order = 4  # Increase polynomial order for better fitting
initial_guess = np.ones((order + 1) * (order + 2) // 2)
luu_params, _ = curve_fit(poly2d, (X_flat, Y_flat), luu_flat, p0=initial_guess)
ss_params, _ = curve_fit(poly2d, (X_flat, Y_flat), ss_flat, p0=initial_guess)


def predict_luu_thermal_conductivity(x, T):
    return poly2d((x, T), *luu_params)


def predict_ss_thermal_conductivity(x, T):
    return poly2d((x, T), *ss_params)

# Print the coefficients
print("Coefficients for Phenomenological Thermal Conductivity (Luu):", luu_params)
print("Coefficients for steady-state thermal conductivity (λ):", ss_params)

# Create contour plots for better visualization
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

fontsize = 17
# Contour plot for Phenomenological Thermal Conductivity (Luu)
contour1 = ax[0].contourf(
    X, Y, luu_thermal_conductivities, cmap='coolwarm', levels=20)
ax[0].set_xlabel('$x_{H_2}$', fontsize=fontsize)  # Increased font size
ax[0].set_ylabel('Temperature (K)', fontsize=fontsize)  # Increased font size
contour1_lines = ax[0].contour(
    X, Y, luu_thermal_conductivities, levels=contour1.levels, colors='black', linewidths=0.5)
ax[0].clabel(contour1_lines, inline=True, fontsize=fontsize)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
# Contour plot for Steady State Thermal Conductivity (λ)
contour2 = ax[1].contourf(
    X, Y, ss_thermal_conductivities, cmap='coolwarm', levels=20)
ax[1].set_xlabel('$x_{H_2}$', fontsize=fontsize)
ax[1].set_ylabel('Temperature (K)', fontsize=fontsize)
contour2_lines = ax[1].contour(
    X, Y, ss_thermal_conductivities, levels=contour2.levels, colors='black', linewidths=0.5)
ax[1].clabel(contour2_lines, inline=True, fontsize=fontsize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

# Save the plot
plt.tight_layout()
plt.savefig("thermal_conductivity_contour_plots.png", dpi=300)
plt.show()

print("Predicted Luu Thermal Conductivity at x=0.5 and T=423.25 K:",
      predict_luu_thermal_conductivity(0, 423.25))
print("Predicted Steady State Thermal Conductivity at x=0.5 and T=423.25 K:",
      predict_ss_thermal_conductivity(0, 423.25))
