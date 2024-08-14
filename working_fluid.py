from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass
import numpy as np
import pyenskog

lambda_0 = 1.02513*(75/64)*(1/np.pi)**0.5
kb = 1.380649e-23  # Boltzmann constant in J/K

def get_kinetic_diameter(k: float, m: float, temp: float):
    sigma = (kb*(lambda_0/k)*(kb*temp/m)**0.5)**0.5
    return sigma

class WorkingFluid(ABC):
    """
    Abstract base class for a working fluid.
    Defines the interface for thermal conductivity, dynamic viscosity, and specific heat capacities.
    """
    def __init__(self):
        self.specific_gas_constant = 0  # Specific gas constant
        self.molar_mass = 0

    @abstractmethod
    def thermal_conductivity(self, temp: float) -> float:
        """
        Calculate the thermal conductivity of the working fluid at a given temperature.

        Args: 
            temp: Temperature in Kelvin

        Returns: 
            Thermal conductivity in W/(m·K)
        """

    @abstractmethod
    def dynamic_viscosity(self, temp: float) -> float:
        """
        Calculate the dynamic viscosity of the working fluid at a given temperature.

        Args: 
            temp: Temperature in Kelvin

        Returns: 
            Dynamic viscosity in Pa·s
        """

    @abstractmethod
    def specific_heat_capacities(self, temp: float) -> Tuple[float, float]:
        """
        Calculate the specific heat capacities (C_v and C_p) of the working fluid at a given temperature.

        Args:
            temp: Temperature in Kelvin

        Returns:
            Tuple of (C_v, C_p) in J/(kg·K)
        """


class IdealGasWorkingFluid(WorkingFluid):
    """
    Class representing an ideal gas working fluid.
    """
    @dataclass
    class Parameters:
        mu0: float       # Reference dynamic viscosity in Pa·s
        mu_t0: float     # Reference temperature for dynamic viscosity in Kelvin
        mu_t_suth: float  # Sutherland's constant for dynamic viscosity in Kelvin
        k0: float        # Reference thermal conductivity in W/(m·K)
        k_t0: float      # Reference temperature for thermal conductivity in Kelvin
        k_t_suth: float  # Sutherland's constant for thermal conductivity in Kelvin
        a: float         # Coefficient for C_v polynomial term T^0 in J/(kg·K)
        # Coefficient for C_v polynomial term T^1 in J/(kg·K^2)
        b: float
        # Coefficient for C_v polynomial term T^2 in J/(kg·K^3)
        c: float
        # Coefficient for C_v polynomial term T^3 in J/(kg·K^4)
        d: float
        # Coefficient for C_v polynomial term T^4 in J/(kg·K^5)
        e: float
        molar_mass: float  # Molar mass in kg/mol

    def __init__(self, params: Parameters):
        self.params = params
        # Specific gas constant in J/(kg·K)
        self.specific_gas_constant = 8.314 / self.params.molar_mass
        self.molar_mass = self.params.molar_mass

    def thermal_conductivity(self, temp: float) -> float:
        k = self.params.k0 * ((self.params.k_t0 + self.params.k_t_suth) /
                              (temp + self.params.k_t_suth)) * (temp / self.params.k_t0)**1.5
        return k

    def dynamic_viscosity(self, temp: float) -> float:
        mu = self.params.mu0 * ((self.params.mu_t0 + self.params.mu_t_suth) /
                                (temp + self.params.mu_t_suth)) * (temp / self.params.mu_t0)**1.5
        return mu

    def specific_heat_capacities(self, temp: float) -> Tuple[float, float]:
        t = temp / 1000.0
        cp_mol = (self.params.a + self.params.b * t +
                  self.params.c * t**2 + self.params.d * t**3 +
                  # Shomate equation for C_p in J/(mol·K)
                  self.params.e / t**2)
        # Convert C_p from J/(mol·K) to J/(kg·K)
        cp_mass = cp_mol / self.params.molar_mass
        cv_mass = cp_mass - self.specific_gas_constant  # Calculate C_v
        return cv_mass, cp_mass


class GasMixtures(WorkingFluid):
    """
    Class representing a binary gas mixture working fluid.
    """
    @dataclass
    class Parameters:
        fluid1: WorkingFluid
        fluid2: WorkingFluid
        x1: float
        x2: float
        is_classical: bool
        is_max: bool

    def __init__(self, params: Parameters):
        self.params = params
        self.avogadro = 6.022e23  # Avogadro's constant in particles/mol
        self.molar_mass = self.params.fluid1.molar_mass * \
            self.params.x1 + self.params.fluid2.molar_mass * self.params.x2
        self.specific_gas_constant = 8.314 / self.molar_mass

        # Helium
        self.mass2 = self.params.fluid2.molar_mass / self.avogadro  # in kg

        # Hydrogen
        self.mass1 = self.params.fluid1.molar_mass / self.avogadro  # in kg

    def thermal_conductivity(self, temp: float) -> float:
        x1, x2 = 1 - self.params.x2, self.params.x2
        if not self.params.is_classical:
            kT = 1.0  # This is an "uninteresting" variable for hard spheres, so is set to 1
            verbose = False  # Don't output lots of data
            use_tables = False
            SonineOrder = 10  # Reduced the order of the SonineOrder to 10
            EOS = pyenskog.EOStype.BMCSL  # Choose the equation of state used

            density = 0.00001  # Not actual density but taken in limit as Density -> 0

            e = pyenskog.Enskog(density, EOS, SonineOrder,
                                verbose, kT, use_tables)

            k1 = self.params.fluid1.thermal_conductivity(temp)
            k2 = self.params.fluid2.thermal_conductivity(temp)

            diameter1 = get_kinetic_diameter(k1, self.mass1, temp)
            diameter2 = get_kinetic_diameter(k2, self.mass2, temp)

            e.addSpecies(diameter2 / diameter1, self.mass2 /
                         self.mass1, self.params.x2+1e-8)
            e.addSpecies(1, 1, x1+1e-8)
            e.init()

            k_mix_red = e.Luu() if self.params.is_max else e.Luu() - e.Lau(0)**2 / e.Lab(0, 0)
            k_mix = k_mix_red * \
                (kb * temp / (self.mass1 * diameter1**2))**0.5 * kb / diameter1
            return k_mix

        k1 = self.params.fluid1.thermal_conductivity(temp)
        k2 = self.params.fluid2.thermal_conductivity(temp)
        k_mix = x1 * k1 + x2 * \
            k2 if self.params.is_max else 1 / (x1 / k1 + x2 / k2)
        return k_mix

    def dynamic_viscosity(self, temp: float) -> float:
        mu_1 = self.params.fluid1.dynamic_viscosity(temp)
        mu_2 = self.params.fluid2.dynamic_viscosity(temp)
        x1 = 1 - self.params.x2
        mu_mixture = mu_1 * x1 + mu_2 * self.params.x2
        return mu_mixture

    def specific_heat_capacities(self, temp: float) -> Tuple[float, float]:
        cv_1, cp_1 = self.params.fluid1.specific_heat_capacities(temp)
        cv_2, cp_2 = self.params.fluid2.specific_heat_capacities(temp)
        x1 = 1 - self.params.x2
        # Weighted average for specific heat capacities
        cp_mix = self.params.x2 * cp_2 + x1 * cp_1
        cv_mix = self.params.x2 * cv_2 + x1 * cv_1
        return cv_mix, cp_mix
