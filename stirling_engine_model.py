"""
Stirling Engine Simulation

This module provides a comprehensive simulation of a Stirling engine,
including ideal adiabatic analysis, regenerator heat loss calculation,
and pressure drop analysis. It uses object-oriented programming
principles and adheres to PEP 8 style guidelines. It is based on the work of Urieli.

Author: Igben O. Nehemiah
Date: 18/07/2024
Version: 1.0
"""

from abc import ABC, abstractmethod
from typing import Tuple, Callable
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import pyenskog
import matplotlib.pyplot as plt


class WorkingFluid(ABC):
    """
    Abstract base class for a working fluid.
    Defines the interface for thermal conductivity, dynamic viscosity, and specific heat capacities.
    """

    def __init__(self):
        self.R_specific = 0  # Specific gas constant
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
        pass

    @abstractmethod
    def dynamic_viscosity(self, temp: float) -> float:
        """
        Calculate the dynamic viscosity of the working fluid at a given temperature.

        Args: 
            temp: Temperature in Kelvin
        
        Returns: 
            Dynamic viscosity in Pa·s
        """
        pass

    @abstractmethod
    def specific_heat_capacities(self, temp: float) -> Tuple[float, float]:
        """
        Calculate the specific heat capacities (C_v and C_p) of the working fluid at a given temperature.

        Args:
            temp: Temperature in Kelvin
        
        Returns:
            Tuple of (C_v, C_p) in J/(kg·K)
        """
        pass


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
        b: float         # Coefficient for C_v polynomial term T^1 in J/(kg·K^2)
        c: float         # Coefficient for C_v polynomial term T^2 in J/(kg·K^3)
        d: float         # Coefficient for C_v polynomial term T^3 in J/(kg·K^4)
        e: float         # Coefficient for C_v polynomial term T^4 in J/(kg·K^5)
        molar_mass: float  # Molar mass in kg/mol

    def __init__(self, params: Parameters):
        self.params = params
        self.R_specific = 8.314 / self.params.molar_mass  # Specific gas constant in J/(kg·K)
        self.molar_mass = self.params.molar_mass

    def thermal_conductivity(self, temp: float) -> float:
        k = self.params.k0 * ((self.params.k_t0 + self.params.k_t_suth) /
                              (temp + self.params.k_t_suth) * (temp / self.params.k_t0)**1.5)
        return k

    def dynamic_viscosity(self, temp: float) -> float:
        mu = self.params.mu0 * ((self.params.mu_t0 + self.params.mu_t_suth) /
                                (temp + self.params.mu_t_suth) * (temp / self.params.mu_t0)**1.5)
        return mu

    def specific_heat_capacities(self, temp: float) -> Tuple[float, float]:
        t = temp / 1000.0
        cp_mol = (self.params.a + self.params.b * t +
                  self.params.c * t**2 + self.params.d * t**3 +
                  self.params.e / t**2)  # Shomate equation for C_p in J/(mol·K)
        cp_mass = cp_mol / self.params.molar_mass  # Convert C_p from J/(mol·K) to J/(kg·K)
        cv_mass = cp_mass - self.R_specific  # Calculate C_v
        return cv_mass, cp_mass


class GasMixtures(WorkingFluid):
    """
    Class representing a binary gas mixture working fluid.
    """
    @dataclass
    class Parameters:
        fluid1: WorkingFluid
        fluid2: WorkingFluid
        x2: float
        is_classical: bool
        is_max: bool

    def __init__(self, params: Parameters):
        self.params = params
        self.kb = 1.380649e-23  # Boltzmann constant in J/K
        avogadro = 6.022e23  # Avogadro's constant in particles/mol
        x1 = 1 - self.params.x2
        self.molar_mass = self.params.fluid1.molar_mass * x1 + self.params.fluid2.molar_mass * self.params.x2
        self.R_specific = 8.314 / self.molar_mass

        kT = 1.0  # This is an "uninteresting" variable for hard spheres, so is set to 1
        verbose = False  # Don't output lots of data
        use_tables = False
        SonineOrder = 10  # Reduced the order of the SonineOrder to 10
        EOS = pyenskog.EOStype.BMCSL  # Choose the equation of state used

        density = 0.00001  # Not actual density but taken in limit as Density -> 0

        self.e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)

        # Helium
        self.diameter2 = 208e-12
        molar_mass_helium = 4e-3  # Molar mass in kg/mol
        self.mass2 = molar_mass_helium / avogadro  # in kg

        # Hydrogen
        self.diameter1 = 228e-12
        molar_mass_hydrogen = 2e-3  # Molar mass in kg/mol
        self.mass1 = molar_mass_hydrogen / avogadro  # in kg

        self.e.addSpecies(self.diameter2 / self.diameter1, self.mass2 / self.mass1, self.params.x2)
        self.e.addSpecies(1, 1, x1)
        self.e.init()

    def thermal_conductivity(self, temp: float) -> float:
        x1, x2 = 1 - self.params.x2, self.params.x2
        if not self.params.is_classical:
            k_mix_red = self.e.Luu() if self.params.is_max else self.e.Luu() - self.e.Lau(0)**2 / self.e.Lab(0, 0)
            k_mix = k_mix_red * (self.kb * temp / (self.mass1 * self.diameter1**2))**0.5 * self.kb / self.diameter1
            return k_mix

        k1 = self.params.fluid1.thermal_conductivity(temp)
        k2 = self.params.fluid2.thermal_conductivity(temp)
        k_mix = x1 * k1 + x2 * k2 if self.params.is_max else 1 / (x1 / k1 + x2 / k2)
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
        cp_mix = self.params.x2 * cp_2 + x1 * cp_1  # Weighted average for specific heat capacities
        cv_mix = self.params.x2 * cv_2 + x1 * cv_1
        return cv_mix, cp_mix


class StirlingEngine:
    """
    A class representing a Stirling engine simulation.

    This class encapsulates all the necessary components and calculations
    for simulating a Stirling engine, including thermodynamic processes,
    heat transfer, and performance metrics.
    """

    class Index(IntEnum):
        """Enumeration for indexing state variables and derivatives."""
        TC = 0   # Compression space temperature (K)
        TE = 1   # Expansion space temperature (K)
        QK = 2   # Heat transferred to the cooler (J)
        QR = 3   # Heat transferred to the regenerator (J)
        QH = 4   # Heat transferred to the heater (J)
        WC = 5   # Work done by the compression space (J)
        WE = 6   # Work done by the expansion space (J)
        W = 7    # Total work done (WC + WE) (J)
        P = 8    # Pressure (Pa)
        VC = 9   # Compression space volume (m^3)
        VE = 10  # Expansion space volume (m^3)
        MC = 11  # Mass of gas in the compression space (kg)
        MK = 12  # Mass of gas in the cooler (kg)
        MR = 13  # Mass of gas in the regenerator (kg)
        MH = 14  # Mass of gas in the heater (kg)
        ME = 15  # Mass of gas in the expansion space (kg)
        TCK = 16  # Conditional temperature compression space / cooler (K)
        THE = 17  # Conditional temperature heater / expansion space (K)
        MCK = 18  # Conditional mass flow compression space / cooler (kg/rad)
        MKR = 19  # Conditional mass flow cooler / regenerator (kg/rad)
        MRH = 20  # Conditional mass flow regenerator / heater (kg/rad)
        MHE = 21  # Conditional mass flow heater / expansion space (kg/rad)

    @dataclass
    class Parameters:
        """Dataclass for storing Stirling engine parameters."""
        freq: float = 0.0
        charge_pressure: float = 0.0
        tk: float = 0.0
        tr: float = 0.0
        th: float = 0.0
        twk: float = 0.0
        twh: float = 0.0
        vk: float = 0.0
        vr: float = 0.0
        vh: float = 0.0
        inside_dia_h_tube: float = 0.0
        no_h_tubes: int = 0
        len_h_tube: float = 0.0
        inside_dia_k_tube: float = 0.0
        no_k_tubes: int = 0
        len_k_tube: float = 0.0
        no_r_tubes: int = 0
        reg_porosity: float = 0.0
        reg_len: float = 0.0
        dwire: float = 0.0
        domat: float = 0.0
        dout: float = 0.0
        mgas: float = 0.0
        gamma: float = 0.0
        working_fluid: None | WorkingFluid = None
        vclc: float = 0.0
        vcle: float = 0.0
        vswc: float = 0.0
        vswe: float = 0.0
        alpha: float = 0.0

    def __init__(self, params: Parameters):
        """
        Initialise the StirlingEngine instance.

        Args:
            params (Parameters): Engine parameters
        """
        self.params = params
        self._initialise_engine_parameters()
        self._initialise_heat_exchanger_parameters()

        self.dependent_var_indices = [
            self.Index.TC, self.Index.TE, self.Index.QK,
            self.Index.QR, self.Index.QH, self.Index.WC, self.Index.WE
        ]

    def _initialise_engine_parameters(self):
        """Initialise derived engine parameters."""
        self.params.th, self.params.tk = self.params.twh, self.params.twk
        
        self.params.tr = (self.params.th - self.params.tk) / \
            np.log(self.params.th / self.params.tk)
        self.omega = 2 * np.pi * self.params.freq

    def _initialise_heat_exchanger_parameters(self):
        """Initialise heat exchanger parameters."""
        self.vh, self.ah, self.awgh, self.dh, self.lh = self._calculate_pipe_parameters(
            self.params.no_h_tubes, self.params.inside_dia_h_tube, self.params.len_h_tube
        )
        self.vk, self.ak, self.awgk, self.dk, self.lk = self._calculate_pipe_parameters(
            self.params.no_k_tubes, self.params.inside_dia_k_tube, self.params.len_k_tube
        )
        self.vr, self.ar, self.awgr, self.dr, self.lr, self.cqwr = self._calculate_regenerator_parameters(
            self.params.no_r_tubes, self.params.reg_len, self.params.dwire
        )

    @staticmethod
    def _calculate_pipe_parameters(num: int, inner_dia: float, pipe_len: float) -> Tuple[float, float, float, float, float]:
        """
        Calculate pipe parameters.

        Args:
            num (int): Number of pipes
            inner_dia (float): Inner diameter of pipes
            pipe_len (float): Length of pipes

        Returns:
            Tuple[float, float, float, float, float]: Volume, area, wetted area, diameter, length
        """
        a = (num * np.pi * inner_dia**2) / 4
        v = a * pipe_len
        awg = num * np.pi * inner_dia * pipe_len
        return v, a, awg, inner_dia, pipe_len

    def _calculate_regenerator_parameters(self, num: int, lr: float, dwire: float) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate regenerator parameters.

        Args:
            num (int): Number of regenerator tubes
            lr (float): Regenerator effective length
            dwire (float): Wire diameter

        Returns:
            Tuple[float, float, float, float, float, float]: Volume, area, wetted area, hydraulic diameter, length, thermal conductance
        """
        dimat = 0
        awgr0 = num * np.pi * self.params.domat * lr
        amat = num * np.pi * (self.params.domat**2 - dimat**2) / 4
        ar = amat * self.params.reg_porosity
        vr = ar * lr
        dr = dwire * self.params.reg_porosity / (1 - self.params.reg_porosity)
        awgr = 4 * vr / dr + awgr0

        kwr = 25  # thermal conductivity [W/m/K]
        awr = num * np.pi * (self.params.dout**2 - self.params.domat**2) / 4
        cqwr = kwr * awr / lr
        return vr, ar, awgr, dr, lr, cqwr

    def run_simulation(self):
        """
        Run the Stirling engine simulation.

        This method performs the main simulation loop, including adiabatic analysis,
        regenerator heat loss calculation, and pressure drop analysis.
        """
        epsilon = 1e-6
        terror = 10 * epsilon

        var = np.zeros((22, 37))
        dvar = np.zeros((16, 37))

        self.params.mgas = self._schmidt_analysis()

        qrloss = 0
        while terror > epsilon:
            var, dvar = self._adiabatic_analysis()
            qrloss = self._calculate_regenerator_heat_loss(var)
            tgh = self._simulate_heater(var, self.params.twh, qrloss)
            tgk = self._simulate_cooler(var, self.params.twk, qrloss)
            terror = abs(self.params.th - tgh) + abs(self.params.tk - tgk)
            self.params.th = tgh
            self.params.tk = tgk
            self.params.tr = (self.params.th - self.params.tk) / \
                np.log(self.params.th / self.params.tk)

        self._print_results(qrloss, var, dvar)

    def _schmidt_analysis(self) -> float:
        """
        Perform Schmidt analysis of the Stirling engine to determine the mass of the working fluid.

        Returns:
            float: Mass of the working fluid
        """
        c = (((self.params.vswe/self.params.th)**2 + (self.params.vswc/self.params.tk)**2 + 2 *
             (self.params.vswe/self.params.th)*(self.params.vswc/self.params.tk)*np.cos(self.params.alpha))**0.5)/2
        s = (self.params.vswc/2 + self.params.vclc + self.params.vk)/self.params.tk + self.params.vr / \
            self.params.tr + (self.params.vswe/2 +
                              self.params.vcle + self.params.vh)/self.params.th
        b = c/s
        sqrtb = (1 - b**2)**0.5

        mgas = self.params.charge_pressure*s*sqrtb/self.params.working_fluid.R_specific  # Total mass of working gas in engine
        return mgas

    def _adiabatic_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform adiabatic analysis of the Stirling engine.

        Returns:
            Tuple[np.ndarray, np.ndarray]: State variables and their derivatives
        """
        epsilon = 1
        max_iterations = 100
        increments = 360
        step = increments // 36
        dtheta = 2.0 * np.pi / increments

        var = np.zeros((22, 37))
        dvar = np.zeros((16, 37))
        y = np.zeros(22)

        y[self.Index.THE] = self.params.th
        y[self.Index.TCK] = self.params.tk
        y[self.Index.TE] = self.params.th
        y[self.Index.TC] = self.params.tk

        iter_count = 0
        temp_error = 10 * epsilon

        while temp_error >= epsilon and iter_count < max_iterations:
            tc0, te0 = y[self.Index.TC], y[self.Index.TE]
            theta = 0
            y[self.Index.QK:self.Index.W + 1] = 0

            for _ in range(increments):
                theta, y, dy = self._runge_kutta_4(
                    self._adiabatic_derivatives, theta, dtheta, y)

            temp_error = abs(tc0 - y[self.Index.TC]) + \
                abs(te0 - y[self.Index.TE])
            iter_count += 1

        theta = 0
        y[self.Index.QK:self.Index.W + 1] = 0
        var, dvar = self._fill_matrix(0, y, dy, var, dvar)

        for i in range(1, 37):
            for _ in range(step):
                theta, y, dy = self._runge_kutta_4(
                    self._adiabatic_derivatives, theta, dtheta, y)
            var, dvar = self._fill_matrix(i, y, dy, var, dvar)

        return var, dvar

    def _adiabatic_derivatives(self, theta: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the derivatives for the adiabatic analysis.

        Args:
            theta (float): Crank angle
            y (np.ndarray): State variables

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated state variables and their derivatives
        """
        dy = np.zeros(16)

        y[self.Index.VC], y[self.Index.VE], dy[self.Index.VC], dy[self.Index.VE] = self._calculate_volumes(
            theta)

        t_avg = 0.5*(self.params.th + self.params.tk)
        self.cv, self.cp = self.params.working_fluid.specific_heat_capacities(
            t_avg)
        self.params.gamma = self.cp/self.cv

        vot = self.vk / self.params.tk + self.vr / \
            self.params.tr + self.vh / self.params.th
        y[self.Index.P] = (self.params.mgas * self.params.working_fluid.R_specific) / (
            y[self.Index.VC] / y[self.Index.TC] +
            vot + y[self.Index.VE] / y[self.Index.TE]
        )

        top = -y[self.Index.P] * (dy[self.Index.VC] /
                                  y[self.Index.TCK] + dy[self.Index.VE] / y[self.Index.THE])
        bottom = (
            y[self.Index.VC] / (y[self.Index.TCK] * self.params.gamma)
            + vot
            + y[self.Index.VE] / (y[self.Index.THE] * self.params.gamma)
        )
        dy[self.Index.P] = top / bottom

        self._calculate_mass_accumulations(y, dy)
        self._calculate_mass_flow(y, dy)
        self._calculate_conditional_temperatures(y)
        self._calculate_working_space_temperatures(y, dy)
        self._calculate_energy_transfers(y, dy)

        return y, dy

    def _calculate_volumes(self, theta: float) -> Tuple[float, float, float, float]:
        """
        Calculate volumes for sinusoidal motion.

        Args:
            theta (float): Crank angle

        Returns:
            Tuple[float, float, float, float]: Compression volume, expansion volume, and their derivatives
        """
        vc = self.params.vclc + 0.5 * \
            self.params.vswc * (1 + np.cos(theta + np.pi))
        ve = self.params.vcle + 0.5 * self.params.vswe * \
            (1 + np.cos(theta + self.params.alpha + np.pi))
        dvc = -0.5 * self.params.vswc * np.sin(theta + np.pi)
        dve = -0.5 * self.params.vswe * \
            np.sin(theta + self.params.alpha + np.pi)
        return vc, ve, dvc, dve

    def _calculate_mass_accumulations(self, y: np.ndarray, dy: np.ndarray):
        """
        Calculate mass accumulations and their derivatives.

        Args:
            y (np.ndarray): State variables
            dy (np.ndarray): Derivatives of state variables
        """
        y[self.Index.MC] = y[self.Index.P] * y[self.Index.VC] / \
            (self.params.working_fluid.R_specific * y[self.Index.TC])
        y[self.Index.MK] = y[self.Index.P] * self.vk / \
            (self.params.working_fluid.R_specific * self.params.tk)
        y[self.Index.MR] = y[self.Index.P] * self.vr / \
            (self.params.working_fluid.R_specific * self.params.tr)
        y[self.Index.MH] = y[self.Index.P] * self.vh / \
            (self.params.working_fluid.R_specific * self.params.th)
        y[self.Index.ME] = y[self.Index.P] * y[self.Index.VE] / \
            (self.params.working_fluid.R_specific * y[self.Index.TE])

        dy[self.Index.MC] = (y[self.Index.P] * dy[self.Index.VC] + y[self.Index.VC] *
                             dy[self.Index.P] / self.params.gamma) / (self.params.working_fluid.R_specific * y[self.Index.TCK])
        dy[self.Index.ME] = (y[self.Index.P] * dy[self.Index.VE] + y[self.Index.VE] *
                             dy[self.Index.P] / self.params.gamma) / (self.params.working_fluid.R_specific * y[self.Index.THE])

        dpop = dy[self.Index.P] / y[self.Index.P]
        dy[self.Index.MK] = y[self.Index.MK] * dpop
        dy[self.Index.MR] = y[self.Index.MR] * dpop
        dy[self.Index.MH] = y[self.Index.MH] * dpop

    def _calculate_mass_flow(self, y: np.ndarray, dy: np.ndarray):
        """
        Calculate mass flow between cells.

        Args:
            y (np.ndarray): State variables
            dy (np.ndarray): Derivatives of state variables
        """
        y[self.Index.MCK] = -dy[self.Index.MC]
        y[self.Index.MKR] = y[self.Index.MCK] - dy[self.Index.MK]
        y[self.Index.MHE] = dy[self.Index.ME]
        y[self.Index.MRH] = y[self.Index.MHE] + dy[self.Index.MH]

    def _calculate_conditional_temperatures(self, y: np.ndarray):
        """
        Calculate conditional temperatures between cells.

        Args:
            y (np.ndarray): State variables
        """
        y[self.Index.TCK] = self.params.tk if y[self.Index.MCK] <= 0 else y[self.Index.TC]
        y[self.Index.THE] = y[self.Index.TE] if y[self.Index.MHE] <= 0 else self.params.th

    def _calculate_working_space_temperatures(self, y: np.ndarray, dy: np.ndarray):
        """
        Calculate working space temperatures.

        Args:
            y (np.ndarray): State variables
            dy (np.ndarray): Derivatives of state variables
        """
        dpop = dy[self.Index.P] / y[self.Index.P]
        dy[self.Index.TC] = y[self.Index.TC] * \
            (dpop + dy[self.Index.VC] / y[self.Index.VC] -
             dy[self.Index.MC] / y[self.Index.MC])
        dy[self.Index.TE] = y[self.Index.TE] * \
            (dpop + dy[self.Index.VE] / y[self.Index.VE] -
             dy[self.Index.ME] / y[self.Index.ME])

    def _calculate_energy_transfers(self, y: np.ndarray, dy: np.ndarray):
        """
        Calculate energy transfers.

        Args:
            y (np.ndarray): State variables
            dy (np.ndarray): Derivatives of state variables
        """
        dy[self.Index.QK] = self.vk * dy[self.Index.P] * self.cv / self.params.working_fluid.R_specific - self.cp * (
            y[self.Index.TCK] * y[self.Index.MCK] -
            self.params.tk * y[self.Index.MKR]
        )
        dy[self.Index.QR] = self.vr * dy[self.Index.P] * self.cv / self.params.working_fluid.R_specific - self.cp * (
            self.params.tk * y[self.Index.MKR] -
            self.params.th * y[self.Index.MRH]
        )
        dy[self.Index.QH] = self.vh * dy[self.Index.P] * self.cv / self.params.working_fluid.R_specific - self.cp * (
            self.params.th * y[self.Index.MRH] -
            y[self.Index.THE] * y[self.Index.MHE]
        )
        dy[self.Index.WC] = y[self.Index.P] * dy[self.Index.VC]
        dy[self.Index.WE] = y[self.Index.P] * dy[self.Index.VE]

        dy[self.Index.W] = dy[self.Index.WC] + dy[self.Index.WE]
        y[self.Index.W] = y[self.Index.WC] + y[self.Index.WE]

    def _runge_kutta_4(self, derivs: Callable, theta: float, dtheta: float, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Perform a single step of the fourth-order Runge-Kutta method.

        Args:
            derivs (Callable): Function to calculate derivatives
            theta (float): Current angle
            dtheta (float): Angle step
            y (np.ndarray): Current state variables

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: New angle, updated state variables, and their derivatives
        """
        theta0 = theta
        y0 = y.copy()

        y, dy1 = derivs(theta0, y)
        y_temp = y.copy()
        y_temp[self.dependent_var_indices] = y0[self.dependent_var_indices] + \
            0.5 * dtheta * dy1[self.dependent_var_indices]

        theta_m = theta0 + 0.5 * dtheta
        y, dy2 = derivs(theta_m, y_temp)
        y_temp = y.copy()
        y_temp[self.dependent_var_indices] = y0[self.dependent_var_indices] + \
            0.5 * dtheta * dy2[self.dependent_var_indices]

        y, dy3 = derivs(theta_m, y_temp)
        y_temp = y.copy()
        y_temp[self.dependent_var_indices] = y0[self.dependent_var_indices] + \
            dtheta * dy3[self.dependent_var_indices]

        theta = theta0 + dtheta
        y, dy = derivs(theta, y_temp)

        dy[self.dependent_var_indices] = (dy1[self.dependent_var_indices] + 2*(
            dy2[self.dependent_var_indices] + dy3[self.dependent_var_indices]) + dy[self.dependent_var_indices]) / 6
        y[self.dependent_var_indices] = y0[self.dependent_var_indices] + \
            dtheta * dy[self.dependent_var_indices]

        return theta, y, dy

    def _fill_matrix(self, idx: int, y: np.ndarray, dy: np.ndarray, var: np.ndarray, dvar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill the matrix with current state variables and their derivatives.

        Args:
            idx (int): Current index
            y (np.ndarray): State variables
            dy (np.ndarray): Derivatives of state variables
            var (np.ndarray): Matrix of state variables
            dvar (np.ndarray): Matrix of derivatives

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated matrices of state variables and derivatives
        """
        var[:, idx] = y
        dvar[:, idx] = dy
        return var, dvar

    def _calculate_regenerator_heat_loss(self, var: np.ndarray) -> float:
        """
        Calculate the regenerator heat loss.

        Args:
            var (np.ndarray): State variables

        Returns:
            float: Regenerator heat loss
        """
        re = np.zeros(37)
        for i in range(37):
            gar = (var[self.Index.MKR, i] +
                   var[self.Index.MRH, i]) * self.omega / 2
            gr = gar / self.ar
            _, _, re[i] = self._calculate_reynolds(self.params.tr, gr, self.dr)

        reavg = np.mean(re)
        st, _ = self._calculate_matrix_friction(reavg)
        ntu = st * self.awgr / (2 * self.ar)
        effect = ntu / (ntu + 1)

        qrmin = np.min(var[self.Index.QR])
        qrmax = np.max(var[self.Index.QR])
        qrloss = (1 - effect) * (qrmax - qrmin)

        return qrloss

    def _calculate_reynolds(self, t: float, g: float, d: float) -> Tuple[float, float, float]:
        """
        Evaluate dynamic viscosity, thermal conductivity, and Reynolds number.

        Args:
            t (float): Temperature
            g (float): Mass flow rate per unit area
            d (float): Characteristic diameter

        Returns:
            Tuple[float, float, float]: Dynamic viscosity, thermal conductivity, and Reynolds number
        """
        mu = self.params.working_fluid.dynamic_viscosity(t)
        re = max(1, abs(g) * d / mu)
        return mu, self.params.working_fluid.thermal_conductivity(t), re

    def _calculate_matrix_friction(self, re: float) -> Tuple[float, float]:
        """
        Evaluate regenerator mesh matrix Stanton number and friction factor.

        Args:
            re (float): Reynolds number

        Returns:
            Tuple[float, float]: Stanton number and friction factor
        """
        _, cp = self.params.working_fluid.specific_heat_capacities(
            self.params.tr)
        mu = self.params.working_fluid.dynamic_viscosity(self.params.tr)
        k = self.params.working_fluid.thermal_conductivity(self.params.tr)
        prandtl = cp*mu/k
        st = 0.46 * re**(-0.4) / prandtl
        fr = 54 + 1.43 * re**0.78
        return st, fr

    def _simulate_heater(self, var: np.ndarray, twh: float, qrloss: float) -> float:
        """
        Evaluate heater average heat transfer performance.

        Args:
            var (np.ndarray): State variables
            twh (float): Heater wall temperature
            qrloss (float): Regenerator heat loss

        Returns:
            float: New heater gas temperature
        """
        re = np.zeros(37)
        for i in range(37):
            gah = (var[self.Index.MRH, i] +
                   var[self.Index.MHE, i]) * self.omega / 2
            gh = gah / self.ah
            mu, _, re[i] = self._calculate_reynolds(
                self.params.th, gh, self.dh)

        reavg = np.mean(re)
        ht, _ = self._calculate_pipe_friction(self.dh, mu, reavg)
        tgh = twh - (var[self.Index.QH, -1] + qrloss) * \
            self.params.freq / (ht * self.awgh)
        return tgh

    def _simulate_cooler(self, var: np.ndarray, twk: float, qrloss: float) -> float:
        """
        Evaluate cooler average heat transfer performance.

        Args:
            var (np.ndarray): State variables
            twk (float): Cooler wall temperature
            qrloss (float): Regenerator heat loss

        Returns:
            float: New cooler gas temperature
        """
        re = np.zeros(37)
        for i in range(37):
            gak = (var[self.Index.MCK, i] +
                   var[self.Index.MKR, i]) * self.omega / 2
            gk = gak / self.ak
            mu, _, re[i] = self._calculate_reynolds(
                self.params.tk, gk, self.dk)

        reavg = np.mean(re)
        ht, _ = self._calculate_pipe_friction(self.dk, mu, reavg)
        tgk = twk - (var[self.Index.QK, -1] - qrloss) * \
            self.params.freq / (ht * self.awgk)
        return tgk

    def _calculate_pipe_friction(self, d: float, mu: float, re: float) -> Tuple[float, float]:
        """
        Evaluate heat transfer coefficient and Reynolds friction factor for pipes.

        Args:
            d (float): Pipe diameter
            mu (float): Dynamic viscosity
            re (float): Reynolds number

        Returns:
            Tuple[float, float]: Heat transfer coefficient and friction factor
        """
        t_avg = 0.5*(self.params.th + self.params.tk)
        _, cp = self.params.working_fluid.specific_heat_capacities(
            t_avg)
        mu = self.params.working_fluid.dynamic_viscosity(t_avg)
        k = self.params.working_fluid.thermal_conductivity(t_avg)
        prandtl = cp*mu/k
        fr = 0.0791 * re**0.75
        ht = fr * mu * self.cp / (2 * d * prandtl)
        return ht, fr

    def _calculate_pressure_drop(self, var: np.ndarray, dvar: np.ndarray) -> float:
        dtheta = 2*np.pi/36
        dwork = 0  # initialise pumping work loss
        dpkol = np.zeros((37,))
        dpreg = np.zeros_like(dpkol)
        dphot = np.zeros_like(dpkol)
        dp = np.zeros_like(dpkol)
        pcom = np.zeros_like(dpkol)
        pexp = np.zeros_like(dpkol)

        for i in range(36):
            # Cooler pressure drop
            gk = (var[self.Index.MCK, i] + var[self.Index.MKR, i]) * \
                self.omega/(2*self.ak)

            mu, _, re = self._calculate_reynolds(self.params.tk, gk, self.dk)
            _, fr = self._calculate_pipe_friction(self.dk, mu, re)

            dpkol[i] = 2*fr*mu*self.vk*gk*self.lk / \
                (var[self.Index.MK, i]*self.dk**2)

            # Regenerator pressure drop
            gr = (var[self.Index.MKR, i] + var[self.Index.MRH, i]) * \
                self.omega/(2*self.ar)

            mu, _, re = self._calculate_reynolds(self.params.tr, gr, self.dr)

            _, fr = self._calculate_matrix_friction(re)
            dpreg[i] = 2*fr*mu*self.vr*gr*self.lr / \
                (var[self.Index.MR, i]*self.dr**2)

            # Heater pressure drop
            gh = (var[self.Index.MRH, i] + var[self.Index.MHE, i]) * \
                self.omega/(2*self.ah)

            mu, _, re = self._calculate_reynolds(self.params.th, gh, self.dh)
            _, fr = self._calculate_pipe_friction(self.dh, mu, re)

            dphot[i] = 2*fr*mu*self.vk*gk*self.lk / \
                (var[self.Index.MK, i]*self.dk**2)

            dp[i] = dpkol[i] + dpreg[i] + dphot[i]  # Total pressure drop

            dwork += dtheta*dp[i]*dvar[self.Index.VE, i]  # Pumping work [J]
            pcom[i] = var[self.Index.P, i]
            pexp[i] = pcom[i] + dp[i]

        dpkol[-1] = dpkol[0]
        dpreg[-1] = dpreg[0]
        dphot[-1] = dphot[0]
        dp[-1] = dp[0]
        pcom[-1] = pcom[0]
        pexp[-1] = pexp[0]

        return dwork

    def _print_results(self, qrloss:float, var: np.ndarray, dvar: np.ndarray,):
        """
        Print out the results of the Stirling engine simulation.

        Args:
            var (np.ndarray): State variables
        """
        eff = var[self.Index.W, -1] / var[self.Index.QH, -1]
        qk_power = var[self.Index.QK, -1] * self.params.freq
        qr_power = var[self.Index.QR, -1] * self.params.freq
        qh_power = var[self.Index.QH, -1] * self.params.freq
        w_power = var[self.Index.W, -1] * self.params.freq

        print('========== Ideal Adiabatic Analysis Results ============')
        print(f'Heat transferred to the cooler: {qk_power:.2f} [W]')
        print(f'Net heat transferred to the regenerator: {qr_power:.2f} [W]')
        print(f'Heat transferred to the heater: {qh_power:.2f} [W]')
        print(f'Total power output: {w_power:.2f} [W]')
        print(f'Thermal efficiency: {eff * 100:.1f} [%]')
        print('============= Regenerator Analysis Results =============')
        print(f'Regenerator net enthalpy loss: {qrloss * self.params.freq:.1f} [W]')
        qwrl = self.cqwr * (self.params.twh - self.params.twk) / self.params.freq
        print(f'Regenerator wall heat leakage: {qwrl * self.params.freq:.1f} [W]')

        print('========= Pressure Drop Simple Analysis =============')
        dwork = self._calculate_pressure_drop(var, dvar)
        print(f'Pressure drop available work loss: {dwork * self.params.freq:.1f} [W]')
        actual_w_power = w_power - dwork * self.params.freq
        actual_qh_power = qh_power + qrloss * self.params.freq + qwrl * self.params.freq
        actual_eff = actual_w_power / actual_qh_power
        print(f'Actual power from simple analysis: {actual_w_power:.1f} [W]')
        print(f'Actual heat power in from simple analysis: {actual_qh_power:.1f} [W]')
        print(f'Actual efficiency from simple analysis: {actual_eff * 100:.1f} [%]')


def main():
    # Data parameters for hydrogen, helium, and air

    hydrogen_params = IdealGasWorkingFluid.Parameters(
        mu0=8.76e-6,
        mu_t0=273,
        mu_t_suth=72,
        k0=0.168,
        k_t0=273,
        k_t_suth=72,
        a=33.066178,
        b=-11.363417,
        c=11.432816,
        d=-2.772874,
        e=-0.158558,
        molar_mass=0.002016  # kg/mol
    )

    helium_params = IdealGasWorkingFluid.Parameters(
        mu0=19.0e-6,
        mu_t0=273,
        mu_t_suth=79.4,
        k0=0.1513,
        k_t0=273,
        k_t_suth=79.4,
        a=20.78603,
        b=4.850638e-10,
        c=-1.582916e-10,
        d=-1.525102e-11,
        e=3.1963470e-11,
        molar_mass=0.0040026  # kg/mol
    )

    nitrogen_params = IdealGasWorkingFluid.Parameters(
        mu0=17.81e-6,     # Viscosity at reference temperature [Pa·s]
        mu_t0=300,        # Reference temperature for viscosity [K]
        mu_t_suth=111,    # Sutherland's constant for viscosity [K]
        k0=0.02583,       # Thermal conductivity at reference temperature [W/(m·K)]
        k_t0=300,         # Reference temperature for thermal conductivity [K]
        k_t_suth=111,     # Sutherland's constant for thermal conductivity [K]
        a=28.98641,
        b=1.853978,
        c=-9.647459,
        d=16.63537,
        e=0.000117,
        molar_mass=28.0134e-3  # kg/mol
    )

    # Creating instances for each gas
    hydrogen = IdealGasWorkingFluid(hydrogen_params)
    helium = IdealGasWorkingFluid(helium_params)
    nitrogen = IdealGasWorkingFluid(nitrogen_params)

    hydrogen_helium_params = GasMixtures.Parameters(
        fluid1=hydrogen,
        fluid2=helium,
        x2=.85,
        is_classical=False,
        is_max=False
    )

    hydrogen_helium = GasMixtures(hydrogen_helium_params)

    # Define engine parameters
    params_mod = StirlingEngine.Parameters(
        freq=41.7,  # Operating frequency (Hz)
        charge_pressure=4.13e6,  # Charge pressure (Pa)
        twk=288,  # Cooler wall temperature (K)
        twh=977,  # Heater wall temperature (K)
        inside_dia_h_tube=0.00302,  # Heater tube inside diameter (m)
        no_h_tubes=40,  # Number of heater tubes
        len_h_tube=0.2423,  # Heater tube length (m)
        inside_dia_k_tube=0.00108,  # Cooler tube inside diameter (m)
        no_k_tubes=312,  # Number of cooler tubes
        len_k_tube=0.0461,  # Cooler tube length (m)
        no_r_tubes=8,  # Number of regenerator tubes
        reg_porosity=0.697,  # Regenerator matrix porosity
        reg_len=0.022,  # Regenerator length (m)
        dwire=0.0000406,  # Regenerator matrix wire diameter (m)
        domat=0.023,  # Regenerator housing internal diameter (m)
        dout=0.03,  # Regenerator housing external diameter (m)
        working_fluid=hydrogen,
        vclc=0.0000468,  # Compression space total volume (m^3)
        vcle=0.0000168,  # Expansion space total volume (m^3)
        vswc=0.0001128,  # Cooler void volume (m^3)
        vswe=0.0001128,  # Heater void volume (m^3)
        alpha=np.pi/2  # Expansion phase angle advance (radians)
    )

    # Create StirlingEngine instance and run simulation
    engine = StirlingEngine(params_mod)
    engine.run_simulation()


if __name__ == "__main__":
    main()
