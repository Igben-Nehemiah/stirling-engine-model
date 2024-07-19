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

from typing import Tuple, Callable
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt


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
        rgas: float = 0.0
        mu0: float = 0.0
        t_suth: float = 0.0
        t0: float = 0.0
        kgas: float = 0.0
        prandtl: float = 0.0
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
        self._initialize_engine_parameters()
        self._initialize_heat_exchanger_parameters()

        self.DEP_VAR_INDICES = [
            self.Index.TC, self.Index.TE, self.Index.QK,
            self.Index.QR, self.Index.QH, self.Index.WC, self.Index.WE
        ]

    def _initialize_engine_parameters(self):
        """Initialize derived engine parameters."""
        self.params.tr = (self.params.th - self.params.tk) / \
            np.log(self.params.th / self.params.tk)
        self.omega = 2 * np.pi * self.params.freq
        self.cv = self.params.rgas / (self.params.gamma - 1)
        self.cp = self.params.gamma * self.cv

    def _initialize_heat_exchanger_parameters(self):
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

        Returns:
            None
        """
        twk = self.params.tk
        twh = self.params.th
        epsilon = 1e-6
        terror = 10 * epsilon

        var = np.zeros((22, 37))
        dvar = np.zeros((16, 37))

        self.params.mgas = self._schmidt_analysis()

        while terror > epsilon:
            var, dvar = self._adiabatic_analysis()
            qrloss = self._calculate_regenerator_heat_loss(var)
            tgh = self._simulate_heater(var, twh, qrloss)
            tgk = self._simulate_cooler(var, twk, qrloss)
            terror = abs(self.params.th - tgh) + abs(self.params.tk - tgk)
            self.params.th = tgh
            self.params.tk = tgk
            self.params.tr = (self.params.th - self.params.tk) / \
                np.log(self.params.th / self.params.tk)

        # Print out ideal adiabatic analysis results
        eff = var[self.Index.W, -1] / var[self.Index.QH, -1]
        qk_power = var[self.Index.QK, -1] * self.params.freq
        qr_power = var[self.Index.QR, -1] * self.params.freq
        qh_power = var[self.Index.QH, -1] * self.params.freq
        w_power = var[self.Index.W, -1] * self.params.freq

        # print('========== Ideal Adiabatic Analysis Results ============')
        # print(f'Heat transferred to the cooler: {qk_power:.2f} [W]')
        # print(f'Net heat transferred to the regenerator: {qr_power:.2f} [W]')
        # print(f'Heat transferred to the heater: {qh_power:.2f} [W]')
        # print(f'Total power output: {w_power:.2f} [W]')
        # print(f'Thermal efficiency: {eff * 100:.1f} [%]')
        # print('============= Regenerator Analysis Results =============')
        # print(f'Regenerator net enthalpy loss: {
        #       qrloss * self.params.freq:.1f} [W]')
        qwrl = self.cqwr * (twh - twk) / self.params.freq
        # print(f'Regenerator wall heat leakage: {
        #       qwrl * self.params.freq:.1f} [W]')

        print('========= Pressure Drop Simple Analysis =============')
        dwork = self._calculate_pressure_drop(var, dvar)
        print(f'Pressure drop available work loss: {
              dwork * self.params.freq:.1f} [W]')
        actual_w_power = w_power - dwork * self.params.freq
        actual_qh_power = qh_power + qrloss * self.params.freq + qwrl * self.params.freq
        actual_eff = actual_w_power / actual_qh_power
        print(f'Actual power from simple analysis: {actual_w_power:.1f} [W]')
        print(f'Actual heat power in from simple analysis: {
              actual_qh_power:.1f} [W]')
        print(f'Actual efficiency from simple analysis: {
              actual_eff * 100:.1f} [%]')

    def _schmidt_analysis(self) -> float:
        """
        Perform Schmidt analysis of the Stirling engine to determine the mass of the 
        working fluid.

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

        # total mass of working gas in engine
        mgas = self.params.charge_pressure*s*sqrtb/self.params.rgas

        return mgas

    def _adiabatic_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform adiabatic analysis of the Stirling engine.

        Returns:
            Tuple[np.ndarray, np.ndarray]: State variables and their derivatives
        """
        print('============ Ideal Adiabatic Analysis ====================')
        print(f'Cooler Tk = {self.params.tk:.1f}[K], Heater Th = {
              self.params.th:.1f}[K]')

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

            print(f'Iteration {iter_count}: Tc = {
                  y[self.Index.TC]:.1f}[K], Te = {y[self.Index.TE]:.1f}[K]')

            for _ in range(increments):
                theta, y, dy = self._runge_kutta_4(
                    self._adiabatic_derivatives, theta, dtheta, y)

            temp_error = abs(tc0 - y[self.Index.TC]) + \
                abs(te0 - y[self.Index.TE])
            iter_count += 1

        if iter_count >= max_iterations:
            print(f'No convergence within {max_iterations} iterations')

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

        vot = self.vk / self.params.tk + self.vr / \
            self.params.tr + self.vh / self.params.th
        y[self.Index.P] = (self.params.mgas * self.params.rgas) / (
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

        # Calculate mass accumulations and derivatives
        self._calculate_mass_accumulations(y, dy)

        # Calculate mass flow between cells
        self._calculate_mass_flow(y, dy)

        # Calculate conditional temperatures between cells
        self._calculate_conditional_temperatures(y)

        # Calculate working space temperatures
        self._calculate_working_space_temperatures(y, dy)

        # Calculate energy transfers
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
            (self.params.rgas * y[self.Index.TC])
        y[self.Index.MK] = y[self.Index.P] * self.vk / \
            (self.params.rgas * self.params.tk)
        y[self.Index.MR] = y[self.Index.P] * self.vr / \
            (self.params.rgas * self.params.tr)
        y[self.Index.MH] = y[self.Index.P] * self.vh / \
            (self.params.rgas * self.params.th)
        y[self.Index.ME] = y[self.Index.P] * y[self.Index.VE] / \
            (self.params.rgas * y[self.Index.TE])

        dy[self.Index.MC] = (y[self.Index.P] * dy[self.Index.VC] + y[self.Index.VC] *
                             dy[self.Index.P] / self.params.gamma) / (self.params.rgas * y[self.Index.TCK])
        dy[self.Index.ME] = (y[self.Index.P] * dy[self.Index.VE] + y[self.Index.VE] *
                             dy[self.Index.P] / self.params.gamma) / (self.params.rgas * y[self.Index.THE])

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
        dy[self.Index.QK] = self.vk * dy[self.Index.P] * self.cv / self.params.rgas - self.cp * (
            y[self.Index.TCK] * y[self.Index.MCK] -
            self.params.tk * y[self.Index.MKR]
        )
        dy[self.Index.QR] = self.vr * dy[self.Index.P] * self.cv / self.params.rgas - self.cp * (
            self.params.tk * y[self.Index.MKR] -
            self.params.th * y[self.Index.MRH]
        )
        dy[self.Index.QH] = self.vh * dy[self.Index.P] * self.cv / self.params.rgas - self.cp * (
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
        y_temp[self.DEP_VAR_INDICES] = y0[self.DEP_VAR_INDICES] + \
            0.5 * dtheta * dy1[self.DEP_VAR_INDICES]

        theta_m = theta0 + 0.5 * dtheta
        y, dy2 = derivs(theta_m, y_temp)
        y_temp = y.copy()
        y_temp[self.DEP_VAR_INDICES] = y0[self.DEP_VAR_INDICES] + \
            0.5 * dtheta * dy2[self.DEP_VAR_INDICES]

        y, dy3 = derivs(theta_m, y_temp)
        y_temp = y.copy()
        y_temp[self.DEP_VAR_INDICES] = y0[self.DEP_VAR_INDICES] + \
            dtheta * dy3[self.DEP_VAR_INDICES]

        theta = theta0 + dtheta
        y, dy = derivs(theta, y_temp)

        dy[self.DEP_VAR_INDICES] = (dy1[self.DEP_VAR_INDICES] + 2*(
            dy2[self.DEP_VAR_INDICES] + dy3[self.DEP_VAR_INDICES]) + dy[self.DEP_VAR_INDICES]) / 6
        y[self.DEP_VAR_INDICES] = y0[self.DEP_VAR_INDICES] + \
            dtheta * dy[self.DEP_VAR_INDICES]

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
        mu = self.params.mu0 * (self.params.t0 + self.params.t_suth) / \
            (t + self.params.t_suth) * (t / self.params.t0)**1.5
        self.params.prandtl = self.cp * mu / self.params.kgas
        re = max(1, abs(g) * d / mu)
        return mu, self.params.kgas, re

    def _calculate_matrix_friction(self, re: float) -> Tuple[float, float]:
        """
        Evaluate regenerator mesh matrix Stanton number and friction factor.

        Args:
            re (float): Reynolds number

        Returns:
            Tuple[float, float]: Stanton number and friction factor
        """
        st = 0.46 * re**(-0.4) / self.params.prandtl
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
        fr = 0.0791 * re**0.75
        ht = fr * mu * self.cp / (2 * d * self.params.prandtl)
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

            # Total pressure drop
            dp[i] = dpkol[i] + dpreg[i] + dphot[i]

            dwork += dtheta*dp[i]*dvar[self.Index.VE, i]  # pumping work [J]
            pcom[i] = var[self.Index.P, i]
            pexp[i] = pcom[i] + dp[i]

        dpkol[36] = dpkol[0]
        dpreg[36] = dpreg[0]
        dphot[36] = dphot[0]
        dp[36] = dp[0]
        pcom[36] = pcom[0]
        pexp[36] = pexp[0]

        return dwork

    def _calculate_pressure_drop2(self, var: np.ndarray, dvar: np.ndarray) -> float:
        dtheta = 2 * np.pi / 36
        dwork = 0  # Initialise pumping work loss
        dpkol = np.zeros(37)
        dpreg = np.zeros(37)
        dphot = np.zeros(37)
        dp = np.zeros(37)
        pcom = np.zeros(37)
        pexp = np.zeros(37)

        for i in range(36):
            # Cooler pressure drop
            gk = (var[self.Index.MCK, i] + var[self.Index.MKR, i]) * \
                self.omega / (2 * self.ak)
            mu, _, re = self._calculate_reynolds(self.params.tk, gk, self.dk)
            _, fr = self._calculate_pipe_friction(self.dk, mu, re)
            dpkol[i] = 2 * fr * mu * self.vk * \
                gk * self.lk / (var[self.Index.MK, i] * self.dk**2)

            # Regenerator pressure drop
            gr = (var[self.Index.MKR, i] +
                  var[self.Index.MRH, i]) / (2 * self.ar)
            mu, _, re = self._calculate_reynolds(self.params.tr, gr, self.dr)
            _, fr = self._calculate_matrix_friction(re)
            dpreg[i] = 2*fr*mu*self.vr*gr*self.lr / \
                (var[self.Index.MR, i]*self.dr**2)

            # Heater pressure drop
            gh = (var[self.Index.MRH, i] +
                  var[self.Index.MHE, i]) / (2 * self.ah)
            mu, _, re = self._calculate_reynolds(self.params.th, gh, self.dh)
            _, fr = self._calculate_pipe_friction(self.dh, mu, re)
            dphot[i] = 2 * fr * mu * self.vh * \
                gh * self.lh / (var[self.Index.MK, i] * self.dh**2)

            dp[i] = dpkol[i] + dpreg[i] + dphot[i]
            dwork += dp[i] * dtheta

            # Total pressure drop
            dp[i] = dpkol[i] + dpreg[i] + dphot[i]
            dwork += dtheta*dp[i]*dvar[self.Index.VE, i]  # pumping work [J]
            pcom[i] = var[self.Index.P, i]
            pexp[i] = pcom[i] + dp[i]

        dpkol[36] = dpkol[0]
        dpreg[36] = dpreg[0]
        dphot[36] = dphot[0]
        dp[36] = dp[0]
        pcom[36] = pcom[0]
        pexp[36] = pexp[0]

        return dwork


def main():
    # Define engine parameters
    params_mod = StirlingEngine.Parameters(
        freq=41.7,  # Operating frequency (Hz)
        charge_pressure=4130000,  # Charge pressure (Pa)
        tk=288,  # Cold sink temperature (K)
        th=977,  # Hot source temperature (K)
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

        # gamma=1.67,  # Specific heat ratio for helium
        # rgas=2078.6,  # Specific gas constant for helium (J/kg.K)
        # mu0=18.85e-6,  # Dynamic viscosity (kg/m.s)
        # t_suth=80.0,  # Sutherland constant for helium (K)
        # t0=273,  # Reference temperatu  re (K)
        # kgas=0.1513,  # Thermal conductivity of helium (W/m.K)

        # gamma=1.41,  # Specific heat ratio for hydrogen
        # rgas=4157.2,  # Gas constant for hydrogen [J/kg.K]
        # # Dynamic viscosity of hydrogen at reference temp t0 [kg.m/s]
        # mu0=8.35e-6,
        # t_suth=84.4,  # Sutherland constant for hydrogen [K]
        # t0=273,
        # kgas=0.168,  # Thermal conductivity for hydrogen [W/m.K]

        gamma=1.4,  # Specific heat ratio for air
        rgas=287.1,  # Gas constant for air [J/kg.K]
        mu0=17.08e-6,  # Dynamic viscosity of air at reference temp t0 [kg.m/s]
        t_suth=112,  # Sutherland constant for air [K]
        t0=273,
        kgas=0.025,  # Thermal conductivity for air [W/m.K]

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
