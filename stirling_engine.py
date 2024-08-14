from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from working_fluid import GasMixtures, WorkingFluid

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
    class EngineKPIs:
        net_heat: float = 0
        net_power: float = 0
        net_work: float = 0
        efficiency: float = 0

    @dataclass
    class Parameters:
        """Dataclass for storing Stirling engine parameters."""
        freq: float = 0.0
        mean_pressure: float = 0.0
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
        k_reg: float = 0.0
        dwire: float = 0.0
        domat: float = 0.0
        dout: float = 0.0
        mgas: float = 0.0
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
        self.engineKPIs = StirlingEngine.EngineKPIs()
        self._initialise_engine_parameters()
        self._initialise_heat_exchanger_parameters()

        self.dependent_var_indices = [
            self.Index.TC, self.Index.TE, self.Index.QK,
            self.Index.QR, self.Index.QH, self.Index.WC, self.Index.WE
        ]

    def _initialise_engine_parameters(self):
        """Initialise derived engine parameters."""
        self.params.th, self.params.tk = self.params.twh, self.params.twk

        self.params.tr = self._calculate_log_mean_temperature(
            self.params.th, self.params.tk)
        self.omega = 2 * np.pi * self.params.freq

    @staticmethod
    def _calculate_log_mean_temperature(th: float, tk: float) -> float:
        return (th - tk)/np.log(th/tk)

    def _initialise_heat_exchanger_parameters(self):
        """Initialise heat exchanger parameters."""
        self.params.vh, self.params.ah, self.params.awgh, self.params.dh, self.params.lh = self._calculate_pipe_parameters(
            self.params.no_h_tubes, self.params.inside_dia_h_tube, self.params.len_h_tube
        )
        self.params.vk, self.params.ak, self.params.awgk, self.params.dk, self.params.lk = self._calculate_pipe_parameters(
            self.params.no_k_tubes, self.params.inside_dia_k_tube, self.params.len_k_tube
        )
        self.params.vr, self.params.ar, self.params.awgr, self.params.dr, self.params.lr, self.params.cqwr = self._calculate_regenerator_parameters(
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

        awr = num * np.pi * (self.params.dout**2 - self.params.domat**2) / 4
        cqwr = self.params.k_reg * awr / lr
        return vr, ar, awgr, dr, lr, cqwr

    def run_simulation(self):
        """
        Run the Stirling engine simulation.

        This method performs the main simulation loop, including adiabatic analysis,
        regenerator heat loss calculation, and pressure drop analysis.
        """
        epsilon = 1e-6
        temp_error = 10 * epsilon

        var = np.zeros((22, 37))
        dvar = np.zeros((16, 37))

        self.params.mgas = self._schmidt_analysis()

        qrloss = 0
        while temp_error > epsilon:
            var, dvar = self._adiabatic_analysis()
            qrloss = self._calculate_regenerator_heat_loss(var)
            tgh = self._simulate_heater(var, self.params.twh, qrloss)
            tgk = self._simulate_cooler(var, self.params.twk, qrloss)
            temp_error = abs(self.params.th - tgh) + abs(self.params.tk - tgk)
            self.params.th = tgh
            self.params.tk = tgk
            self.params.tr = self._calculate_log_mean_temperature(
                self.params.th, self.params.tk)

        self._set_engine_KPI(qrloss, var, dvar)
        # self._make_plots(var)

    def _schmidt_analysis(self) -> float:
        """
        Perform Schmidt analysis of the Stirling engine to determine the mass of the working fluid and engine performance.

        Returns:
            float: Mass of the working fluid
        """
        # Unpack parameters for easier access
        params = self.params
        vswe, vswc = params.vswe, params.vswc
        th, tk, alpha = params.th, params.tk, params.alpha
        rgas = params.working_fluid.specific_gas_constant

        # Calculate geometric factor 'c'
        c = np.sqrt((vswe / th)**2 + (vswc / tk)**2 + 2 *
                    (vswe / th) * (vswc / tk) * np.cos(alpha)) / 2

        # Calculate sum of volumes over their respective temperatures
        s = ((vswc / 2 + params.vclc + params.vk) / tk +
             params.vr / params.tr +
             (vswe / 2 + params.vcle + params.vh) / th)

        # Calculate the 'b' and 'sqrtb' factors for the Schmidt analysis
        b = c / s
        sqrtb = np.sqrt(1 - b**2)

        # Calculate the mass of the working gas
        mgas = params.mean_pressure * s * sqrtb / \
            rgas  # Total mass of working gas in engine

        # Calculate the phase angle beta
        beta = np.arctan((vswe * np.sin(alpha) / th) /
                         (vswe * np.cos(alpha) / th + vswc / tk))

        # Calculate the factor bf
        bf = 1 - 1 / sqrtb

        # Calculate work done by the compression and expansion processes
        wc = np.pi * vswc * mgas * rgas * np.sin(beta) * bf / c
        we = np.pi * vswe * mgas * rgas * np.sin(beta - alpha) * bf / c

        # Total work and power
        w = wc + we
        power = w * params.freq

        # Efficiency of the engine
        efficiency = w / we

        # Output results
        print('=====================  Schmidt Analysis  =================')
        print(f' Work (Joules): {w:.3e},  Power (Watts): {power:.3e}')
        print(f' Qexp (Joules): {we:.3e},  Qcom (Joules): {wc:.3e}')
        print(f' Indicated Efficiency: {100*efficiency:.3f}')
        print('==========================================================')

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
        cv, cp = self.params.working_fluid.specific_heat_capacities(
            t_avg)
        gamma = cp/cv

        vot = self.params.vk / self.params.tk + self.params.vr / \
            self.params.tr + self.params.vh / self.params.th
        y[self.Index.P] = (self.params.mgas * self.params.working_fluid.specific_gas_constant) / (
            y[self.Index.VC] / y[self.Index.TC] +
            vot + y[self.Index.VE] / y[self.Index.TE]
        )

        top = -y[self.Index.P] * (dy[self.Index.VC] /
                                  y[self.Index.TCK] + dy[self.Index.VE] / y[self.Index.THE])
        bottom = (
            y[self.Index.VC] / (y[self.Index.TCK] * gamma)
            + vot
            + y[self.Index.VE] / (y[self.Index.THE] * gamma)
        )
        dy[self.Index.P] = top / bottom

        self._calculate_mass_accumulations(y, dy, gamma)
        self._calculate_mass_flow(y, dy)
        self._calculate_conditional_temperatures(y)
        self._calculate_working_space_temperatures(y, dy)
        self._calculate_energy_transfers(y, dy, cp, cv)

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

    def _calculate_mass_accumulations(self, y: np.ndarray, dy: np.ndarray, gamma: float):
        """
        Calculate mass accumulations and their derivatives.

        Args:
            y (np.ndarray): State variables
            dy (np.ndarray): Derivatives of state variables
        """
        y[self.Index.MC] = y[self.Index.P] * y[self.Index.VC] / \
            (self.params.working_fluid.specific_gas_constant *
             y[self.Index.TC])
        y[self.Index.MK] = y[self.Index.P] * self.params.vk / \
            (self.params.working_fluid.specific_gas_constant * self.params.tk)
        y[self.Index.MR] = y[self.Index.P] * self.params.vr / \
            (self.params.working_fluid.specific_gas_constant * self.params.tr)
        y[self.Index.MH] = y[self.Index.P] * self.params.vh / \
            (self.params.working_fluid.specific_gas_constant * self.params.th)
        y[self.Index.ME] = y[self.Index.P] * y[self.Index.VE] / \
            (self.params.working_fluid.specific_gas_constant *
             y[self.Index.TE])

        dy[self.Index.MC] = (y[self.Index.P] * dy[self.Index.VC] + y[self.Index.VC] *
                             dy[self.Index.P] / gamma) / (self.params.working_fluid.specific_gas_constant * y[self.Index.TCK])
        dy[self.Index.ME] = (y[self.Index.P] * dy[self.Index.VE] + y[self.Index.VE] *
                             dy[self.Index.P] / gamma) / (self.params.working_fluid.specific_gas_constant * y[self.Index.THE])

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

    def _calculate_energy_transfers(self, y: np.ndarray, dy: np.ndarray, cp: float, cv: float):
        """
        Calculate energy transfers.

        Args:
            y (np.ndarray): State variables
            dy (np.ndarray): Derivatives of state variables
        """
        dy[self.Index.QK] = self.params.vk * dy[self.Index.P] * cv / self.params.working_fluid.specific_gas_constant - cp * (
            y[self.Index.TCK] * y[self.Index.MCK] -
            self.params.tk * y[self.Index.MKR]
        )
        dy[self.Index.QR] = self.params.vr * dy[self.Index.P] * cv / self.params.working_fluid.specific_gas_constant - cp * (
            self.params.tk * y[self.Index.MKR] -
            self.params.th * y[self.Index.MRH]
        )
        dy[self.Index.QH] = self.params.vh * dy[self.Index.P] * cv / self.params.working_fluid.specific_gas_constant - cp * (
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
            gr = gar / self.params.ar
            _, re[i] = self._calculate_reynolds(
                self.params.tr, gr, self.params.dr)

        reavg = np.mean(re)
        st, _ = self._calculate_matrix_friction(reavg)
        ntu = st * self.params.awgr / (2 * self.params.ar)
        effect = ntu / (ntu + 1)

        qrmin = np.min(var[self.Index.QR])
        qrmax = np.max(var[self.Index.QR])
        qrloss = (1 - effect) * (qrmax - qrmin)

        return qrloss

    def _calculate_reynolds(self, t: float, g: float, d: float) -> Tuple[float, float]:
        """
        Evaluate dynamic viscosity, thermal conductivity, and Reynolds number.

        Args:
            t (float): Temperature
            g (float): Mass flow rate per unit area
            d (float): Characteristic diameter

        Returns:
            Tuple[float, float]: Dynamic viscosity and Reynolds number
        """
        mu = self.params.working_fluid.dynamic_viscosity(t)
        re = max(1, abs(g) * d / mu)
        return mu, re

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
        # Kays & London (1995)
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
            gh = gah / self.params.ah
            mu, re[i] = self._calculate_reynolds(
                self.params.th, gh, self.params.dh)

        reavg = np.mean(re)
        ht = self._calculate_pipe_heat_transfer_coefficient(
            self.params.dh, mu, reavg, self.params.th)
        tgh = twh - (var[self.Index.QH, -1] + qrloss) * \
            self.params.freq / (ht * self.params.awgh)
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
            gk = gak / self.params.ak
            mu, re[i] = self._calculate_reynolds(
                self.params.tk, gk, self.params.dk)

        reavg = np.mean(re)
        ht = self._calculate_pipe_heat_transfer_coefficient(
            self.params.dk, mu, reavg, self.params.tk)
        tgk = twk - (var[self.Index.QK, -1] - qrloss) * \
            self.params.freq / (ht * self.params.awgk)
        return tgk

    def _calculate_pipe_heat_transfer_coefficient(self, d: float, mu: float, re: float, temp: float) -> float:
        """
        Evaluate heat transfer coefficient for pipes.

        Args:
            d (float): Pipe diameter
            mu (float): Dynamic viscosity
            re (float): Reynolds number
            temp (float): Temperature

        Returns:
            float: Heat transfer coefficient
        """
        _, cp = self.params.working_fluid.specific_heat_capacities(temp)
        mu = self.params.working_fluid.dynamic_viscosity(temp)
        k = self.params.working_fluid.thermal_conductivity(temp)
        prandtl = cp*mu/k
        fr = 0.0791 * re**0.75
        ht = fr * mu * cp / (2 * d * prandtl)
        return ht

    def _calculate_pressure_drop(self, var: np.ndarray, dvar: np.ndarray) -> float:
        dtheta = 2*np.pi/36
        dwork = 0  # initialise pumping work loss
        dpkol = np.zeros((37,))
        dpreg = np.zeros_like(dpkol)
        dphot = np.zeros_like(dpkol)
        dp = np.zeros_like(dpkol)
        self.pcom = np.zeros_like(dpkol)
        self.pexp = np.zeros_like(dpkol)

        for i in range(36):
            # Cooler pressure drop
            gk = (var[self.Index.MCK, i] + var[self.Index.MKR, i]) * \
                self.omega/(2*self.params.ak)

            mu, re = self._calculate_reynolds(
                self.params.tk, gk, self.params.dk)
            fr = self._calculate_pipe_friction(re)

            dpkol[i] = 2*fr*mu*self.params.vk*gk*self.params.lk / \
                (var[self.Index.MK, i]*self.params.dk**2)

            # Regenerator pressure drop
            gr = (var[self.Index.MKR, i] + var[self.Index.MRH, i]) * \
                self.omega/(2*self.params.ar)

            mu, re = self._calculate_reynolds(
                self.params.tr, gr, self.params.dr)

            _, fr = self._calculate_matrix_friction(re)
            dpreg[i] = 2*fr*mu*self.params.vr*gr*self.params.lr / \
                (var[self.Index.MR, i]*self.params.dr**2)

            # Heater pressure drop
            gh = (var[self.Index.MRH, i] + var[self.Index.MHE, i]) * \
                self.omega/(2*self.params.ah)

            mu, re = self._calculate_reynolds(
                self.params.th, gh, self.params.dh)
            fr = self._calculate_pipe_friction(re)

            dphot[i] = 2*fr*mu*self.params.vk*gk*self.params.lk / \
                (var[self.Index.MK, i]*self.params.dk**2)

            dp[i] = dpkol[i] + dpreg[i] + dphot[i]  # Total pressure drop

            dwork += dtheta*dp[i]*dvar[self.Index.VE, i]  # Pumping work [J]
            self.pcom[i] = var[self.Index.P, i]
            self.pexp[i] = self.pcom[i] + dp[i]

        dpkol[-1] = dpkol[0]
        dpreg[-1] = dpreg[0]
        dphot[-1] = dphot[0]
        dp[-1] = dp[0]
        self.pcom[-1] = self.pcom[0]
        self.pexp[-1] = self.pexp[0]

        return dwork

    def _calculate_pipe_friction(self, re: float) -> float:
        """
        Evaluate Reynolds friction factor for pipes.

        Args:
            re (float): Reynolds number

        Returns:
            float: Friction factor
        """
        if re < 2000:
            return 16
        if re >= 2000 and re < 4000:
            return (7.343e-4)*re**1.3142

        return 0.0791 * re**0.75

    def _set_engine_KPI(self, qrloss: float, var: np.ndarray, dvar: np.ndarray):
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
        qwrl = self.params.cqwr * \
            (self.params.twh - self.params.twk) / self.params.freq
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

        self.engineKPIs = self.EngineKPIs(actual_qh_power, actual_w_power, dwork * self.params.freq, actual_eff*100)
       
    def _make_plots(self, var: np.ndarray):
        self._plot_pv_diagram(var)
        self._plot_wall_and_gas_temperatures(var)

    def _plot_pv_diagram(self, var: np.ndarray):
        # plt.style.use('seaborn-v0_8-darkgrid')
        plt.style.use(["science", "ieee", "no-latex"])
        fig, ax = plt.subplots()

        ax.plot(var[self.Index.VC, :], self.pcom *
                1e-6, "r--", label="Compression")
        ax.plot(var[self.Index.VE, :], self.pexp*1e-6, "b-", label="Expansion")
        ax.set_xlabel("Volume ($m^3$)")
        ax.set_ylabel("Pressure (MPa)")

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.xaxis.get_offset_text().set_fontsize(10)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        ax.legend(loc='upper right', fontsize=12,
                  frameon=True, edgecolor='black')

        ax.grid(True)
        ax.legend()
        fig.savefig("pv_plot.png", dpi=300, bbox_inches='tight')

    def _plot_wall_and_gas_temperatures(self, var: np.ndarray):
        # Sample data, you should replace these with your actual data
        plt.style.use(["science", "ieee", "no-latex"])
        fig, ax = plt.subplots()

        x = np.arange(0, 361, 10)

        # Plotting Tcomp and Texp
        ax.plot(x, var[self.Index.TC, :], 'b-', label='$T_{comp}$')
        ax.plot(x, var[self.Index.TE, :], 'r-', label='$T_{exp}$')

        ax.plot([0, 360], [self.params.twk, self.params.twk],
                'b--', label='$T_{wk}$')
        ax.plot([0, 360], [self.params.tk, self.params.tk],
                'b', linestyle=(0, (2, 2)), label='$T_k$')
        # ax.plot([0, 360], [self.params.tr, self.params.tr], 'g-', label='tr')
        ax.plot([0, 360], [self.params.th, self.params.th],
                'r', linestyle=(0, (2, 2)), label='$T_h$')
        ax.plot([0, 360], [self.params.twh, self.params.twh],
                'r--', label='$T_{wh}$')

        # Adding grid, labels, and title
        ax.grid(True)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.set_xlabel('Crank angle (degrees)')
        ax.set_ylabel('Temperature (K)')
        ax.set_xlim(0, 360)

        ax.legend()
        fig.savefig("cycle_temperature_variation",
                    dpi=300, bbox_inches='tight')
