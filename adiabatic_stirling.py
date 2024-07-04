import numpy as np
from typing import Tuple, Callable
from enum import IntEnum
import matplotlib.pyplot as plt


class StirlingEngine:
    class Index(IntEnum):
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
        GACK = 18  # Conditional mass flow compression space / cooler (kg/rad)
        GAKR = 19  # Conditional mass flow cooler / regenerator (kg/rad)
        GARH = 20  # Conditional mass flow regenerator / heater (kg/rad)
        GAHE = 21  # Conditional mass flow heater / expansion space (kg/rad)

    def __init__(self):
        # Basic engine parameters
        self.freq = 0    # cycle frequency [Hz]
        self.tk = 0      # cooler temperature [K]
        self.tr = 0      # regenerator temperature [K]
        self.th = 0      # heater temperature [K]
        self.vk = 0      # cooler void volume [m^3]
        self.vr = 0      # regenerator void volume [m^3]
        self.vh = 0      # heater void volume [m^3]

        # Gas properties
        self.rgas_mix = 0  # gas constant [J/kg.K]
        # specific heat capacity at constant pressure [J/kg.K]
        self.cp_mix = 0
        self.cv_mix = 0    # specific heat capacity at constant volume [J/kg.K]
        self.gama_mix = 0  # ratio: cp/cv
        self.mgas = 0      # total mass of gas in engine [kg]

        # Engine geometry
        self.vclc = 0  # compression clearance volume [m^3]
        self.vcle = 0  # expansion clearance volume [m^3]
        self.vswc = 0  # compression swept volume [m^3]
        self.vswe = 0  # expansion swept volume [m^3]
        self.alpha = 0  # phase angle advance of expansion space [radians]

        self.DEP_VAR_INDICES = [self.Index.TC, self.Index.TE, self.Index.QK,
                                self.Index.QR, self.Index.QH, self.Index.WC, self.Index.WE]

    def set_parameters(self, params: dict):
        """
        Set engine parameters from a dictionary.

        Args:
            params: Dictionary containing parameter names and values
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")

        # Calculate derived parameters
        self.gama_mix = self.cp_mix / self.cv_mix

    def adiab(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ideal adiabatic model simulation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                var: Array of variable values every 10 degrees (0 - 360)
                dvar: Array of derivatives every 10 degrees (0 - 360)
        """
        print('============Ideal Adiabatic Analysis====================')
        print(f'Cooler Tk = {self.tk:.1f}[K], Heater Th = {self.th:.1f}[K]')

        EPSILON = 1  # Allowable error in temperature (K)
        MAX_ITERATIONS = 20
        INCREMENTS = 360
        STEP = INCREMENTS // 36
        DTHETA = 2.0 * np.pi / INCREMENTS

        var = np.zeros((22, 37))
        dvar = np.zeros((16, 37))
        y = np.zeros(22)

        y[self.Index.THE] = self.th
        y[self.Index.TCK] = self.tk
        y[self.Index.TE] = self.th
        y[self.Index.TC] = self.tk

        iter_count = 0
        temp_error = 10 * EPSILON

        while temp_error >= EPSILON and iter_count < MAX_ITERATIONS:
            tc0, te0 = y[self.Index.TC], y[self.Index.TE]
            theta = 0
            y[self.Index.QK:self.Index.W + 1] = 0

            print(f'iteration {iter_count}: Tc = {
                  y[self.Index.TC]:.1f}[K], Te = {y[self.Index.TE]:.1f}[K]')

            for _ in range(INCREMENTS):
                theta, y, dy = self.rk4(self.dadiab, theta, DTHETA, y)

            temp_error = abs(tc0 - y[self.Index.TC]) + \
                abs(te0 - y[self.Index.TE])
            iter_count += 1

        if iter_count >= MAX_ITERATIONS:
            print(f'No convergence within {MAX_ITERATIONS} iterations')

        theta = 0
        y[self.Index.QK:self.Index.W + 1] = 0
        var, dvar = self.fill_matrix(0, y, dy, var, dvar)

        for i in range(1, 37):
            for _ in range(STEP):
                theta, y, dy = self.rk4(self.dadiab, theta, DTHETA, y)
            var, dvar = self.fill_matrix(i, y, dy, var, dvar)

        return var, dvar

    def adiabatic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ideal adiabatic simulation and temperature/energy vs theta plots.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                var: Array of variable values every 10 degrees (0 - 360)
                dvar: Array of derivatives every 10 degrees (0 - 360)
        """
        var, dvar = self.adiab()

        eff = var[self.Index.W, -1] / var[self.Index.QH, -1]
        qk_power = var[self.Index.QK, -1] * self.freq
        qr_power = var[self.Index.QR, -1] * self.freq
        qh_power = var[self.Index.QH, -1] * self.freq
        w_power = var[self.Index.W, -1] * self.freq

        print('========== Ideal Adiabatic Analysis Results ============')
        print(f' Heat transferred to the cooler: {qk_power:.2f}[W]')
        print(f' Net heat transferred to the regenerator: {qr_power:.2f}[W]')
        print(f' Heat transferred to the heater: {qh_power:.2f}[W]')
        print(f' Total power output: {w_power:.2f}[W]')
        print(f' Thermal efficiency: {eff * 100:.1f}[%]')
        print('========================================================')

        self.plot_adiabatic(var, dvar)

        return var, dvar

    def dadiab(self, theta: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derivatives for ideal adiabatic model simulation.

        Args:
            theta (float): Crank angle (radians)
            y (np.ndarray): Current state variables

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                y (np.ndarray): Updated state variables
                dy (np.ndarray): Derivatives of the state variables
        """
        dy = np.zeros(16)

        y[self.Index.VC], y[self.Index.VE], dy[self.Index.VC], dy[self.Index.VE] = self.volume(
            theta)

        vot = self.vk/self.tk + self.vr/self.tr + self.vh/self.th
        y[self.Index.P] = (self.mgas * self.rgas_mix) / (y[self.Index.VC] /
                                                         y[self.Index.TC] + vot + y[self.Index.VE]/y[self.Index.TE])

        top = -y[self.Index.P] * (dy[self.Index.VC] /
                                  y[self.Index.TCK] + dy[self.Index.VE]/y[self.Index.THE])
        bottom = (y[self.Index.VC]/(y[self.Index.TCK]*self.gama_mix) +
                  vot + y[self.Index.VE]/(y[self.Index.THE]*self.gama_mix))
        dy[self.Index.P] = top / bottom

        # Mass accumulations and derivatives
        y[self.Index.MC] = y[self.Index.P] * \
            y[self.Index.VC] / (self.rgas_mix * y[self.Index.TC])
        y[self.Index.MK] = y[self.Index.P] * \
            self.vk / (self.rgas_mix * self.tk)
        y[self.Index.MR] = y[self.Index.P] * \
            self.vr / (self.rgas_mix * self.tr)
        y[self.Index.MH] = y[self.Index.P] * \
            self.vh / (self.rgas_mix * self.th)
        y[self.Index.ME] = y[self.Index.P] * \
            y[self.Index.VE] / (self.rgas_mix * y[self.Index.TE])

        dy[self.Index.MC] = (y[self.Index.P]*dy[self.Index.VC] + y[self.Index.VC]
                             * dy[self.Index.P]/self.gama_mix) / (self.rgas_mix*y[self.Index.TCK])
        dy[self.Index.ME] = (y[self.Index.P]*dy[self.Index.VE] + y[self.Index.VE]
                             * dy[self.Index.P]/self.gama_mix) / (self.rgas_mix*y[self.Index.THE])

        dpop = dy[self.Index.P] / y[self.Index.P]
        dy[self.Index.MK] = y[self.Index.MK] * dpop
        dy[self.Index.MR] = y[self.Index.MR] * dpop
        dy[self.Index.MH] = y[self.Index.MH] * dpop

        # Mass flow between cells
        y[self.Index.GACK] = -dy[self.Index.MC]
        y[self.Index.GAKR] = y[self.Index.GACK] - dy[self.Index.MK]
        y[self.Index.GAHE] = dy[self.Index.ME]
        y[self.Index.GARH] = y[self.Index.GAHE] + dy[self.Index.MH]

        # Conditional temperatures between cells
        y[self.Index.TCK] = self.tk if y[self.Index.GACK] <= 0 else y[self.Index.TC]
        y[self.Index.THE] = y[self.Index.TE] if y[self.Index.GAHE] <= 0 else self.th

        # Working space temperatures
        dy[self.Index.TC] = y[self.Index.TC] * \
            (dpop + dy[self.Index.VC]/y[self.Index.VC] -
             dy[self.Index.MC]/y[self.Index.MC])
        dy[self.Index.TE] = y[self.Index.TE] * \
            (dpop + dy[self.Index.VE]/y[self.Index.VE] -
             dy[self.Index.ME]/y[self.Index.ME])

        # Energy
        dy[self.Index.QK] = self.vk * dy[self.Index.P] * self.cv_mix / self.rgas_mix - \
            self.cp_mix * (y[self.Index.TCK] *
                           y[self.Index.GACK] - self.tk*y[self.Index.GAKR])
        dy[self.Index.QR] = self.vr * dy[self.Index.P] * self.cv_mix / self.rgas_mix - \
            self.cp_mix * (self.tk*y[self.Index.GAKR] -
                           self.th*y[self.Index.GARH])
        dy[self.Index.QH] = self.vh * dy[self.Index.P] * self.cv_mix / self.rgas_mix - \
            self.cp_mix * (self.th*y[self.Index.GARH] -
                           y[self.Index.THE]*y[self.Index.GAHE])
        dy[self.Index.WC] = y[self.Index.P] * dy[self.Index.VC]
        dy[self.Index.WE] = y[self.Index.P] * dy[self.Index.VE]

        # Net work done
        dy[self.Index.W] = dy[self.Index.WC] + dy[self.Index.WE]
        y[self.Index.W] = y[self.Index.WC] + y[self.Index.WE]

        return y, dy

    def fill_matrix(self, idx: int, y: np.ndarray, dy: np.ndarray, var: np.ndarray, dvar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fills the variable and derivative matrices with values.

        Args:
            idx (int): Index for the matrices
            y (np.ndarray): Current state variables
            dy (np.ndarray): Derivatives of the state variables
            var (np.ndarray): Variable matrix
            dvar (np.ndarray): Derivative matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                var (np.ndarray): Updated variable matrix
                dvar (np.ndarray): Updated derivative matrix
        """
        var[:, idx] = y
        dvar[:, idx] = dy
        return var, dvar

    def rk4(self, derivs: Callable, theta: float, dtheta: float, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Runge-Kutta 4th order method for ODE integration.

        Args:
            derivs (Callable): Function to compute derivatives
            theta (float): Current angle (radians)
            dtheta (float): Angle increment (radians)
            y (np.ndarray): Current state variables

        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                theta (float): Updated angle
                y (np.ndarray): Updated state variables
                dy (np.ndarray): Derivatives of the state variables
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

        dy[self.DEP_VAR_INDICES] = (dy1[self.DEP_VAR_INDICES] + 2*(dy2[self.DEP_VAR_INDICES] +
                                                                   dy3[self.DEP_VAR_INDICES]) + dy[self.DEP_VAR_INDICES]) / 6
        y[self.DEP_VAR_INDICES] = y0[self.DEP_VAR_INDICES] + \
            dtheta * dy[self.DEP_VAR_INDICES]

        return theta, y, dy

    def volume(self, theta: float) -> Tuple[float, float, float, float]:
        """Calculate volumes for sinusoidal motion."""
        vc = self.vclc + 0.5 * self.vswc * (1 + np.cos(theta + np.pi))
        ve = self.vcle + 0.5 * self.vswe * \
            (1 + np.cos(theta + self.alpha + np.pi))
        dvc = -0.5 * self.vswc * np.sin(theta + np.pi)
        dve = -0.5 * self.vswe * np.sin(theta + self.alpha + np.pi)
        return vc, ve, dvc, dve

    def plot_adiabatic(self, var: np.ndarray, dvar: np.ndarray):
        """
        Plot temperature and energy vs theta.

        Args:
            var: Variable matrix
            dvar: Derivative matrix
        """
        vol = (var[self.Index.VC, :] + self.vk +
               # Volume in cubic centimeters...
               self.vr + self.vh + var[self.Index.VE, :])*1e6
        pres = var[self.Index.P]*1e-5  # Pressure in bar
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.plot(vol, pres)
        plt.xlabel("$Volume\ (cm^3)$")
        plt.ylabel("$Pressure\ (bar)$")
        plt.grid(True)
        plt.title("$P-V\ diagram$")
        plt.show()

    def run_simulation(self):
        """Run the Stirling engine simulation."""
        var, dvar = self.adiabatic()
        self.plot_adiabatic(var, dvar)
        return var, dvar


if __name__ == "__main__":
    engine = StirlingEngine()

    th, tk = 977, 288

    # Define engine parameters
    params = {
        # Basic engine parameters
        'freq': 41.7,  # Hz
        'tk': tk,   # K
        'tr': (th-tk)/np.log(th/tk),   # K
        'th': th,   # K
        'vk': 4.783800000000000e-04,  # m^3
        'vr': 1.560600000000000e-04,  # m^3
        'vh': 4.783800000000000e-04,  # m^3

        'rgas_mix': 1.899440000000000e+03,  # J/kg.K
        'cp_mix': 4.763339253731343e+03,    # J/kg.K
        'cv_mix': 2.863899253731343e+03,    # J/kg.K
        'mgas': 100,     # kg

        # Engine geometry
        'vclc': 2.868000000000000e-05,  # m^3
        'vcle': 3.052000000000000e-05,  # m^3
        'vswc': 5.246000000000000e-04,   # m^3
        'vswe': 5.246000000000000e-04,   # m^3
        'alpha': np.pi/2,  # radians
    }

    # Set the parameters
    engine.set_parameters(params)

    _, __ = engine.run_simulation()
