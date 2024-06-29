import numpy as np
from typing import Tuple, Callable
from enum import IntEnum


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

        # Engine type and specific parameters
        self.engine_type = ''  # Engine type (sinusoidal, yoke, rocker, etc.)
        self.crank = 0   # crank radius [m]
        self.b1 = 0      # Ross yoke length (1/2 yoke base) [m]
        self.b2 = 0      # Ross yoke height [m]
        self.dcomp = 0   # diameter of compression piston [m]
        self.dexp = 0    # diameter of expansion piston [m]
        self.acomp = 0   # area of compression piston [m^2]
        self.aexp = 0    # area of expansion piston [m^2]
        self.ymin = 0    # minimum yoke vertical displacement [m]
        self.conrodc = 0  # length of compression piston connecting rod [m]
        self.conrode = 0  # length of expansion piston connecting rod [m]
        self.ycmax = 0   # maximum compression piston vertical displacement [m]
        self.yemax = 0   # maximum expansion piston vertical displacement [m]
        self.r = 0       # crank radius for beta configuration [m]

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
        self.acomp = np.pi * (self.dcomp/2)**2
        self.aexp = np.pi * (self.dexp/2)**2

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
        terror = 10 * EPSILON

        while terror >= EPSILON and iter_count < MAX_ITERATIONS:
            tc0, te0 = y[self.Index.TC], y[self.Index.TE]
            theta = 0
            y[self.Index.QK:self.Index.W+1] = 0

            print(f'iteration {iter_count}: Tc = {
                  y[self.Index.TC]:.1f}[K], Te = {y[self.Index.TE]:.1f}[K]')

            for _ in range(INCREMENTS):
                theta, y, dy = self.rk4(self.dadiab, 7, theta, DTHETA, y)

            terror = abs(tc0 - y[self.Index.TC]) + abs(te0 - y[self.Index.TE])
            iter_count += 1

        if iter_count >= MAX_ITERATIONS:
            print(f'No convergence within {MAX_ITERATIONS} iterations')

        theta = 0
        y[self.Index.QK:self.Index.W+1] = 0
        var, dvar = self.fill_matrix(0, y, dy, var, dvar)

        for i in range(1, 37):
            for _ in range(STEP):
                theta, y, dy = self.rk4(self.dadiab, 7, theta, DTHETA, y)
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

        print('========== ideal adiabatic analysis results ============')
        print(f' Heat transferred to the cooler: {qk_power:.2f}[W]')
        print(f' Net heat transferred to the regenerator: {qr_power:.2f}[W]')
        print(f' Heat transferred to the heater: {qh_power:.2f}[W]')
        print(f' Total power output: {w_power:.2f}[W]')
        print(f' Thermal efficiency : {eff*100:.1f}[%]')
        print('========================================================')

        # Plotting function would go here
        # self.plot_adiabatic(var, dvar)

        return var, dvar

    def dadiab(self, theta: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate ideal adiabatic model derivatives.

        Args:
            theta: Current cycle angle [radians]
            y: Vector of current variable values

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated y and dy vectors
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

    def rk4(self, deriv: Callable, n: int, x: float, dx: float, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Classical fourth order Runge-Kutta method.

        Args:
            deriv: Derivative function
            n: Number of first order differential equations
            x: Independent variable
            dx: Step size
            y: Dependent variable vector

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: Updated x, y, and dy
        """
        x0 = x
        y0 = y.copy()

        y, dy1 = deriv(x0, y)
        y_temp = y.copy()
        y_temp[:7] = y0[:7] + 0.5 * dx * dy1[:7]

        xm = x0 + 0.5 * dx
        y, dy2 = deriv(xm, y_temp)
        y_temp = y.copy()
        y_temp[:7] = y0[:7] + 0.5 * dx * dy2[:7]

        y, dy3 = deriv(xm, y_temp)
        y_temp = y.copy()
        y_temp[:7] = y0[:7] + dx * dy3[:7]

        x = x0 + dx
        y, dy = deriv(x, y_temp)

        dy[:7] = (dy1[:7] + 2*(dy2[:7] + dy3[:7]) + dy[:7]) / 6
        y[:7] = y0[:7] + dx * dy[:7]

        return x, y, dy

    def volume(self, theta: float) -> Tuple[float, float, float, float]:
        """
        Determine working space volume variations and derivatives.

        Args:
            theta: Current cycle angle [radians]

        Returns:
            Tuple[float, float, float, float]: vc, ve, dvc, dve
        """
        if self.engine_type.startswith('s'):
            return self.sine_vol(theta)
        elif self.engine_type.startswith('y'):
            return self.yoke_vol(theta)
        elif self.engine_type.startswith('r'):
            return self.rock_vol(theta)
        elif self.engine_type.startswith('b'):
            return self.sine_vol(theta)
        elif self.engine_type.startswith('m'):
            return self.sine_vol(theta)
        elif self.engine_type.startswith('n'):
            return self.gamma_vol(theta)
        elif self.engine_type.startswith('i'):
            return self.beta_vol(theta)
        else:
            raise ValueError(f"Unknown engine type: {self.engine_type}")

    def sine_vol(self, theta: float) -> Tuple[float, float, float, float]:
        """Calculate volumes for sinusoidal motion."""
        vc = self.vclc + 0.5 * self.vswc * (1 + np.cos(theta + np.pi))
        ve = self.vcle + 0.5 * self.vswe * \
            (1 + np.cos(theta + self.alpha + np.pi))
        dvc = -0.5 * self.vswc * np.sin(theta + np.pi)
        dve = -0.5 * self.vswe * np.sin(theta + self.alpha + np.pi)
        return vc, ve, dvc, dve

    def yoke_vol(self, theta: float) -> Tuple[float, float, float, float]:
        """Calculate volumes for yoke-drive motion."""
        sinth, costh = np.sin(theta), np.cos(theta)
        bth = np.sqrt(self.b1**2 - (self.crank * costh)**2)
        ye = self.crank * (sinth + (self.b2/self.b1) * costh) + bth
        yc = self.crank * (sinth - (self.b2/self.b1) * costh) + bth
        ve = self.vcle + self.aexp * (ye - self.ymin)
        vc = self.vclc + self.acomp * (yc - self.ymin)
        dvc = self.acomp * self.crank * \
            (costh + (self.b2/self.b1) * sinth + self.crank * sinth * costh / bth)
        dve = self.aexp * self.crank * \
            (costh - (self.b2/self.b1) * sinth + self.crank * sinth * costh / bth)
        return vc, ve, dvc, dve

    def rock_vol(self, theta: float) -> Tuple[float, float, float, float]:
        """Calculate volumes for rocker motion."""
        sinth, costh = np.sin(theta), np.cos(theta)
        beth = np.sqrt(self.conrode**2 - (self.crank * costh)**2)
        bcth = np.sqrt(self.conrodc**2 - (self.crank * sinth)**2)
        ye = beth - self.crank * sinth
        yc = bcth + self.crank * costh
        ve = self.vcle + self.aexp * (self.yemax - ye)
        vc = self.vclc + self.acomp * (self.ycmax - yc)
        dvc = self.acomp * self.crank * sinth * (self.crank * costh / bcth + 1)
        dve = -self.aexp * self.crank * costh * (self.crank * sinth / beth - 1)
        return vc, ve, dvc, dve

    def gamma_vol(self, theta: float) -> Tuple[float, float, float, float]:
        """Calculate volumes for gamma configuration."""
        landac, landae = 3.75/14.6, 3.75/13
        b1 = (self.vswe/2) * (1 + np.cos(theta) - (1/landae) *
                              (1 - np.sqrt(1 - (landae**2) * (np.sin(theta)**2))))
        b2 = (self.vswc/2) * (1 - np.cos(theta - np.pi/2) + (1/landac) *
                              (1 - np.sqrt(1 - (landac**2) * (np.sin(theta - np.pi/2)**2))))
        b3 = (self.vswe/2) * (1 - np.cos(theta) + (1/landae) *
                              (1 - np.sqrt(1 - (landae**2) * (np.sin(theta)**2))))
        vc = self.vclc + b1 + b2
        ve = self.vcle + b3
        dve = (self.vswe/2) * (np.sin(theta) + (landae * np.cos(theta)
                                                * np.sin(theta)) / np.sqrt(1 - landae**2 * np.sin(theta)**2))
        dvc = (self.vswc/2) * (np.sin(theta - np.pi/2) + (landac * np.cos(theta - np.pi/2) *
                                                          np.sin(theta - np.pi/2)) / np.sqrt(1 - landac**2 * np.sin(theta - np.pi/2)**2)) - dve
        return vc, ve, dvc, dve

    def beta_vol(self, theta: float) -> Tuple[float, float, float, float]:
        """Calculate volumes for beta configuration."""
        L, e = 4.6e-2, 2.08e-2
        Ad = Ap = np.pi * (6.96**2) / 4 * 1e-4
        b1 = np.sqrt(L**2 - (e - self.r)**2)
        b2 = np.sqrt((L - self.r)**2 - e**2)
        b = np.sqrt(L**2 - (e + self.r * np.cos(theta))**2)
        vc = self.vclc + (2 * Ap * (b1 - b))
        ve = self.vcle + (Ad * (b - b2 - self.r * np.sin(theta)))
        dvc = -(2 * Ap * self.r * np.sin(theta) * (e + self.r * np.cos(theta))
                ) / np.sqrt(L**2 - (e + self.r * np.cos(theta))**2)
        dve = -Ad * (self.r * np.cos(theta) - (self.r * np.sin(theta) * (e + self.r *
                     np.cos(theta))) / np.sqrt(L**2 - (e + self.r * np.cos(theta))**2))
        return vc, ve, dvc, dve

    def fill_matrix(self, j: int, y: np.ndarray, dy: np.ndarray, var: np.ndarray, dvar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill the j-th column of the var and dvar matrices with values of y and dy.

        Args:
            j: Column index (0 - 36, every 10 degrees of cycle angle)
            y: Vector of current variable values
            dy: Vector of current derivatives
            var: Matrix of current variables vs cycle angle
            dvar: Matrix of current derivatives vs cycle angle

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated var and dvar matrices
        """
        ROWV = 22  # number of rows in the var matrix
        ROWD = 16  # number of rows in the dvar matrix

        var[:ROWV, j] = y[:ROWV]
        dvar[:ROWD, j] = dy[:ROWD]

        return var, dvar

    def plot_adiabatic(self, var: np.ndarray, dvar: np.ndarray):
        """
        Plot temperature and energy vs theta.

        Args:
            var: Variable matrix
            dvar: Derivative matrix
        """
        # Implement plotting logic here
        # This would typically use a library like matplotlib
        pass

    def run_simulation(self):
        """Run the Stirling engine simulation."""
        var, dvar = self.adiabatic()
        self.plot_adiabatic(var, dvar)
        return var, dvar


if __name__ == "__main__":
    engine = StirlingEngine()

    # Define engine parameters
    params = {
        # Basic engine parameters
        'freq': 41.7,  # Hz
        'tk': 288,   # K
        'tr': (977-288)/np.log(977/288),   # K
        'th': 977,   # K
        'vk': 4.783800000000000e-04,  # m^3
        'vr': 1.560600000000000e-04,  # m^3
        'vh': 4.783800000000000e-04,  # m^3

        # Gas properties (example values for helium)
        'rgas_mix': 2077,  # J/kg.K
        'cp_mix': 5193,    # J/kg.K
        'cv_mix': 3116,    # J/kg.K
        'mgas': 0.001,     # kg

        # Engine geometry
        'vclc': 2.868000000000000e-05,  # m^3
        'vcle': 3.052000000000000e-05,  # m^3
        'vswc': 5.246000000000000e-04,   # m^3
        'vswe': 5.246000000000000e-04,   # m^3
        'alpha': np.pi/2,  # radians

        # Engine type and specific parameters
        'engine_type': 'sinusoidal',
        'crank': 0.05,  # m
        'b1': 0.1,      # m
        'b2': 0.05,     # m
        'dcomp': 0.08,  # m
        'dexp': 0.08,   # m
        'ymin': 0,      # m
        'conrodc': 0.15,  # m
        'conrode': 0.15,  # m
        'ycmax': 0.1,     # m
        'yemax': 0.1,     # m
        'r': 0.0138,        # m
    }

    # Set the parameters
    engine.set_parameters(params)

    # Run the simulation
    var, dvar = engine.run_simulation()
