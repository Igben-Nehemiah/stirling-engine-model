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
        self.omega = 0   # cycle frequency [rad/s]
        self.tk = 0      # cooler temperature [K]
        self.tr = 0      # regenerator temperature [K]
        self.th = 0      # heater temperature [K]
        self.vk = 0      # cooler void volume [m^3]
        self.vr = 0      # regenerator void volume [m^3]
        self.vh = 0      # heater void volume [m^3]

        # Engine geometry
        self.vclc = 0  # compression clearance volume [m^3]
        self.vcle = 0  # expansion clearance volume [m^3]
        self.vswc = 0  # compression swept volume [m^3]
        self.vswe = 0  # expansion swept volume [m^3]
        self.alpha = 0  # phase angle advance of expansion space [radians]

        # Heater
        self.vh, self.ah, self.awgh, self.dh, self.lh = 0, 0, 0, 0, 0
        self.inner_dia_htube = 0
        self.num_htubes = 0
        self.length_htube = 0
        # Cooler
        self.vk, self.ak, self.awgk, self.dk, self.lk = 0, 0, 0, 0, 0
        self.inner_dia_ktube = 0
        self.num_ktubes = 0
        self.length_ktube = 0
        # Regenerator
        self.vr, self.ar, self.awgr, self.lr = 0, 0, 0, 0
        self.porosity = 0
        self.dout = 0  # Tube housing external diameter [m]
        self.domat = 0  # Tube housing internal diameter [m]
        self.reg_len = 0  # Regenerator length [m]
        self.num_rtubes = 0  # Number of tubes
        self.dwire = 0

        # Gas properties (Helium)
        self.mgas = 0      # total mass of gas in engine [kg]
        self.gamma = 0  # ratio: cp/cv
        self.rgas = 0  # Gas constant [J/kg.K]
        self.mu0 = 0  # Dynamic viscosity at reference temp t0 [kg.m/s]
        self.t_suth = 0  # Sutherland constant [K]
        self.t0 = 0  # Reference temperature [K]
        self.prandtl = 0  # Prandtl number
        self.cp = 0  # specific heat capacity at constant pressure [J/kg.K]
        self.cv = 0  # specific heat capacity at constant volume [J/kg.K]

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
        self.cv = self.rgas/(self.gamma - 1)
        self.cp = self.gamma*self.cv
        self.omega = self.freq*2*np.pi

        self.vh, self.ah, self.awgh, self.dh, self.lh = self.pipes(self.num_htubes, self.inner_dia_htube,
                                                                   self.length_htube)
        self.vk, self.ak, self.awgk, self.dk, self.lk = self.pipes(self.num_ktubes, self.inner_dia_ktube,
                                                                   self.length_ktube)

        self.vr, self.ar, self.awgr, self.dr, self.lr = self.regen(
            self.num_rtubes, self.reg_len, self.dwire)

    def pipes(self, num, inner_dia, pipe_len):
        """
        Homogeneous smooth pipes heat exchanger 
        v: Void volume [m^3]
        a: Internal free flow area [m^2]
        awg: Internal wetted area [m^2]
        d: Hydraulic diameter [m]
        l: Effective length [m]
        """
        a = (num*np.pi*inner_dia**2)/4
        v = a*pipe_len
        awg = num*np.pi*inner_dia*pipe_len
        return v, a, awg, inner_dia, pipe_len

    def regen(self, num, lr, dwire):
        dimat = 0
        awgr0 = num*np.pi*self.domat*lr
        amat = num*np.pi*(self.domat**2 - dimat**2)/4  # regen matrix area
        awr = num*np.pi*(self.dout**2 - self.domat**2) / \
            4  # regen housing wall area

        # # temporary fix (4/20/02):
        # kwr = 25  # thermal conductivity [W/m/K]
        # # note that stainless steel thermal conductivity is temp dependent
        # #   25 W/m/K for normal engine conditions,
        # #    6 W/m/K for cryogenic coolers.
        # cqwr = kwr*awr/lr  # regen wall thermal conductance[W/K]

        # Simple mesh
        ar = amat*self.porosity
        vr = ar*lr
        dr = dwire*self.porosity/(1 - self.porosity)
        awgr = 4*vr/dr + awgr0
        return vr, ar, awgr, dr, lr

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

        # self.plot_adiabatic(var, dvar)

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
        y[self.Index.P] = (self.mgas * self.rgas) / (y[self.Index.VC] /
                                                     y[self.Index.TC] + vot + y[self.Index.VE]/y[self.Index.TE])

        top = -y[self.Index.P] * (dy[self.Index.VC] /
                                  y[self.Index.TCK] + dy[self.Index.VE]/y[self.Index.THE])
        bottom = (y[self.Index.VC]/(y[self.Index.TCK]*self.gamma) +
                  vot + y[self.Index.VE]/(y[self.Index.THE]*self.gamma))
        dy[self.Index.P] = top / bottom

        # Mass accumulations and derivatives
        y[self.Index.MC] = y[self.Index.P] * \
            y[self.Index.VC] / (self.rgas * y[self.Index.TC])
        y[self.Index.MK] = y[self.Index.P] * \
            self.vk / (self.rgas * self.tk)
        y[self.Index.MR] = y[self.Index.P] * \
            self.vr / (self.rgas * self.tr)
        y[self.Index.MH] = y[self.Index.P] * \
            self.vh / (self.rgas * self.th)
        y[self.Index.ME] = y[self.Index.P] * \
            y[self.Index.VE] / (self.rgas * y[self.Index.TE])

        dy[self.Index.MC] = (y[self.Index.P]*dy[self.Index.VC] + y[self.Index.VC]
                             * dy[self.Index.P]/self.gamma) / (self.rgas*y[self.Index.TCK])
        dy[self.Index.ME] = (y[self.Index.P]*dy[self.Index.VE] + y[self.Index.VE]
                             * dy[self.Index.P]/self.gamma) / (self.rgas*y[self.Index.THE])

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
        dy[self.Index.QK] = self.vk * dy[self.Index.P] * self.cv / self.rgas - \
            self.cp * (y[self.Index.TCK] *
                       y[self.Index.GACK] - self.tk*y[self.Index.GAKR])
        dy[self.Index.QR] = self.vr * dy[self.Index.P] * self.cv / self.rgas - \
            self.cp * (self.tk*y[self.Index.GAKR] -
                       self.th*y[self.Index.GARH])
        dy[self.Index.QH] = self.vh * dy[self.Index.P] * self.cv / self.rgas - \
            self.cp * (self.th*y[self.Index.GARH] -
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

        twk = self.tk  # Cooler wall temp - equal to initial cooler gas temp
        twh = self.th  # Heater wall temp - equal to initial heater gas temp
        epsilon = 1  # allowable temperature error bound for cyclic convergence
        terror = 10*epsilon  # Initial temperature error (to enter loop)

        while (terror > epsilon):
            var, dvar = self.adiabatic()
            qrloss = self.get_regenerator_heat_loss(var)
            # new heater gas temperature
            tgh = self.hotsim(var, twh, qrloss)
            # new cooler gas temperature
            tgk = self.kolsim(var, twk, qrloss)
            terror = abs(self.th - tgh) + abs(self.tk - tgk)
            self.th = tgh
            self.tk = tgk
            self.tr = (self.th-self.tk)/np.log(self.th/self.tk)

        var, dvar = self.adiabatic()
        self.plot_adiabatic(var, dvar)
        return var, dvar

    def get_regenerator_heat_loss(self, var):
        # Reynolds number over the cycle
        re = np.zeros((37,))
        for i in range(37):
            gar = (var[self.Index.GAKR, i] +
                   var[self.Index.GARH, i])*self.omega/2
            gr = gar/self.ar
            _, __, re[i] = self.reynolds(self.tr, gr, self.dr)

        # average and maximum Reynolds number
        reavg = np.average(re)

        # Stanton number, number of transfer units, regenerator effectiveness
        st, _ = self.matrixfr(reavg)
        ntu = st*self.awgr/(2*self.ar)
        effect = ntu/(ntu + 1)

        qrmin = np.min(var[self.Index.QR])
        qrmax = np.max(var[self.Index.QR])
        qrloss = (1 - effect)*(qrmax - qrmin)

        return qrloss

    def reynolds(self, t, g, d):
        """evaluate dynamic viscosity, thermal conductivity, Reynolds number
        Arguments:
          t - gas temperature [K]
          g - mass flux [kg/m^2.s]
          d - hydraulic diameter [m]
        Returned values: 
          mu - gas dynamic viscosity [kg/m.s]
          kgas - gas thermal conductivity [W/m.K]
          re - Reynolds number"""
        mu = self.mu0*(self.t0 + self.t_suth) / \
            (t + self.t_suth)*(t/self.t0)**1.5
        kgas = self.cp*mu/self.prandtl

        re = abs(g)*d/mu
        re = max(1, re)
        return mu, kgas, re

    def matrixfr(self, re):
        """Evaluate regenerator mesh matrix stanton number, friction factor
        Israel Urieli, 7/22/2002
        Arguments:
          re - Reynolds number
        Returned values: 
          st - Stanton number
          fr - Reynolds friction factor ( = re*fanning friction factor)"""

        # equations taken from Kays & London (1955 edition)
        st = 0.46*re**(-0.4)/self.prandtl
        fr = 54 + 1.43*re**0.78
        return st, fr

    def hotsim(self, var, twh, qrloss):
        """Evaluate heater average heat transfer performance
        Arguments:
          var(22,37) array of variable values every 10 degrees (0 - 360)
          twh - heater wall temperature [K]
          qrloss - heat loss due to imperfect regenerator [J]
        Returned values: 
          tgh - heater average gas temperature [K]"""

        re = np.zeros((37,))
        for i in range(37):
            gah = (var[self.Index.GARH, i] +
                   var[self.Index.GAHE, i])*self.omega/2
            gh = gah/self.ah
            mu, _, re[i] = self.reynolds(self.th, gh, self.dh)

        reavg = np.average(re)

        ht, _ = self.pipefr(self.dh, mu, reavg)
        # Heat transfer coefficient
        tgh = twh - (var[self.Index.QH, -1]+qrloss)*self.freq / \
            (ht*self.awgh)  # Heater gas temperature[K]

        return tgh

    def pipefr(self, d, mu, re):
        """Evaluate heat transfer coefficient, Reynolds friction factor
        Israel Urieli, 7/22/2002 (corrected header 2/20/2011)
        Arguments:
        d - hydraulic diameter [m]
        mu - gas dynamic viscosity [kg.m/s]
        re - Reynolds number
        Returned values: 
        ht - heat transfer coefficient [W/m^2.K]
        fr - Reynolds friction factor ( = re*fanning friction factor)"""

        # Personal communication with Alan Organ, because of oscillating
        # flow, we assume that flow is always turbulent. Use the Blasius
        # relation for all Reynolds numbers:
        fr = 0.0791*re**0.75
        # From Reynolds simple analogy:
        ht = fr*mu*self.cp/(2*d*self.prandtl)
        return ht, fr

    def kolsim(self, var, twk, qrloss):
        """Evaluate cooler average heat transfer performance
        Israel Urieli, 7/22/2002
        Modified 2/6/2010 to include regenerator qrloss 
        Arguments:
          var(22,37) array of variable values every 10 degrees (0 - 360)
          twk - cooler wall temperature [K]
          qrloss - heat loss due to imperfect regenerator [J]
        Returned values: 
          tgk - cooler average gas temperature [K]"""

        re = np.zeros((37,))
        for i in range(37):
            gak = (var[self.Index.GACK, i] +
                   var[self.Index.GAKR, i])*self.omega/2
            gk = gak/self.ak
            mu, _, re[i] = self.reynolds(self.tk, gk, self.dk)

        reavg = np.average(re)

        ht, _ = self.pipefr(self.dk, mu, reavg)
        # Heat transfer coefficient
        tgk = twk - (var[self.Index.QK, -1]-qrloss)*self.freq / \
            (ht*self.awgk)  # Heater gas temperature[K]

        return tgk


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

        "inner_dia_htube": 0.302e-2,
        "num_htubes": 40,
        "length_htube": 24.53e-2,
        "inner_dia_ktube": 0.108e-2,
        "num_ktubes": 312,
        "length_ktube": 4.6e-2,

        "num_rtubes": 8,
        "porosity": 0.697,
        "reg_len": 2.26e-2,
        "dwire": 0.004e-2,
        "domat": 2.26e-2,
        "dout": 3e-2,


        "mgas": 0.00005,      # total mass of gas in engine [kg]
        "gamma": 1.67,  # ratio: cp/cv
        "rgas": 2078.6,  # Gas constant [J/kg.K]
        "mu0": 18.85e-6,  # Dynamic viscosity at reference temp t0 [kg.m/s]
        "t_suth": 80.0,  # Sutherland constant [K]
        "t0": 273,  # Reference temperature [K]
        "prandtl": 0.71,  # Prandtl number

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
