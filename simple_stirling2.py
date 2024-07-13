import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from dataclasses import dataclass
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

    @dataclass
    class Parameters:
        freq: float = 0
        tk: float = 0
        tr: float = 0
        th: float = 0
        vk: float = 0
        vr: float = 0
        vh: float = 0
        inside_dia_h_tube: float = 0
        no_h_tubes: int = 0
        len_h_tube: float = 0
        inside_dia_k_tube: float = 0
        no_k_tubes: int = 0
        len_k_tube: float = 0
        no_r_tubes: int = 0
        reg_porosity: float = 0
        reg_len: float = 0
        dwire: float = 0
        domat: float = 0
        dout: float = 0
        mgas: float = 0
        gamma: float = 0
        rgas: float = 0
        mu0: float = 0
        t_suth: float = 0
        t0: float = 0
        prandtl: float = 0
        vclc: float = 0
        vcle: float = 0
        vswc: float = 0
        vswe: float = 0
        alpha: float = 0

    def __init__(self, params: Parameters):
        self.params = params
        self.params.tr = (self.params.th - self.params.tk) / \
            (np.log(self.params.th/self.params.tk))
        self.omega = 2 * np.pi * self.params.freq
        self.cv = self.params.rgas / (self.params.gamma - 1)
        self.cp = self.params.gamma * self.cv

        self.vh, self.ah, self.awgh, self.dh, self.lh = self.pipes(
            self.params.no_h_tubes, self.params.inside_dia_h_tube, self.params.len_h_tube
        )
        self.vk, self.ak, self.awgk, self.dk, self.lk = self.pipes(
            self.params.no_k_tubes, self.params.inside_dia_k_tube, self.params.len_k_tube
        )
        self.vr, self.ar, self.awgr, self.dr, self.lr = self.regen(
            self.params.no_r_tubes, self.params.reg_len, self.params.dwire
        )

        self.DEP_VAR_INDICES = [
            self.Index.TC, self.Index.TE, self.Index.QK,
            self.Index.QR, self.Index.QH, self.Index.WC, self.Index.WE
        ]

    def pipes(self, num: int, inner_dia: float, pipe_len: float) -> Tuple[float, float, float, float, float]:
        a = (num * np.pi * inner_dia**2) / 4
        v = a * pipe_len
        awg = num * np.pi * inner_dia * pipe_len
        return v, a, awg, inner_dia, pipe_len

    def regen(self, num: int, lr: float, dwire: float) -> Tuple[float, float, float, float, float]:
        dimat = 0
        awgr0 = num * np.pi * self.params.domat * lr
        amat = num * np.pi * (self.params.domat**2 - dimat**2) / 4
        ar = amat * self.params.reg_porosity
        vr = ar * lr
        dr = dwire * self.params.reg_porosity / (1 - self.params.reg_porosity)
        awgr = 4 * vr / dr + awgr0
        return vr, ar, awgr, dr, lr

    def adiabatic(self) -> Tuple[np.ndarray, np.ndarray]:
        print('============Ideal Adiabatic Analysis====================')
        print(f'Cooler Tk = {self.params.tk:.1f}[K], Heater Th = {
              self.params.th:.1f}[K]')

        EPSILON = 1
        MAX_ITERATIONS = 20
        INCREMENTS = 360
        STEP = INCREMENTS // 36
        DTHETA = 2.0 * np.pi / INCREMENTS

        var = np.zeros((22, 37))
        dvar = np.zeros((16, 37))
        y = np.zeros(22)

        y[self.Index.THE] = self.params.th
        y[self.Index.TCK] = self.params.tk
        y[self.Index.TE] = self.params.th
        y[self.Index.TC] = self.params.tk

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

        self.print_results(var)
        return var, dvar

    def dadiab(self, theta: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dy = np.zeros(16)

        y[self.Index.VC], y[self.Index.VE], dy[self.Index.VC], dy[self.Index.VE] = self.volume(
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

        # Mass accumulations and derivatives
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

        # Mass flow between cells
        y[self.Index.GACK] = -dy[self.Index.MC]
        y[self.Index.GAKR] = y[self.Index.GACK] - dy[self.Index.MK]
        y[self.Index.GAHE] = dy[self.Index.ME]
        y[self.Index.GARH] = y[self.Index.GAHE] + dy[self.Index.MH]

        # Conditional temperatures between cells
        y[self.Index.TCK] = self.params.tk if y[self.Index.GACK] <= 0 else y[self.Index.TC]
        y[self.Index.THE] = y[self.Index.TE] if y[self.Index.GAHE] <= 0 else self.params.th

        # Working space temperatures
        dy[self.Index.TC] = y[self.Index.TC] * \
            (dpop + dy[self.Index.VC] / y[self.Index.VC] -
             dy[self.Index.MC] / y[self.Index.MC])
        dy[self.Index.TE] = y[self.Index.TE] * \
            (dpop + dy[self.Index.VE] / y[self.Index.VE] -
             dy[self.Index.ME] / y[self.Index.ME])

        # Energy
        dy[self.Index.QK] = self.vk * dy[self.Index.P] * self.cv / self.params.rgas - self.cp * \
            (y[self.Index.TCK] * y[self.Index.GACK] -
             self.params.tk * y[self.Index.GAKR])
        dy[self.Index.QR] = self.vr * dy[self.Index.P] * self.cv / self.params.rgas - \
            self.cp * (self.params.tk *
                       y[self.Index.GAKR] - self.params.th * y[self.Index.GARH])
        dy[self.Index.QH] = self.vh * dy[self.Index.P] * self.cv / self.params.rgas - self.cp * \
            (self.params.th * y[self.Index.GARH] -
             y[self.Index.THE] * y[self.Index.GAHE])
        dy[self.Index.WC] = y[self.Index.P] * dy[self.Index.VC]
        dy[self.Index.WE] = y[self.Index.P] * dy[self.Index.VE]

        # Net work done
        dy[self.Index.W] = dy[self.Index.WC] + dy[self.Index.WE]
        y[self.Index.W] = y[self.Index.WC] + y[self.Index.WE]

        return y, dy

    def fill_matrix(self, idx: int, y: np.ndarray, dy: np.ndarray, var: np.ndarray, dvar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        var[:, idx] = y
        dvar[:, idx] = dy
        return var, dvar

    def rk4(self, derivs: Callable, theta: float, dtheta: float, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
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
        vc = self.params.vclc + 0.5 * \
            self.params.vswc * (1 + np.cos(theta + np.pi))
        ve = self.params.vcle + 0.5 * self.params.vswe * \
            (1 + np.cos(theta + self.params.alpha + np.pi))
        dvc = -0.5 * self.params.vswc * np.sin(theta + np.pi)
        dve = -0.5 * self.params.vswe * \
            np.sin(theta + self.params.alpha + np.pi)
        return vc, ve, dvc, dve

    def print_results(self, var: np.ndarray):
        """Print the simulation results."""
        eff = var[self.Index.W, -1] / var[self.Index.QH, -1]
        qk_power = var[self.Index.QK, -1] * self.params.freq
        qr_power = var[self.Index.QR, -1] * self.params.freq
        qh_power = var[self.Index.QH, -1] * self.params.freq
        w_power = var[self.Index.W, -1] * self.params.freq

        print('========== Ideal Adiabatic Analysis Results ============')
        print(f' Heat transferred to the cooler: {qk_power:.2f}[W]')
        print(f' Net heat transferred to the regenerator: {qr_power:.2f}[W]')
        print(f' Heat transferred to the heater: {qh_power:.2f}[W]')
        print(f' Total power output: {w_power:.2f}[W]')
        print(f' Thermal efficiency: {eff * 100:.1f}[%]')
        print('========================================================')

    def plot_pv_diagram(self, var: np.ndarray):
        """Plot the P-V diagram."""
        vol = (var[self.Index.VC, :] + self.vk + self.vr + self.vh +
               var[self.Index.VE, :]) * 1e6  # Volume in cubic centimeters
        pres = var[self.Index.P] * 1e-5  # Pressure in bar

        plt.figure(figsize=(10, 6))
        plt.plot(vol, pres)
        plt.xlabel("Volume $(cm^3)$")
        plt.ylabel("Pressure (bar)")
        plt.title("P-V Diagram")
        plt.grid(True)
        plt.show()

    def run_simulation(self):
        """Run the Stirling engine simulation."""
        twk = self.params.tk  # Cooler wall temp - equal to initial cooler gas temp
        twh = self.params.th  # Heater wall temp - equal to initial heater gas temp
        epsilon = 1  # allowable temperature error bound for cyclic convergence
        terror = 10 * epsilon  # Initial temperature error (to enter loop)

        while terror > epsilon:
            var, _ = self.adiabatic()
            qrloss = self.get_regenerator_heat_loss(var)
            tgh = self.hotsim(var, twh, qrloss)  # new heater gas temperature
            tgk = self.kolsim(var, twk, qrloss)  # new cooler gas temperature
            terror = abs(self.params.th - tgh) + abs(self.params.tk - tgk)
            self.params.th = tgh
            self.params.tk = tgk
            self.params.tr = (self.params.th - self.params.tk) / \
                np.log(self.params.th / self.params.tk)

        var, _ = self.adiabatic()
        self.plot_pv_diagram(var)
        return var

    def get_regenerator_heat_loss(self, var: np.ndarray) -> float:
        """Calculate the regenerator heat loss."""
        re = np.zeros(37)
        for i in range(37):
            gar = (var[self.Index.GAKR, i] +
                   var[self.Index.GARH, i]) * self.omega / 2
            gr = gar / self.ar
            _, _, re[i] = self.reynolds(self.params.tr, gr, self.dr)

        reavg = np.mean(re)
        st, _ = self.matrixfr(reavg)
        ntu = st * self.awgr / (2 * self.ar)
        effect = ntu / (ntu + 1)

        qrmin = np.min(var[self.Index.QR])
        qrmax = np.max(var[self.Index.QR])
        qrloss = (1 - effect) * (qrmax - qrmin)

        return qrloss

    def reynolds(self, t: float, g: float, d: float) -> Tuple[float, float, float]:
        """Evaluate dynamic viscosity, thermal conductivity, Reynolds number."""
        mu = self.params.mu0 * (self.params.t0 + self.params.t_suth) / \
            (t + self.params.t_suth) * (t / self.params.t0)**1.5
        kgas = self.cp * mu / self.params.prandtl
        re = max(1, abs(g) * d / mu)
        return mu, kgas, re

    def matrixfr(self, re: float) -> Tuple[float, float]:
        """Evaluate regenerator mesh matrix Stanton number, friction factor."""
        st = 0.46 * re**(-0.4) / self.params.prandtl
        fr = 54 + 1.43 * re**0.78
        return st, fr

    def hotsim(self, var: np.ndarray, twh: float, qrloss: float) -> float:
        """Evaluate heater average heat transfer performance."""
        re = np.zeros(37)
        for i in range(37):
            gah = (var[self.Index.GARH, i] +
                   var[self.Index.GAHE, i]) * self.omega / 2
            gh = gah / self.ah
            mu, _, re[i] = self.reynolds(self.params.th, gh, self.dh)

        reavg = np.mean(re)
        ht, _ = self.pipefr(self.dh, mu, reavg)
        tgh = twh - (var[self.Index.QH, -1] + qrloss) * \
            self.params.freq / (ht * self.awgh)
        return tgh

    def kolsim(self, var: np.ndarray, twk: float, qrloss: float) -> float:
        """Evaluate cooler average heat transfer performance."""
        re = np.zeros(37)
        for i in range(37):
            gak = (var[self.Index.GACK, i] +
                   var[self.Index.GAKR, i]) * self.omega / 2
            gk = gak / self.ak
            mu, _, re[i] = self.reynolds(self.params.tk, gk, self.dk)

        reavg = np.mean(re)
        ht, _ = self.pipefr(self.dk, mu, reavg)
        tgk = twk - (var[self.Index.QK, -1] - qrloss) * \
            self.params.freq / (ht * self.awgk)
        return tgk

    def pipefr(self, d: float, mu: float, re: float) -> Tuple[float, float]:
        """Evaluate heat transfer coefficient, Reynolds friction factor."""
        fr = 0.0791 * re**0.75
        ht = fr * mu * self.cp / (2 * d * self.params.prandtl)
        return ht, fr


def main():
    # Define engine parameters
    params = StirlingEngine.Parameters(
        freq=41.7,
        tk=288,
        th=977,
        inside_dia_h_tube=0.302e-2,
        no_h_tubes=40,
        len_h_tube=24.53e-2,
        inside_dia_k_tube=0.108e-2,
        no_k_tubes=312,
        len_k_tube=4.6e-2,
        no_r_tubes=8,
        reg_porosity=0.697,
        reg_len=2.26e-2,
        dwire=0.004e-2,
        domat=2.26e-2,
        dout=3e-2,
        mgas=0.00005,
        gamma=1.67,
        rgas=2078.6,
        mu0=18.85e-6,
        t_suth=80.0,
        t0=273,
        prandtl=0.71,
        vclc=2.868e-5,
        vcle=3.052e-5,
        vswc=5.246e-4,
        vswe=5.246e-4,
        alpha=np.pi/2,
    )

    # Create StirlingEngine instance and run simulation
    engine = StirlingEngine(params)
    engine.run_simulation()


if __name__ == "__main__":
    main()
