import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Global variables (as a dictionary for better manageability)
globals_dict = {
    'freq': 1.0,  # Example frequency in Hz
    'cooler_temp': 300.0,  # Example cooler temperature in K
    'regenerator_temp': 350.0,  # Example regenerator temperature in K
    'heater_temp': 400.0,  # Example heater temperature in K
    'cooler_void_volume': 0.001,  # Example cooler void volume in m^3
    'regenerator_void_volume': 0.001,  # Example regenerator void volume in m^3
    'heater_void_volume': 0.001   # Example heater void volume in m^3
}


class AdiabaticSimulation:
    def __init__(self):
        self.num_variables = 22
        self.num_derivatives = 16
        self.num_steps = 37
        self.variables = np.zeros((self.num_variables, self.num_steps))
        self.derivatives = np.zeros((self.num_derivatives, self.num_steps))

    def run_simulation(self):
        epsilon = 1e-6
        max_iterations = 20
        num_increments = 360
        step = num_increments // 36
        dtheta = 2.0 * np.pi / num_increments
        y = np.zeros(22)
        y[17] = globals_dict['heater_temp']
        y[18] = globals_dict['heater_temp']
        y[1] = globals_dict['cooler_temp']
        y[2] = globals_dict['cooler_temp']

        iteration = 0
        temperature_error = 10 * epsilon

        while temperature_error >= epsilon and iteration < max_iterations:
            initial_cooler_temp = y[1]
            initial_heater_temp = y[2]
            theta = 0
            y[3:8] = 0
            print(f"Iteration {iteration}: Cooler Temp = {
                  y[1]:.1f}[K], Heater Temp = {y[2]:.1f}[K]")

            for _ in range(num_increments):
                theta, y, dy = self.rk4(
                    self.adiabatic_derivatives, 7, theta, dtheta, y)

            temperature_error = abs(
                initial_cooler_temp - y[1]) + abs(initial_heater_temp - y[2])
            iteration += 1

        if iteration >= max_iterations:
            print(f"No convergence within {max_iterations} iterations")

        theta = 0
        y[3:8] = 0
        self.store_results(0, y, dy)

        for i in range(1, self.num_steps):
            for _ in range(step):
                theta, y, dy = self.rk4(
                    self.adiabatic_derivatives, 7, theta, dtheta, y)
            self.store_results(i, y, dy)

        return self.variables, self.derivatives

    def rk4(self, deriv, n, x, dx, y):
        x0, y0 = x, np.copy(y)
        y, dy1 = deriv(x0, y)

        for i in range(n):
            y[i] = y0[i] + 0.5 * dx * dy1[i]
        xm = x0 + 0.5 * dx
        y, dy2 = deriv(xm, y)

        for i in range(n):
            y[i] = y0[i] + 0.5 * dx * dy2[i]
        y, dy3 = deriv(xm, y)

        for i in range(n):
            y[i] = y0[i] + dx * dy3[i]
        x = x0 + dx
        y, dy = deriv(x, y)

        for i in range(n):
            dy[i] = (dy1[i] + 2 * (dy2[i] + dy3[i]) + dy[i]) / 6
            y[i] = y0[i] + dx * dy[i]

        return x, y, dy

    def store_results(self, index, y, dy):
        self.variables[:, index] = y[:self.num_variables]
        self.derivatives[:, index] = dy[:self.num_derivatives]

    def adiabatic_derivatives(self, theta, y):
        dy = np.zeros_like(y)
        # Placeholder for derivative calculations (implement your equations here)
        return y, dy

    def plot_results(self):
        choice = 'x'
        while choice not in ['q', 'quit']:
            print("Choose plot type:")
            print("   p - for a PV diagram")
            print("   t - for a temperature vs crank angle plot")
            print("   e - for an energy vs crank angle plot")
            print("   q - to quit")
            choice = input(
                "p)vdiagram, t)emperature, e)nergy, q)uit: ").strip().lower()

            if choice == 'p':
                self.plot_pv_diagram()
            elif choice == 't':
                self.plot_temperature_vs_crank_angle()
            elif choice == 'e':
                self.plot_energy_vs_crank_angle()

    def plot_pv_diagram(self):
        volume = (self.variables[10, :] + globals_dict['cooler_void_volume'] +
                  globals_dict['regenerator_void_volume'] + globals_dict['heater_void_volume'] + self.variables[11, :]) * 1e6
        pressure = self.variables[9, :] * 1e-5
        plt.figure()
        plt.plot(volume, pressure, 'k')
        plt.grid(True)
        plt.xlabel('Volume (cc)')
        plt.ylabel('Pressure (bar [1bar = 100kPa])')
        plt.title('P-v diagram')
        plt.show()

    def plot_temperature_vs_crank_angle(self):
        x = np.arange(0, 361, 10)
        cooler_temp = self.variables[1, :]
        heater_temp = self.variables[2, :]
        plt.figure()
        plt.plot(x, cooler_temp, 'b-', x, heater_temp, 'r-')
        plt.axhline(y=globals_dict['cooler_temp'], color='b', linestyle='-')
        plt.axhline(y=globals_dict['regenerator_temp'],
                    color='g', linestyle='-')
        plt.axhline(y=globals_dict['heater_temp'], color='r', linestyle='-')
        plt.grid(True)
        plt.xlabel('Crank angle (degrees)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature vs crank angle')
        plt.show()

    def plot_energy_vs_crank_angle(self):
        x = np.arange(0, 361, 10)
        heat_cooler = self.variables[3, :]
        heat_regenerator = self.variables[4, :]
        heat_heater = self.variables[5, :]
        work = self.variables[8, :]
        work_compression = self.variables[6, :]
        work_expansion = self.variables[7, :]
        plt.figure()
        plt.plot(x, heat_cooler, 'b-', x, heat_regenerator, 'g-', x, heat_heater,
                 'r-', x, work, 'k-.', x, work_compression, 'b--', x, work_expansion, 'r--')
        plt.grid(True)
        plt.xlabel('Crank angle (degrees)')
        plt.ylabel('Energy [Joules]')
        plt.title('Energy vs crank angle')
        plt.show()

    def print_results(self):
        efficiency = self.variables[8, -1] / self.variables[5, -1]
        heat_cooler_power = self.variables[3, -1] * globals_dict['freq']
        heat_regenerator_power = self.variables[4, -1] * globals_dict['freq']
        heat_heater_power = self.variables[5, -1] * globals_dict['freq']
        work_power = self.variables[8, -1] * globals_dict['freq']

        print('========== Ideal Adiabatic Analysis Results ============')
        print(f' Heat transferred to the cooler: {heat_cooler_power:.2f} [W]')
        print(f' Net heat transferred to the regenerator: {
              heat_regenerator_power:.2f} [W]')
        print(f' Heat transferred to the heater: {heat_heater_power:.2f} [W]')
        print(f' Total power output: {work_power:.2f} [W]')
        print(f' Thermal efficiency: {efficiency * 100:.1f} [%]')
        print('========================================================')


# Example usage:
adiabatic_sim = AdiabaticSimulation()
variables, derivatives = adiabatic_sim.run_simulation()
adiabatic_sim.print_results()
adiabatic_sim.plot_results()
