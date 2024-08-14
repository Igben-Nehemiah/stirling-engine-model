import os
from math import pi
from stirling_engine import StirlingEngine
from working_fluid import GasMixtures, IdealGasWorkingFluid
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

def simulate_pure_gas_KPIs(heater_wall_temperature_values, working_fluid):
    heat_values = []
    power_values = []
    work_values = []
    efficiency_values = []

    for temp in heater_wall_temperature_values:
        params_mod = StirlingEngine.Parameters(
            freq=41.7,  # Operating frequency (Hz)
            mean_pressure=4.13e6,  # Charge pressure (Pa)
            twk=288,  # Cooler wall temperature (K)
            twh=temp,  # Heater wall temperature (K)
            inside_dia_h_tube=0.00302,  # Heater tube inside diameter (m)
            no_h_tubes=40,  # Number of heater tubes
            len_h_tube=0.2453,  # Heater tube length (m)
            inside_dia_k_tube=0.00108,  # Cooler tube inside diameter (m)
            no_k_tubes=312,  # Number of cooler tubes
            len_k_tube=0.0461,  # Cooler tube length (m)
            no_r_tubes=8,  # Number of regenerator tubes
            reg_porosity=0.69,  # Regenerator matrix porosity
            reg_len=0.0226,  # Regenerator length (m)
            k_reg=15,  # Regenerator thermal conductivity [W/m/K] - Stainless Steel
            dwire=0.0000406,  # Regenerator matrix wire diameter (m)
            domat=0.0226,  # Regenerator housing internal diameter (m)
            dout=0.03,  # Regenerator housing external diameter (m)
            working_fluid=working_fluid,
            vclc=28.68e-6,  # Compression space total volume (m^3)
            vcle=30.52e-6,  # Expansion space total volume (m^3)
            vswc=114.13e-6,  # Cooler void volume (m^3)
            vswe=120.82e-6,  # Heater void volume (m^3)
            alpha=pi/2  # Expansion phase angle advance (radians)
        )

        engine = StirlingEngine(params_mod)
        engine.run_simulation()
        heat_values.append(engine.EngineKPIs.net_heat)
        power_values.append(engine.EngineKPIs.net_power)
        work_values.append(engine.EngineKPIs.net_work)
        efficiency_values.append(engine.engineKPIs.efficiency)

    return heat_values, power_values, work_values, efficiency_values

def simulate_gas_mixtures_KPIs(temp_hot, hydrogen_mole_fractions, 
                               hydrogen_helium_params, is_classical):
    heat_values = []
    power_values = []
    work_values = []
    efficiency_values = []

    for hydrogen_mole_fraction in hydrogen_mole_fractions:
        hydrogen_helium_params.x1 = hydrogen_mole_fraction
        hydrogen_helium_params.x2 = 1-hydrogen_mole_fraction
        hydrogen_helium_params.is_classical = is_classical
        hydrogen_helium = GasMixtures(hydrogen_helium_params)

        params_mod = StirlingEngine.Parameters(
            freq=41.7,  # Operating frequency (Hz)
            mean_pressure=4.13e6,  # Charge pressure (Pa)
            twk=288,  # Cooler wall temperature (K)
            twh=temp_hot,  # Heater wall temperature (K)
            inside_dia_h_tube=0.00302,  # Heater tube inside diameter (m)
            no_h_tubes=40,  # Number of heater tubes
            len_h_tube=0.2453,  # Heater tube length (m)
            inside_dia_k_tube=0.00108,  # Cooler tube inside diameter (m)
            no_k_tubes=312,  # Number of cooler tubes
            len_k_tube=0.0461,  # Cooler tube length (m)
            no_r_tubes=8,  # Number of regenerator tubes
            reg_porosity=0.69,  # Regenerator matrix porosity
            reg_len=0.0226,  # Regenerator length (m)
            # Regenerator thermal conductivity [W/m/K] - Stainless Steel
            k_reg=15,
            dwire=0.0000406,  # Regenerator matrix wire diameter (m)
            domat=0.0226,  # Regenerator housing internal diameter (m)
            dout=0.03,  # Regenerator housing external diameter (m)
            working_fluid=hydrogen_helium,
            vclc=28.68e-6,  # Compression space total volume (m^3)
            vcle=30.52e-6,  # Expansion space total volume (m^3)
            vswc=114.13e-6,  # Cooler void volume (m^3)
            vswe=120.82e-6,  # Heater void volume (m^3)
            alpha=pi/2  # Expansion phase angle advance (radians)
        )

        engine = StirlingEngine(params_mod)
        engine.run_simulation()
        heat_values.append(engine.EngineKPIs.net_heat)
        power_values.append(engine.EngineKPIs.net_power)
        work_values.append(engine.EngineKPIs.net_work)
        efficiency_values.append(engine.engineKPIs.efficiency)

    return heat_values, power_values, work_values, efficiency_values

def plot_pure_gas_KPIs(T_cold, heater_wall_temperature_values, 
           power_values_hydrogen, power_values_helium,
           efficiency_values_hydrogen, efficiency_values_helium,
           ):
    carnot_efficiency = (1 - (T_cold / heater_wall_temperature_values)) * 100
    font_size = 20
    # Figure 1: Power Output
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(heater_wall_temperature_values, power_values_hydrogen, label='Hydrogen', color='#1f77b4', linestyle='-', linewidth=2, marker='o', markersize=6)
    ax1.plot(heater_wall_temperature_values, power_values_helium, label='Helium', color='#d62728', linestyle='--', linewidth=2, marker='s', markersize=6)
    ax1.set_xlabel('Temperature (K)', fontsize=font_size, fontweight='bold')
    ax1.set_ylabel('Power Output (W)', fontsize=font_size, fontweight='bold')
    ax1.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(0.05, 0.95))  # Adjusted legend position inside the plot area
    ax1.grid(True)  # Adjusted grid style
    ax1.tick_params(axis='both', which='major', labelsize=font_size)

    # Save Figure 1
    fig1.savefig('./figures/power_vs_temperature.pdf', dpi=300)

    # Figure 2: Efficiency
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(heater_wall_temperature_values, efficiency_values_hydrogen, label='Hydrogen', color='#1f77b4', linestyle='-', linewidth=2, marker='o')
    ax2.plot(heater_wall_temperature_values, efficiency_values_helium, label='Helium', color='#d62728', linestyle='--', linewidth=2, marker='s')
    ax2.plot(heater_wall_temperature_values, carnot_efficiency, label='Carnot', color='green', linestyle='-.', linewidth=2, marker='^')
    ax2.set_xlabel('Temperature (K)', fontsize=font_size, fontweight='bold')
    ax2.set_ylabel('Thermal Efficiency (%)', fontsize=font_size, fontweight='bold')
    ax2.legend(fontsize=font_size)  # Adjusted legend position inside the plot area
    ax2.grid(True)  # Adjusted grid style
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    # Save Figure 2
    fig2.savefig('./figures/efficiency_vs_temperature.pdf', dpi=300)

def plot_gas_mixture_KPI(temp_hot, hydrogen_mole_fractions, kpi_enskog_max, kpi_enskog_min,
         kpi_classical_max, kpi_classical_min, kpi_name=""):
    fontsize = 20
    plt.figure(figsize=(10, 6))
    plt.plot(hydrogen_mole_fractions, kpi_enskog_max, label='Enskog transient',
             color='#1f77b4', linestyle='-', linewidth=2, marker='o')
    plt.plot(hydrogen_mole_fractions, kpi_classical_max, label='Parallel bound',
             color='#d62728', linestyle='-', linewidth=2, marker='s')
    plt.plot(hydrogen_mole_fractions, kpi_enskog_min, label='Enskog steady state',
             color='#1f77b4', linestyle='--', linewidth=2, marker='o')
    plt.plot(hydrogen_mole_fractions, kpi_classical_min, label='Series bound',
             color='#d62728', linestyle='--', linewidth=2, marker='s')
    plt.xlabel(
        'Mole Fraction of Hydrogen ($x_{H_2}$)', fontsize=fontsize, fontweight='bold')
    plt.ylabel('Power (W)', fontsize=fontsize, fontweight='bold')
    plt.xlim(0, 1)  # Set x-axis limits from 0 to 1
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./figures/{kpi_name}_{temp_hot}.pdf", dpi=300)

def run():
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
        
    # Data parameters for hydrogen and helium
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

    # Creating instances for each gas
    hydrogen = IdealGasWorkingFluid(hydrogen_params)
    helium = IdealGasWorkingFluid(helium_params)

    hydrogen_helium_params = GasMixtures.Parameters(
        fluid1=hydrogen,
        fluid2=helium,
        x1=0.25,
        x2=0.75,
        is_classical=False,
        is_max=False
    )

    heater_wall_temps = np.linspace(450, 1050, 10)
    
    # Pure gases working fluid
    _, power_hydrogen, _, efficiency_hydrogen = simulate_pure_gas_KPIs(heater_wall_temps, hydrogen)
    _, power_helium, _, efficiency_helium = simulate_pure_gas_KPIs(heater_wall_temps, helium)

    # Plot setting 
    plt.style.use(["science", "ieee", "no-latex"])

    cold_sink_temp = 288  # Temperature of the cold reservoir (in Kelvin)


    plot_pure_gas_KPIs(cold_sink_temp, heater_wall_temps, power_hydrogen,
                       power_helium, efficiency_hydrogen, efficiency_helium)

    # Gas mixtures working fluid
    # temp_hot = 1200
    # hydrogen_mole_fractions = np.linspace(1, 0, 21, endpoint=True)

    # # Simulate KPIs for Enskog and Classical models
    # hydrogen_helium_params.is_max = True
    # heat_enskog_max, power_enskog_max, work_enskog_max, efficiency_enskog_max = simulate_gas_mixtures_KPIs(
    #     temp_hot, hydrogen_mole_fractions, hydrogen_helium_params, is_classical=False)
    # hydrogen_helium_params.is_max = False
    # heat_enskog_min, power_enskog_min, work_enskog_min, efficiency_enskog_min = simulate_gas_mixtures_KPIs(
    #     temp_hot, hydrogen_mole_fractions, hydrogen_helium_params, is_classical=False)
    # hydrogen_helium_params.is_max = True
    # heat_classical_max, power_classical_max, work_classical_max, efficiency_classical_max = simulate_gas_mixtures_KPIs(
    #     temp_hot, hydrogen_mole_fractions, hydrogen_helium_params, is_classical=True)
    # hydrogen_helium_params.is_max = False
    # heat_classical_min, power_classical_min, work_classical_min, efficiency_classical_min = simulate_gas_mixtures_KPIs(
    #     temp_hot, hydrogen_mole_fractions, hydrogen_helium_params, is_classical=True)

    # plot_gas_mixture_KPI(temp_hot, hydrogen_mole_fractions, power_enskog_max, 
    #                      power_enskog_min, power_classical_min, power_enskog_min, "power_vs_hydrogen_mole_fraction")
    
    
if __name__ == "__main__":
    run()