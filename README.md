# Stirling Engine Model with Anomalous Thermal Conductivity of Gas Mixtures

## Overview

This repository contains the implementation of a Stirling engine model that investigates the effects of gas mixtures, specifically hydrogen-helium mixtures, on the performance of Stirling engines. The model is developed as part of a study aimed at enhancing renewable energy systems by exploring the use of gas mixtures as working fluids in Stirling engines. The innovation in this study lies in the consideration of the "anomalous" thermal conductivity behavior exhibited by hydrogen-helium mixtures, which can significantly impact the engine's performance.

## Features

- **Stirling Engine Simulation**: A detailed thermodynamic model that simulates the performance of a Stirling engine under different working conditions.
- **Gas Mixtures as Working Fluids**: Incorporation of hydrogen-helium mixtures, focusing on their anomalous thermal conductivity properties.
- **Model Validation**: Validation of the model against existing data, including the use of Enskog theory to predict thermophysical properties of the gas mixtures.
- **Visualisation**: Generation of figures to visualise the thermal conductivity effects and engine performance.

## Repository Structure

- **`pyenskog/`**: Contains the precompiled `.whl` file for the C++ extensions.
- **`enskog.py`**: Python module that for simulating the thermophysical properties of gas mixtures using Enskog theory.
- **`stirling_engine.py`**: The main script that runs the Stirling engine simulation.
- **`thermal_conductivity_predictions_validation.py`**: Script for validating the thermal conductivity predictions using experimental data.
- **`working_fluid.py`**: Module that handles the properties of working fluids, including the prediction of their behavior in the engine.
- **`figures/`**: Directory for storing generated figures from simulations.
- **`README.md`**: This file.
- **`LICENSE`**: The license under which this code is distributed.
- **`requirements.txt`**: List of Python dependencies required to run the simulations.

## Installation

### Prerequisites

- Python 3.8 or higher

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Igben-Nehemiah/stirling-engine-model.git
   cd stirling-engine-model
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
3. Install Enskog calculator:
    - Linux
        ```bash 
        pip install pyenskog/pyenskog-0.2-cp310-cp310-linux_x86_64.whl
    - Windows (Not yet implemented)

### Usage
1. To run the Stirling engine simulation:
    ```bash
    python stirling_engine.py
2. To validate the thermal conductivity predictions:
    ```bash
    python thermal_conductivity_predictions_validation.py
This script compares the predicted thermal conductivities of the gas mixtures against experimental data to ensure accuracy.

