import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Data from the table
model_types = ['Adiabatic model', 'Simple model',
               'Present Study', 'Experimental Results']
heat_input = [13.280, 12.762, 14.9, 11.31]
power = [8.3, 6.7, 7.0, 3.958]
efficiency = [62.5, 52.5, 46.9, 35]

# Setting the positions and width for the bars
bar_width = 0.35
r1 = np.arange(len(heat_input))
r2 = [x + bar_width for x in r1]

# Plotting the results
plt.style.use(["science", "ieee", "no-latex"])
# Creating the bar chart
plt.figure(figsize=(12, 7))
plt.bar(r1, heat_input, color='#5DADE2', width=bar_width,
        edgecolor='grey', label='Heat Input (KW)')
plt.bar(r2, power, color='#E74C3C', width=bar_width,
        edgecolor='grey', label='Power (KW)')

# Adding the line plot for Efficiency
plt.plot(r1 + bar_width/2, efficiency, color='#2ECC71', marker='o',
         markersize=8, linestyle='-', linewidth=2, label='Efficiency (%)')

# Adding labels
plt.xlabel('Model Type', fontweight='bold', fontsize=14)
plt.xticks([r + bar_width/2 for r in range(len(heat_input))],
           model_types, fontsize=14)
plt.ylabel('Values (KW and %)', fontweight='bold', fontsize=14)

plt.legend(loc='upper right', fontsize=14)

# Adding gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Tight layout for better spacing
plt.tight_layout()

# Displaying the plot
plt.savefig("./figures/performance_comparison.png")