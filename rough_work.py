import pandas as pd
import plotly.express as px

# Define the data for the Gantt chart
data = {
    'Task': [
        'Literature Review Part 1', 'Literature Review Part 2', 'Identify and Evaluate Technologies Part 1', 'Identify and Evaluate Technologies Part 2',
        'Analyze Data Part 1', 'Analyze Data Part 2', 'Techno-Economic Analysis Part 1', 'Techno-Economic Analysis Part 2',
        'Apply Findings and Model Process Part 1', 'Apply Findings and Model Process Part 2', 'Environmental Impact Assessment',
        'Compile Report and Recommendations'
    ],
    'Start': [
        '2024-06-01', '2024-06-15', '2024-06-15', '2024-06-30',
        '2024-07-01', '2024-07-15', '2024-07-15', '2024-07-31',
        '2024-08-01', '2024-08-15', '2024-08-15', '2024-08-22'
    ],
    'Finish': [
        '2024-06-15', '2024-06-30', '2024-06-30', '2024-07-15',
        '2024-07-15', '2024-07-31', '2024-07-31', '2024-08-15',
        '2024-08-15', '2024-08-22', '2024-08-22', '2024-08-31'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert the Start and Finish columns to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['Finish'] = pd.to_datetime(df['Finish'])

# Create the Gantt chart using Plotly
fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", title="Project Timeline for CO2 Purification for Sequestration",
                  labels={"Task": "Project Task"})
fig.update_yaxes(categoryorder="total ascending")

# Display the Gantt chart
fig.show()
