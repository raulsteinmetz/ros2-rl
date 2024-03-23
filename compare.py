import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
file_paths = ['./mbpo_scores.csv']
dfs = [pd.read_csv(file_path) for file_path in file_paths]



# Plot
plt.figure(figsize=(10, 6))
for i, df in enumerate(dfs):
    moving_averages = np.array(df['scores'].rolling(window=50).mean())
    steps = np.array(df['steps'])


    plt.plot(moving_averages)

plt.grid(True)
plt.show()