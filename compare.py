import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example list of file paths. Replace these with the actual paths to your CSV files.
file_paths = ['./mbpo_scores.csv', './model_free/sac/stage-one-models/scores.csv']  # Add more file paths as needed.

plt.figure(figsize=(10, 6))

for file_path in file_paths:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Calculate the moving average of the scores with a window size of 50
    df['MA_Scores'] = df['scores'].rolling(window=50).mean()

    # Plot the moving average against the steps
    plt.plot(np.array(df['MA_Scores']), label=file_path)

# Add labels and legend
plt.xlabel('Steps')
plt.ylabel('Scores (Moving Average)')
plt.title('Moving Average of Scores vs. Steps')
plt.legend()

# Show the plot
plt.show()
