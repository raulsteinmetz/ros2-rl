import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('mbpo_scores.csv')

# Calculate moving average with a window size of 50
scores = df['scores'].rolling(window=50).mean()


# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['steps'])
plt.show()
