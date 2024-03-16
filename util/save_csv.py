import pandas as pd

def save_rewards_scores(scores, steps, network):
    # Create a DataFrame from the scores list
    df = pd.DataFrame({'scores': scores, 'steps':steps})

    # Add an 'episode' index column
    df.index.name = 'episode'

    # Save the DataFrame to a CSV file
    df.to_csv(f'networks/{network}/scores.csv')