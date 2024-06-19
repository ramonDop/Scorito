print("Lets use bookie odds")

import pandas as pd

#%%

def calculate_points(predicted_results, actual_results, game):
    """
    Calculate the points scored based on predicted and actual results.

    Args:
    predicted_results (list of str): List of predicted results.
    actual_results (list of str): List of actual results.
    game (str): Game name.

    Returns:
    int: Total points scored.
    """
    # Ensure both lists are of the same length, incase one team scores more than 9 goals
    if len(predicted_results) != len(actual_results):
        raise ValueError("Predicted and actual results lists must be of the same length.")

    predicted_home, predicted_away = map(int, predicted_results.split('-'))
    actual_home, actual_away = map(int, actual_results.split('-'))

    if game == 'Tokai':
        if predicted_results == actual_results:
            return 200  # Exact scoreline
        elif predicted_home == predicted_away and actual_home == actual_away:
            return 100  # Correct draw but not exact score
        elif (predicted_home > predicted_away and actual_home > actual_away) or \
                (predicted_home < predicted_away and actual_home < actual_away):
            if predicted_home == actual_home or predicted_away == actual_away:
                return 95  # Correct winner and one of the two scores exact
            else:
                return 75  # Correct winner and no scores exact
        elif predicted_home == actual_home or predicted_away == actual_away:
            return 20  # One of the scores is correct but the winner is not

    elif game == 'Scorito':
        if predicted_results == actual_results:
            return 90  # Exact scoreline
        elif predicted_home == predicted_away and actual_home == actual_away:
            return 60  # Correct draw
        elif (predicted_home > predicted_away and actual_home > actual_away) or \
                (predicted_home < predicted_away and actual_home < actual_away):
            return 60  # Correct winner and one of the two scores exact
    return 0


def get_probabilities():
    
    wins = ["1-0", "2-0", "2-1", "3-0", "3-1", "3-2", "4-0", "4-1", "4-2"]
    draws = ["0-0", "1-1", "2-2", "3-3"]
    losses = ["0-1", "0-2", "1-2", "0-3", "1-3", "2-3", "0-4", "1-4", "2-4"]
    
    all_results = wins + draws + losses
    
    probability_dict = {}
    for result in all_results:
        while True:
            try:
                odd = float(input(f"Odd on {result}: "))
                probability_dict[result] = odd
                break  # Exit the loop if the input is valid
            except ValueError:
                print(f"Please enter a valid odd for {result}")
        
    return probability_dict
    

def calculate_expected_points(row, df, game):
    xP = 0
    predicted_result = row["Result"]
    
    for _, prob_row in df.iterrows():
        actual_result = prob_row["Result"]
        probability_result = prob_row["Probability"]
        points_result = calculate_points(predicted_results=predicted_result, actual_results=actual_result, game=game)
        xP_added = points_result * probability_result
        xP += xP_added
    
    return xP

#%%

probabilities = get_probabilities()

#%%

df = pd.DataFrame(list(probabilities.items()), columns=["Result", "Odd"])

implied_odds = sum(1 / odd for odd in df["Odd"])
df["Probability"] = (1/df["Odd"]) / implied_odds

# Apply the calculation to each row to get the expected points
df["ExpectedPoints"]   = df.apply(lambda row: calculate_expected_points(row, df, game="Tokai"), axis=1)

# Round some columns
df["Probability"] = round(df["Probability"], 2)
df["ExpectedPoints"] = round(df["ExpectedPoints"], 1)

# Sort the dataframe by amount of expected points
df = df.sort_values(by="ExpectedPoints", ascending=False).reset_index(drop=True)

# Print the top predictions
print(df.head().to_string(index=False))