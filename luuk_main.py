# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:58:12 2024

@author: l.vansteenoven
"""
import warnings
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier


# Ignore specific warning
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from numpy.linalg import lstsq

import requests

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


#%% Define functions to load in the results

def get_results_dataframe(start_year, end_year):

    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Loop through the years and read each TSV file
    for year in range(start_year, end_year):
        # Define the URL of the TSV file
        results_url = f"https://www.eloratings.net/{year}_results.tsv"
        
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(results_url, sep='\t', header=None)
            
        # Append the DataFrame to the list
        dfs.append(df)
    
    # Concatenate all DataFrames in the list into one DataFrame
    df = pd.concat(dfs, ignore_index=True)
    
    # Define the column names
    column_names = ["Year", "Month", "Day", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore", "Tournament", "Location", "RatingChangeTeam1", "RatingTeam1", "RatingTeam2", "RankChangeTeam1", "RankChangeTeam2", "RankTeam1", "RankTeam2"]
    
    # Assign the column names to the combined DataFrame
    df.columns = column_names
    
    # Only keep relevant columns
    df = df[["Year", "Month", "Day", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore", "Tournament", "Location", "RatingTeam1", "RatingTeam2"]]
    
    return df

def get_dataframe(url):

    # Fetch the data from the URL
    response = requests.get(url)
    data = response.text
    
    # Initialize a list to store rows
    rows = []
    
    # Read the data line by line
    for line in data.splitlines():
        # Split each line by tab
        row = line.strip().split('\t')
        # Append the row to the rows list
        rows.append(row)
    
    # Determine the maximum number of columns
    max_columns = max(len(row) for row in rows)
    
    # Normalize the rows by adding empty strings to rows with fewer columns
    normalized_rows = [row + [''] * (max_columns - len(row)) for row in rows]
    
    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(normalized_rows)
    
    df = df[df.columns[:2]]
    
    df.columns = ["Id", "Name"]
    
    return df

def get_fixtures(fixtures_url):
    
    df = pd.read_csv(fixtures_url, sep='\t', header=None)
    df = df[df.columns[:11]]

    column_names = ["Year", "Month", "Day", "HomeTeam", "AwayTeam", "Tournament", "Location", "RankTeam1", "RankTeam2", "RatingTeam1", "RatingTeam2"]
    df.columns = column_names
    
    return df

#%% Load in the data

df = get_results_dataframe(start_year=2016, end_year=2025)

# Define the URL of the TSV files
teams_url = "https://www.eloratings.net/en.teams.tsv"
tournaments_url = "https://www.eloratings.net/en.tournaments.tsv"
fixtures_url = "https://www.eloratings.net/fixtures.tsv"

teams_df = get_dataframe(teams_url)
tournaments_df = get_dataframe(tournaments_url)
fixtures_df = get_fixtures(fixtures_url)

#%% Map the information from the teams and tournaments dataframe to the results dataframe

# Create a dictionary for id to name mapping
team_id_to_name = dict(zip(teams_df['Id'], teams_df['Name']))
competition_id_to_name = dict(zip(tournaments_df['Id'], tournaments_df['Name']))

# Replace HomeTeam and AwayTeam ids with names for the results dataframe
df['HomeTeam'] = df['HomeTeam'].map(team_id_to_name)
df['AwayTeam'] = df['AwayTeam'].map(team_id_to_name)

# Replace HomeTeam and AwayTeam ids with names for the fixtures dataframe
fixtures_df['HomeTeam'] = fixtures_df['HomeTeam'].map(team_id_to_name)
fixtures_df['AwayTeam'] = fixtures_df['AwayTeam'].map(team_id_to_name)

# Replace Tournament id with name for the results dataframe
df['Tournament'] = df['Tournament'].map(competition_id_to_name)

# Replace Tournament id with name for the fixtures dataframe
fixtures_df['Tournament'] = fixtures_df['Tournament'].map(competition_id_to_name)

# Add a MatchName column to results and fixtures
df['MatchName'] =  df["HomeTeam"].astype(str) + "-" + df["AwayTeam"].astype(str)
fixtures_df['MatchName'] =  fixtures_df["HomeTeam"].astype(str) + "-" + fixtures_df["AwayTeam"].astype(str)

#%% Rolling average of goals scored and conceded

# Create date column
df["Date"] = df["Month"].astype(str) + "-" + df["Day"].astype(str) + "-" + df["Year"].astype(str)
df["Date"] = pd.to_datetime(df["Date"])

# Create date column for fixtures
fixtures_df["Date"] = fixtures_df["Month"].astype(str) + "-" + fixtures_df["Day"].astype(str) + "-" + fixtures_df["Year"].astype(str)
fixtures_df["Date"] = pd.to_datetime(fixtures_df["Date"])

# Calculate rolling averages for each team
def calculate_rolling_average_goals(df, window=10):
    # Initialize dictionaries to store rolling averages
    rolling_averages = {team: [] for team in pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()}

    # Create a new dataframe to store the results
    results = pd.DataFrame(columns=["Team", "Date", "RollingAvgGoalsScored", "RollingAvgGoalsConceded"])

    # Iterate over each team
    for team in rolling_averages.keys():
        # Filter matches involving the team
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
        team_matches["GoalsScored"] = team_matches.apply(
            lambda row: row["HomeScore"] if row["HomeTeam"] == team else row["AwayScore"], axis=1
        )
        team_matches["GoalsConceded"] = team_matches.apply(
            lambda row: row["AwayScore"] if row["HomeTeam"] == team else row["HomeScore"], axis=1
        )
        team_matches["RollingAvgGoalsScored"] = team_matches["GoalsScored"].ewm(span=window, adjust=False).mean().shift(1)
        team_matches["RollingAvgGoalsConceded"] = team_matches["GoalsConceded"].ewm(span=window, adjust=False).mean().shift(1)
        
        # Remove the first 10 matches since they don't have a valid rolling average
        team_matches = team_matches.iloc[window:]
        
        # Append results to the results dataframe
        team_results = team_matches[["Date", "RollingAvgGoalsScored", "RollingAvgGoalsConceded"]].copy()
        team_results["Team"] = team
        results = pd.concat([results, team_results], ignore_index=True)

    # Sort results by date and team
    results = results.sort_values(by=["Team", "Date"])
    
    return results

rolling_avg_df = calculate_rolling_average_goals(df)

#%% Filter the dataframe

# Throw all games away with too much goals
df["TotalGoals"] = df["HomeScore"] + df["AwayScore"]
df = df[df["TotalGoals"] <= 5]

# Adding a Result column to the results
df["Result"] = df["HomeScore"].astype(str) + "-" + df["AwayScore"].astype(str)

#%% Swapping results so higher ranked team 'plays at home' to improve model

def reorder_result_and_rating(row):
    rating_team1 = row['RatingTeam1']
    rating_team2 = row['RatingTeam2']
    result = row['Result']

    name_team1 = row["HomeTeam"]
    name_team2 = row["AwayTeam"]

    score_team1, score_team2 = map(int, result.split('-'))

    if rating_team1 >= rating_team2:
        new_result = f"{score_team1}-{score_team2}"
        higher_rating = rating_team1
        lower_rating = rating_team2
        higher_team_name = name_team1
        lower_team_name  = name_team2
    else:
        new_result = f"{score_team2}-{score_team1}"
        higher_rating = rating_team2
        lower_rating = rating_team1
        higher_team_name = name_team2
        lower_team_name  = name_team1

    return pd.Series([new_result, higher_rating, lower_rating, higher_team_name, lower_team_name], index=['NewResult', 'HigherRating', 'LowerRating', 'HigherTeam', 'LowerTeam'])

df[['NewResult', 'HigherRating', 'LowerRating', 'HigherTeam', 'LowerTeam']] = df.apply(reorder_result_and_rating, axis=1)

#%% Do something similar but for fixtures_df

def reorder_matchname_and_rating(row):
    rating_team1 = row['RatingTeam1']
    rating_team2 = row['RatingTeam2']
    match_name = row['MatchName']
    
    name_team1 = row["HomeTeam"]
    name_team2 = row["AwayTeam"]
    
    team1, team2 = match_name.split('-')

    if rating_team1 >= rating_team2:
        new_match_name = f"{team1}-{team2}"
        higher_rating = rating_team1
        lower_rating = rating_team2
        higher_team_name = name_team1
        lower_team_name  = name_team2

    else:
        new_match_name = f"{team2}-{team1}"
        higher_rating = rating_team2
        lower_rating = rating_team1
        higher_team_name = name_team2
        lower_team_name  = name_team1

    return pd.Series([new_match_name, higher_rating, lower_rating, higher_team_name, lower_team_name],
                     index=['NewMatchName', 'HigherRating', 'LowerRating', 'HigherTeam', 'LowerTeam'])

fixtures_df[['NewMatchName', 'HigherRating', 'LowerRating', 'HigherTeam', 'LowerTeam']] = fixtures_df.apply(reorder_matchname_and_rating, axis=1)

#%% Merge the Rolling average goals scored and conceded

# Merge the rolling averages with the main dataframe
def add_rolling_averages(df, rolling_avg_df):
    # Merge rolling averages for HigherTeam
    higher_team_avg = rolling_avg_df.rename(columns={
        "Team": "HigherTeam", 
        "RollingAvgGoalsScored": "HigherTeamScoring", 
        "RollingAvgGoalsConceded": "HigherTeamConceding"
    })
    df = pd.merge(df, higher_team_avg, how="left", left_on=["HigherTeam", "Date"], right_on=["HigherTeam", "Date"])
    
    # Merge rolling averages for LowerTeam
    lower_team_avg = rolling_avg_df.rename(columns={
        "Team": "LowerTeam", 
        "RollingAvgGoalsScored": "LowerTeamScoring", 
        "RollingAvgGoalsConceded": "LowerTeamConceding"
    })
    df = pd.merge(df, lower_team_avg, how="left", left_on=["LowerTeam", "Date"], right_on=["LowerTeam", "Date"])
    
    return df

df = add_rolling_averages(df, rolling_avg_df)

df = df.dropna(subset=["HigherTeamScoring", "HigherTeamConceding", "LowerTeamScoring", "LowerTeamConceding"])

df["HigherTeamExpectedScoring"] = df["HigherTeamScoring"] - df["LowerTeamConceding"]
df["HigherTeamExpectedConceding"] = df["HigherTeamConceding"] - df["LowerTeamScoring"]


#%% Add columns to the results and fixtures dataframes

fixtures_df["RatingTotal"] = fixtures_df["HigherRating"] + fixtures_df["LowerRating"]
fixtures_df["RatingDifferenceAbsolute"] = fixtures_df["HigherRating"] - fixtures_df["LowerRating"]
fixtures_df["RatingDifferenceRelative"] = fixtures_df["HigherRating"] / fixtures_df["LowerRating"]

df["RatingTotal"] = df["HigherRating"] + df["LowerRating"]
df["RatingDifferenceAbsolute"] = df["HigherRating"] - df["LowerRating"]
df["RatingDifferenceRelative"] = df["HigherRating"] / df["LowerRating"]


#%% Filter the results and fixtures dataframes

# Filter fixtures to only contain the EURO's
fixtures_df = fixtures_df[fixtures_df["Tournament"] == "European Championship"]

# Filter results to only contain real matches, and to exclude low level matches
df = df[(df["Tournament"] != "Friendly") & (df["Tournament"] != "Friendly tournament")] 

# Do not use the world cup in training
world_cup = df[df["Tournament"] == "World Cup"]
world_cup[['NewMatchName', 'HigherRating', 'LowerRating', 'HigherTeam', 'LowerTeam']] = world_cup.apply(reorder_matchname_and_rating, axis=1)


df = df[df["Tournament"] != "World Cup"]
df = df[df["LowerRating"] >= 1600]


print(f"Predicting {len(fixtures_df)} fixtures for the EURO's matches using {len(df)} previous matches")

df.to_csv("international_matches.csv")


#%% Build model

# Encode the target variable
label_encoder = LabelEncoder()
df["ResultTransformed"] = label_encoder.fit_transform(df["Result"])

# Verify and correct the unique classes
unique_classes = label_encoder.classes_
num_classes = len(unique_classes)

# Split the data into training and testing sets
# TODO; run model with only RatingDifferenceAbsolute
X = df[["RatingDifferenceAbsolute"]] #, "RatingDifferenceRelative", "RatingTotal"]] #TODO also change back down below
y = df["ResultTransformed"]

# Check the unique classes in the training set
unique_classes_train = y.unique()
unique_classes_train.sort()

# Add a dummy row with the missing class to the training data
missing_classes = set(range(num_classes)) - set(unique_classes_train)
for missing_class in missing_classes:
    dummy_row = pd.DataFrame({
        "RatingDifferenceAbsolute": [0],
        "ResultTransformed": [missing_class]
    })
    # Append the dummy row to the training data
    X_train = pd.concat([X, dummy_row[["RatingDifferenceAbsolute"]]], ignore_index=True)
    y_train = pd.concat([y, dummy_row["ResultTransformed"]], ignore_index=True)


#%% Get the best param grid


#%% Get the best param grid

# Define the parameter grid for GridSearchCV including max_iter
param_grid = { # TODO create custom score here, based on points for Tokai
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'saga'],
#    'solver': ['saga'],

    'max_iter': [100, 500, 1000, 5000, 10000]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1, scoring='neg_log_loss')

# Perform the grid search on the training data
grid_search.fit(X, y)

# Get the best parameters and the best score from the grid search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

#%% Print the best parameters and the best score

import json

# Save best parameters to a JSON file
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)

#%%

# Load best parameters from JSON file
import json
with open('best_params.json', 'r') as f:
    best_params = json.load(f)
print("Loaded Best Parameters (JSON):", best_params)

# Train the Logistic Regression model with the best parameters
best_model = LogisticRegression(**best_params)
best_model.fit(X, y)

#%% Get the expect scoring and conceding 

def get_latest_record(team, date, rolling_avg_df):
    # Filter the rolling_avg_df for the team and dates before the given date
    team_records = rolling_avg_df[(rolling_avg_df["Team"] == team) & (rolling_avg_df["Date"] < date)]
    if not team_records.empty:
        return team_records.iloc[-1]  # Get the most recent record
    else:
        return None

# Function to fill in the expected scoring and conceding columns
def fill_expected_columns(fixtures_df, rolling_avg_df):
    fixtures_df["HigherTeamExpectedScoring"] = fixtures_df.apply(
        lambda row: get_latest_record(row["HigherTeam"], row["Date"], rolling_avg_df)["RollingAvgGoalsScored"] 
                    if get_latest_record(row["HigherTeam"], row["Date"], rolling_avg_df) is not None else None,
        axis=1
    )
    
    fixtures_df["HigherTeamExpectedConceding"] = fixtures_df.apply(
        lambda row: get_latest_record(row["HigherTeam"], row["Date"], rolling_avg_df)["RollingAvgGoalsConceded"] 
                    if get_latest_record(row["HigherTeam"], row["Date"], rolling_avg_df) is not None else None,
        axis=1
    )
    
    return fixtures_df

fixtures_df = fill_expected_columns(fixtures_df, rolling_avg_df)

#%% Functions to calculate the points, and expected points per prediction / result

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

def calculate_expected_points(predicted_result, r_diff_abs, best_model, label_encoder, game):
    probabilities = best_model.predict_proba(np.array([[r_diff_abs]]))
    probability_df = pd.DataFrame(probabilities, columns=label_encoder.classes_)
    
    xP = 0
    
    for actual_result in probability_df.columns:
        points_result = calculate_points(predicted_result, actual_result, game)
        probability_result = probability_df[actual_result].iloc[0]
        xP_added = points_result * probability_result
        xP += xP_added
    
    return xP

def apply_calculate_expected_points(fixtures_df, best_model, label_encoder):
    for game in ["Scorito", "Tokai"]:
        for result in label_encoder.classes_:
            col_name = f"{game}{result}"
            fixtures_df[col_name] = fixtures_df.apply(
                lambda row: calculate_expected_points(
                    result,
                    row["RatingDifferenceAbsolute"],
                    #row["RatingDifferenceRelative"],
                    #row["RatingTotal"],
                    best_model,
                    label_encoder,
                    game
                ),
                axis=1
            )
        
    return fixtures_df
    
fixtures_df = apply_calculate_expected_points(fixtures_df, best_model, label_encoder)
world_cup = apply_calculate_expected_points(world_cup, best_model, label_encoder)

#%%

def print_best_prediction(fixtures_df, game, match_name, n_predictions):
    match = fixtures_df[fixtures_df["NewMatchName"] == match_name]
    
    # Assuming match is your dataframe
    game_columns = [col for col in match.columns if game in col]
    match_filtered = match[game_columns]

    points_per_prediction_dataframe = match_filtered.T
    points_per_prediction_dataframe.columns = ["Expected Points"]
        
    # Remove the "game" prefix from the index labels
    points_per_prediction_dataframe.index = points_per_prediction_dataframe.index.str.replace(game, " ")
    
    # Sort the dataframe by "Expected Points" in descending order
    best_predictions = points_per_prediction_dataframe.sort_values(by="Expected Points", ascending=False).head(n_predictions)

    print(f"The best predictions for {match_name} for {game} are:", "\n", best_predictions)
    
    return (best_predictions.iloc[0]["Expected Points"])


def get_best_prediction(fixtures_df, match_name, game):

    match = fixtures_df[fixtures_df["NewMatchName"] == match_name]
    
    # Assuming match is your dataframe
    game_columns = [col for col in match.columns if game in col]
    match_filtered = match[game_columns]

    points_per_prediction_dataframe = match_filtered.T
    points_per_prediction_dataframe.columns = ["Expected Points"]
        
    # Remove the "game" prefix from the index labels
    points_per_prediction_dataframe.index = points_per_prediction_dataframe.index.str.replace(game, " ")
    
    # Sort the dataframe by "Expected Points" in descending order
    best_prediction = points_per_prediction_dataframe.sort_values(by="Expected Points", ascending=False).head(1)
        
    # Get the index of the first row (the best prediction)
    best_prediction_result = best_prediction.index[0].strip()
    best_prediction_xP = best_prediction.iloc[0]["Expected Points"]

    return best_prediction_result, round(best_prediction_xP)
    
def print_all_predictions(fixtures_df, game):
    predicted_points = []
    for match_name in fixtures_df["NewMatchName"]:
        added_predicted_points = print_best_prediction(fixtures_df, match_name=match_name, game=game, n_predictions=5)
        predicted_points.append(added_predicted_points)
        print("\n")
        
    print(f"Total predicted points for {len(predicted_points)} matches: {round(sum(predicted_points))}, which is around {round(sum(predicted_points) / len(predicted_points), 1)} points per game")

#print_best_prediction(fixtures_df, match_name="Germany-Scotland", game="Scorito", n_predictions=5)
print_all_predictions(fixtures_df=fixtures_df, game="Scorito")