import pandas as pd
import urllib
import config

from sqlalchemy import create_engine

### score predictor

full_prob_df = pd.read_csv('probabilities.csv', index_col=0)
prob_df = full_prob_df.drop(full_prob_df.columns[-2:], axis=1)

potential_outcomes = ['1-0', '2-0', '3-0', '2-1', '3-1', '3-2', # win
                      '0-1', '0-2', '0-3', '1-2', '1-3', '2-3', # lose
                      '0-0', '1-1', '2-2', '3-3'] # draw

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


# Function to calculate expected points for each possible prediction
def calculate_expected_points(df):
    expected_points = {}
    for predicted_result in potential_outcomes:
        total_expected_points = 0
        for potential_result in potential_outcomes:
            likeliness = df[potential_result].iloc[0]
            points = calculate_points(predicted_result, potential_result, 'Tokai')
            total_expected_points += likeliness * points

        expected_points[predicted_result] = total_expected_points

    return expected_points

# Function to apply calculate_expected_points to each row
def apply_calculate_expected_points(row):
    row_df = row.to_frame().T  # Convert the row to a DataFrame
    expected_points = calculate_expected_points(row_df)
    # Sort the dictionary by its values in descending order
    sorted_expected_points = sorted(expected_points.items(), key=lambda item: item[1], reverse=True)

    # Select the top 5 entries
    top_5_predictions = sorted_expected_points[:5]
    # Print the results
    print("The top 5 predicted results with their expected points are:")
    for predicted_result, points in top_5_predictions:
        print(f"{predicted_result}: {points} points")

    return max(expected_points, key=expected_points.get)



# Apply the function to each row
expected_points_list = prob_df[0:7].apply(apply_calculate_expected_points, axis=1)



# Deze functie kan wat mij betreft weg
def xP(predicted_result, result_odds, points_result, points_winner):

    losing_columns = ['0-1', '0-2', '0-3', '1-2', '1-3', '2-3']
    drawing_columns = ['0-0', '1-1', '2-2', '3-3']
    winning_columns = ['1-0', '2-0', '3-0', '2-1', '3-1', '3-2']

    # TODO replace
    predicted_result = '1-0'

    if predicted_result in losing_columns:
        winner_columns = losing_columns
    elif predicted_result in drawing_columns:
        winner_columns = drawing_columns
    elif predicted_result in winning_columns:
        winner_columns = winning_columns

    #TODO replace
    result_odds = prob_df.head(1)[prob_df.columns[:-2]]
    points = x #

    points = (predicted_result,x)

    winner_odds = result_odds[winner_columns]


#    prob_correct_result =
#    prob_correct_winner = df[winner_columns].sum()

    xpoints_result = prob_correct_result*points_result
    xpoints_winner = prob_correct_winner*points_winner

    xpoints = xpoints_result + xpoints_winner

    return xpoints



### scoring defenders/midfielders

conn = "Driver={};Server={};Database={};UID={};PWD={}".format(config.driver, config.server, config.database, config.uid, config.pwd)
quoted_conn = urllib.parse.quote_plus(conn)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quoted_conn}")

RELEVANT_COMP_IDS = (55, 2, 7, 9, 12, 11)

# Retrieve Players
qry = (
    f"""SELECT *
        FROM [Football_Dashboards_Testing].[dbo].[player_match_stat]
        WHERE CompetitionId IN {RELEVANT_COMP_IDS}
        AND PositionId <= 8
        """
    )
defenders_stats_df = pd.read_sql(qry, engine)

# calculate xg per 90 minutes played
defenders_stats_df['90s_played'] = defenders_stats_df['EffectiveMinutesPlayed'] / 90
summed_stats_df = defenders_stats_df.groupby('PlayerName').agg({'Xg': 'sum', 'Goals': 'sum', '90s_played': 'sum'}).reset_index()
summed_stats_df['Xg_per_90'] = summed_stats_df['Xg'] / summed_stats_df['90s_played']

# filter for minutes played
summed_stats_df_20_matches_played = summed_stats_df[summed_stats_df['90s_played']>20]
# get xG from event data so you include non open play
