"""This module is used to create a cleaned version of the csv files in the data folder.
The cleanup of each csv file entails:
1. loading the csv file (from the data_folder specified)
2. substituting the values in the "set" column with cleaned values. The clean values are all subsets
    coalitions with the players being seperated with a "_" instead of no seperation between the players.
3. storing (overwriting) the cleaned csv file in the data folder with the same name as the original csv file
"""
import os
import pandas as pd
from approximators.base import powerset
from tqdm import tqdm

if __name__ == "__main__":

    # define data folder
    DATA_FOLDER = "bike"

    # get all player folders of the data folder
    player_folders = os.listdir(DATA_FOLDER)

    # iterate over all player folders
    for player_folder in player_folders:
        # get number of players from the player folder name
        n_players = int(player_folder)
        n_coalitions = 2 ** n_players

        # get all csv files in the player folder
        csv_files = os.listdir(os.path.join(DATA_FOLDER, player_folder))

        # create cleaned "set" column
        cleaned_sets = []
        # iterate over the powerset / all combinations of the
        for coalition in powerset(range(n_players)):
            coalition = sorted(coalition)
            coalition = 's_' + '_'.join([str(player) for player in coalition])
            cleaned_sets.append(coalition)

        # print staring message
        print(f"Cleaning {player_folder} ...")

        # iterate over all csv files
        for csv_file in tqdm(csv_files):
            # load csv file
            df = pd.read_csv(os.path.join(DATA_FOLDER, player_folder, csv_file))

            # replace "set" column with cleaned values
            df["set"] = cleaned_sets

            # save_path
            save_path = os.path.join(DATA_FOLDER, player_folder, csv_file)

            # save cleaned csv file
            df.to_csv(save_path, index=False)
