# Class for loading a looking up data from a sqlite3 database

import pandas
import sqlite3
import numpy as np
from typing import List, Type, Dict
import math

# create soccer database class
class SoccerDatabase:
    def __init__(self, filename: str) -> None:
        # create connection to database
        self.filename = filename
        self.conn = sqlite3.connect(filename)
        self.cur = self.conn.cursor()
        self.team_data = None
        self.player_data = None

    # disconnect from database when object is deleted
    def __del__(self) -> None:
        # close connection to database
        self.conn.close() 

    def get_db_tables(self) -> list:
        '''
        Returns a list of tables in the database
        '''
        # get list of tables in database
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [x[0] for x in self.cur.fetchall()]
    
    def get_pandas_df(self, table: str) -> pandas.DataFrame:
        '''
        Returns a pandas dataframe from the given table
        '''
        # get pandas dataframe from table
        return pandas.read_sql_query(f"SELECT * FROM {table}", self.conn)

    def create_player_data(self) -> None:
        # get pandas dataframe from table
        players = self.get_pandas_df("Player")
        players = players.sort_values(by="player_api_id")
        player_attributes = self.get_pandas_df("Player_Attributes")
        player_attributes = player_attributes.sort_values(by="player_api_id")
        # create dict for storing player data
        player_data = {}
        player_attributes_index = 0
        # iterate through players
        for index, player in players.iterrows():
            # get player id
            player_id = player["player_api_id"]
            player_attributes_list = []
            while(player_attributes_index < len(player_attributes) and player_attributes.iloc[player_attributes_index]["player_api_id"] <= player_id):
                if(player_attributes.iloc[player_attributes_index]["player_api_id"] == player_id):
                    player_attributes_list.append(player_attributes.iloc[player_attributes_index])
                player_attributes_index += 1

            player_attributes_list = pandas.DataFrame(player_attributes_list)
            # create object for player
            pm = PlayerDataManager(self, player, player_attributes_list)
            # add player to player data
            player_data[player_id] = pm

        self.player_data = player_data
    
    def create_team_data(self) -> None:
        # get pandas dataframe from table
        teams = self.get_pandas_df("Team")
        team_attributes = self.get_pandas_df("Team_Attributes")
        team_attributes = team_attributes.sort_values(by="team_api_id")
        # get all matches
        matches = self.get_pandas_df("Match")
        # sort by team id
        teams = teams.sort_values(by="team_api_id")
        # duplicate match data for home and away team
        home_matches = matches.copy()
        away_matches = matches.copy()
        # sort by home team id
        home_matches = home_matches.sort_values(by="home_team_api_id")
        # sort by away team id
        away_matches = away_matches.sort_values(by="away_team_api_id")

        # create dict for storing team data
        team_data = {}
        home_matches_index = 0
        away_matches_index = 0
        team_attributes_index = 0
        # iterate through teams
        for index, team in teams.iterrows():
            # get team id
            team_id = team["team_api_id"]
            team_attributes_list = []
            while(team_attributes_index < len(team_attributes) and team_attributes.iloc[team_attributes_index]["team_api_id"] <= team_id):
                if(team_attributes.iloc[team_attributes_index]["team_api_id"] == team_id):
                    team_attributes_list.append(team_attributes.iloc[team_attributes_index])
                team_attributes_index += 1

            team_attributes_list = pandas.DataFrame(team_attributes_list)
            # create object for team
            tm = TeamManager(self, team, team_attributes_list)
            # get matches for home team
            home_matches_list = []
            # while home matches index is less than rows in home matches and home team id is less than team id
            while(home_matches_index < len(home_matches) and home_matches.iloc[home_matches_index]["home_team_api_id"] <= team_id):
                # if home team id is equal to team id
                if(home_matches.iloc[home_matches_index]["home_team_api_id"] == team_id):
                    # add match to home matches list
                    home_matches_list.append(home_matches.iloc[home_matches_index])
                # increment home matches index
                home_matches_index += 1
                
            # get matches for away team
            away_matches_list = []
            while(away_matches_index < len(away_matches) and away_matches.iloc[away_matches_index]["away_team_api_id"] <= team_id):
                if(away_matches.iloc[away_matches_index]["away_team_api_id"] == team_id):
                    away_matches_list.append(away_matches.iloc[away_matches_index])
                away_matches_index += 1
            # add away matches to home matches
            home_matches_list.extend(away_matches_list)
            # create pandas dataframe from matches
            matches = pandas.DataFrame(home_matches_list)
            matches = matches.sort_values(by="date")
            tm.add_matches(matches)
            # add team to team data
            team_data[team_id] = tm
        self.team_data = team_data
    
    def get_team_data(self, team_id: int) -> Type['TeamManager']:
        '''
        Returns a TeamManager object from team id
        '''
        if(self.team_data == None or team_id not in self.team_data):
            print("Team data not loaded")
            return None
        return self.team_data[team_id]
    
    def get_player(self, player_id) -> Type['PlayerDataManager']:
        if self.player_data == None:
            print("Player data not loaded")
            return None
        if player_id not in self.player_data:
            print(f"Player with id {player_id} not found in database")
            return None
        return self.player_data[player_id]

    
    def get_team_matches(self, team_id: int) -> pandas.DataFrame:
        if(self.team_data == None):
            print("Team data not loaded")
            return None
        return self.team_data[team_id].matches

    def create_train_test_matches(self, train_test_ration=0.8) -> (pandas.DataFrame, pandas.DataFrame):
        '''
        Returns a pandas dataframe of training matches filtered from total matches

        @returns (train_matches, test_matches)
        '''
        # get pandas dataframe from table
        matches = self.get_pandas_df("Match")
        # get list of unique season features from matches
        seasons = matches["season"].unique()
        # get number of seasons
        num_seasons = len(seasons)
        # sort seasons
        seasons.sort()
        # get number of seasons to use for training (round up from train_test_ratio)
        num_train_seasons = int(num_seasons * train_test_ration)
        # get seasons to use for training
        train_seasons = seasons[:num_train_seasons]
        # get seasons to use for testing
        test_seasons = seasons[num_train_seasons:]
        # filter matches for training
        train_matches = matches[matches["season"].isin(train_seasons)]
        # filter matches for testing
        test_matches = matches[matches["season"].isin(test_seasons)]

        # capture list of rows for training and testing
        train_matches = [train_matches.iloc[[i]] for i in range(len(train_matches))]
        test_matches = [test_matches.iloc[[i]] for i in range(len(test_matches))]
        
        train_matches = [MatchDataManager(self, match_data) for match_data in train_matches]
        test_matches = [MatchDataManager(self, match_data) for match_data in test_matches]
        return (train_matches, test_matches)

# player data
class PlayerDataManager:
    def __init__(self, soccer_database, player_data, player_attributes) -> None:
        '''
        Creates a PlayerDataManager object from player id
        '''
        self.sdb = soccer_database
        self.player_data = player_data
        self.player_attributes = player_attributes
        if len(self.player_attributes) > 0:
            self.player_attributes = self.player_attributes.sort_values(by="date")
        self.player_id = player_data["player_api_id"]
    
    def get_player_attributes(self, date, target_feature=None) -> Dict[str, float]:
        '''
        Loads player data from database
        '''
        # there might be multiple entries for the same player, so we need to select the one with the closest date before
        if len(self.player_attributes) == 0:
            return {}
        player_attributes_index = 0
        if type(date) == pandas.Series:
            date = date.iloc[0]
        while(player_attributes_index < len(self.player_attributes) and self.player_attributes.iloc[player_attributes_index]["date"] <= date):
            player_attributes_index += 1
        player_attributes_index -= 1
        player_attributes_index = max(0, player_attributes_index)
        player_attributes = self.player_attributes.iloc[player_attributes_index]

        # get columns from player attributes
        columns = player_attributes.index
        columns_names = [column for column in columns if column != "id" and column != "player_api_id" and column != "date" and column != "player_fifa_api_id"]

        attributes = {}
        for column in columns_names:
            if target_feature is not None and column != target_feature:
                continue
            attributes[column] = player_attributes[column]
        attributes["height"] = self.player_data["height"]
        attributes["weight"] = self.player_data["weight"]
        return attributes

class TeamManager:
    def __init__(self, soccer_database, team_data, team_attributes) -> None:
        '''
        Creates a TeamManager object from team id and date
        '''
        # create soccer database object
        self.sdb = soccer_database
        # get team data from database
        self.team_data = team_data
        self.team_attributes = team_attributes
        if len(self.team_attributes) > 0:
            self.team_attributes = self.team_attributes.sort_values(by="date")
        self.team_id = team_data["team_api_id"]
        # get team attributes from database
        self.matches = None

    def add_matches(self, matches: pandas.DataFrame) -> None:
        self.matches = matches

    def get_team_attributes(self, date, target_features=None) -> Dict[str, float]:
        '''
        Loads team data from database
        '''
        # there might be multiple entries for the same team, so we need to select the one with the closest date before
        if len(self.team_attributes) == 0:
            return {}
        team_attributes_index = 0
        while(team_attributes_index < len(self.team_attributes) and self.team_attributes.iloc[team_attributes_index]["date"] <= date):
            team_attributes_index += 1
        team_attributes_index -= 1
        team_attributes_index = max(0, team_attributes_index)
        team_attributes = self.team_attributes.iloc[team_attributes_index]

        # get columns from team attributes
        columns = team_attributes.index
        columns_names = [column for column in columns if column != "id" and column != "team_api_id" and column != "date"]

        attributes = {}
        for column in columns_names:
            if target_features is not None and column not in target_features:
                continue
            attributes[column] = team_attributes[column]
        return attributes


# class for handling each matches data
class MatchDataManager:
    def __init__(self, soccer_database, match_data) -> None:
        '''
        Creates a MatchDataManager object from soccer database
        '''
        self.sdb = soccer_database
        self.match_data = match_data
        self.match_id = match_data["id"].values[0]
        self.date = match_data["date"].values[0]
        self.match_features = {}
        self.team_attribute_features = {}
        self.player_avg_attribute_features = {}
        self.player_individual_attribute_features = {}
    
    def capture_n_prev_matches(self, team_id: int, date: str, n: int) -> List["MatchDataManager"]:
        # query for matches with team_id
        matches = self.sdb.get_team_matches(team_id)
        # already sorted by date
        # get index of current match
        current_match_index = matches[matches["id"] == self.match_id].index[0]
        # get previous matches
        prev_matches = matches.iloc[:current_match_index]
        # get n previous matches
        n_prev_matches = prev_matches.iloc[-n:]
        # return matches
        return n_prev_matches

    def load_match_data(self, target_team_features=None, target_features_avg=None, target_features_ind=None) -> None:
        '''
        Loads match data from database
        '''
        if len(self.match_data) == 0:
            print(f"Match with id {self.match_id} not found in database")
            return None

        # reset stored features
        self.match_features = {}
        self.team_attribute_features = {}
        self.player_avg_attribute_features = {}
        self.player_individual_attribute_features = {}
        # get match data from database
        date = self.match_data["date"].values[0]
        # get team data
        home_team_id = self.match_data["home_team_api_id"].values[0]
        away_team_id = self.match_data["away_team_api_id"].values[0]

        home_team_data = self.capture_team_data(home_team_id, date)
        away_team_data = self.capture_team_data(away_team_id, date)
        # get team previous match data
        home_team_prev_matches = self.capture_n_prev_matches(home_team_id, date, 5)
        away_team_prev_matches = self.capture_n_prev_matches(away_team_id, date, 5)
        
        home_team_record = 0
        home_team_goal_diff = 0
        home_team_oppentent_rating = 0
        if len(home_team_prev_matches) == 0:
            home_team_record = 0
            home_team_goal_diff = 0
        else:
            # get home team record from previous matches
            home_team_record = [MatchDataManager.get_game_result_for_team(home_team_prev_matches.iloc[match_data_index], home_team_id) for match_data_index in range(len(home_team_prev_matches))]
            home_team_record = sum(home_team_record) / len(home_team_record)
            home_team_goal_diff = [MatchDataManager.get_goal_diff_for_team(home_team_prev_matches.iloc[match_data_index], home_team_id) for match_data_index in range(len(home_team_prev_matches))]
            home_team_goal_diff = sum(home_team_goal_diff) / len(home_team_goal_diff)
            home_team_oppentent_rating = [MatchDataManager.get_avg_player_rating_for_other_team(self.sdb, home_team_prev_matches.iloc[match_data_index], home_team_id) for match_data_index in range(len(home_team_prev_matches))]
            i = 0
            while(i < len(home_team_oppentent_rating)):
                if math.isnan(home_team_oppentent_rating[i]):
                    home_team_oppentent_rating.pop(i)
                else:
                    i += 1
            if len(home_team_oppentent_rating) == 0:
                home_team_oppentent_rating = np.nan
            else:
                home_team_oppentent_rating = sum(home_team_oppentent_rating) / len(home_team_oppentent_rating)
        
        away_team_record = 0
        away_team_goal_diff = 0
        away_team_oppentent_rating = np.nan
        if len(away_team_prev_matches) == 0:
            away_team_record = 0
            away_team_goal_diff = 0
        else:
            # get away team record from previous matches
            away_team_record = [MatchDataManager.get_game_result_for_team(away_team_prev_matches.iloc[match_data_index], away_team_id) for match_data_index in range(len(away_team_prev_matches))]
            away_team_record = sum(away_team_record) / len(away_team_record)
            away_team_goal_diff = [MatchDataManager.get_goal_diff_for_team(away_team_prev_matches.iloc[match_data_index], away_team_id) for match_data_index in range(len(away_team_prev_matches))]
            away_team_goal_diff = sum(away_team_goal_diff) / len(away_team_goal_diff)
            away_team_oppentent_rating = [MatchDataManager.get_avg_player_rating_for_other_team(self.sdb, away_team_prev_matches.iloc[match_data_index], away_team_id) for match_data_index in range(len(away_team_prev_matches))]
            i = 0
            while(i < len(away_team_oppentent_rating)):
                if math.isnan(away_team_oppentent_rating[i]):
                    away_team_oppentent_rating.pop(i)
                else:
                    i += 1
            if len(away_team_oppentent_rating) == 0:
                away_team_oppentent_rating = np.nan
            else:
                away_team_oppentent_rating = sum(away_team_oppentent_rating) / len(away_team_oppentent_rating)
        
        # get team attributes
        home_team_attributes = home_team_data.get_team_attributes(date, target_team_features)
        away_team_attributes = away_team_data.get_team_attributes(date, target_team_features)

        self.match_features["home_team_record"] = home_team_record
        self.match_features["away_team_record"] = away_team_record
        self.match_features["home_team_goal_diff"] = home_team_goal_diff
        self.match_features["away_team_goal_diff"] = away_team_goal_diff
        self.match_features["home_team_oppenent_rating"] = home_team_oppentent_rating
        self.match_features["away_team_oppenent_rating"] = away_team_oppentent_rating

        betting_features = ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA", "LBH", "LBD", "LBA", "WHH", "WHD", "WHA", "VCH", "VCD", "VCA", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]

        for column in self.match_data.columns:
            if column not in betting_features:
                continue
            self.match_features[column] = self.match_data[column].values[0]

        for column in home_team_attributes:
            self.team_attribute_features[f"home_team_{column}"] = home_team_attributes[column]
        for column in away_team_attributes:
            self.team_attribute_features[f"away_team_{column}"] = away_team_attributes[column]

        # get average player attributes
        player_avg_attributes = MatchDataManager.get_avg_player_attribute_for_team(self.sdb, self.match_data, target_features_avg)
        # get player attributes
        player_individual_attributes = MatchDataManager.get_player_attributes_for_teams(self.sdb, self.match_data, target_features_ind)

        for column in player_avg_attributes:
            self.player_avg_attribute_features[column] = player_avg_attributes[column] 
        for column in player_individual_attributes:
            self.player_individual_attribute_features[column] = player_individual_attributes[column]

    def get_features(
            self,
            match_features: List[str],
            team_attribute_features: List[str],
            player_avg_attribute_features: List[str],
            player_individual_attribute_features: List[str]
    ) -> pandas.DataFrame:
        '''
        Returns a numpy array of features for the given match data
        '''
        features = []
        columns = []
        for feature in match_features:
            if f"home_team_{feature}" not in self.match_features:
                features.append(np.nan)
                columns.append(f"home_team_{feature}")
            else:
                features.append(self.match_features[f"home_team_{feature}"])
                columns.append(f"home_team_{feature}")
            if f"away_team_{feature}" not in self.match_features:
                features.append(np.nan)
                columns.append(f"away_team_{feature}")
            else:
                features.append(self.match_features[f"away_team_{feature}"])
                columns.append(f"away_team_{feature}")
        for feature in team_attribute_features:
            if f"home_team_{feature}" not in self.team_attribute_features:
                features.append(np.nan)
                columns.append(f"home_team_{feature}")
            else:
                features.append(self.team_attribute_features[f"home_team_{feature}"])
                columns.append(f"home_team_{feature}")
            if f"away_team_{feature}" not in self.team_attribute_features:
                features.append(np.nan)
                columns.append(f"away_team_{feature}")
            else:
                features.append(self.team_attribute_features[f"away_team_{feature}"])
                columns.append(f"away_team_{feature}")
        for feature in player_avg_attribute_features:
            if f"home_team_avg_{feature}" not in self.player_avg_attribute_features:
                features.append(np.nan)
                columns.append(f"home_team_avg_{feature}")
            else:
                features.append(self.player_avg_attribute_features[f"home_team_avg_{feature}"])
                columns.append(f"home_team_avg_{feature}")
            if f"away_team_avg_{feature}" not in self.player_avg_attribute_features:
                features.append(np.nan)
                columns.append(f"away_team_avg_{feature}")
            else:
                features.append(self.player_avg_attribute_features[f"away_team_avg_{feature}"])
                columns.append(f"away_team_avg_{feature}")
        for feature in player_individual_attribute_features:
            if f"home_team_{feature}" not in self.player_individual_attribute_features:
                for i in range(11):
                    features.append(np.nan)
                    columns.append(f"home_team_player_{i+1}_{feature}")
            else:
                home_team_features = self.player_individual_attribute_features[f"home_team_{feature}"]
                for i in range(len(home_team_features)):
                    features.append(home_team_features[i])
                    columns.append(f"home_team_player_{i+1}_{feature}")
            if f"away_team_{feature}" not in self.player_individual_attribute_features:
                for i in range(11):
                    features.append(np.nan)
                    columns.append(f"away_team_player_{i+1}_{feature}")
            else:
                away_team_features = self.player_individual_attribute_features[f"away_team_{feature}"]
                for i in range(len(away_team_features)):
                    features.append(away_team_features[i])
                    columns.append(f"away_team_player_{i+1}_{feature}")
        features = np.array(features)
        return (features, columns)
        
    def capture_team_data(self, team_id: int, date: str) -> TeamManager:
        '''
        Captures team data from database
        '''
        # return team manager object
        return self.sdb.get_team_data(team_id) 
    
        
    def get_game_result_for_team(match_data, team_id: int):
        home_team_id = match_data["home_team_api_id"]
        away_team_id = match_data["away_team_api_id"]

        if type(home_team_id) == pandas.Series:
            home_team_id = home_team_id.iloc[0]
        if type(away_team_id) == pandas.Series:
            away_team_id = away_team_id.iloc[0]
        
        if type(team_id) == pandas.Series:
            team_id = team_id.iloc[0]

        assert team_id == home_team_id or team_id == away_team_id, "team_id must be either home or away team"
        
        home_goals = match_data["home_team_goal"]
        away_goals = match_data["away_team_goal"]

        if type(home_goals) == pandas.Series:
            home_goals = home_goals.iloc[0]
        if type(away_goals) == pandas.Series:
            away_goals = away_goals.iloc[0]

        if team_id == home_team_id:
            if home_goals > away_goals:
                return 1
            elif home_goals == away_goals:
                return 0
            else:
                return -1
            
        elif team_id == away_team_id:
            if away_goals > home_goals:
                return 1
            elif away_goals == home_goals:
                return 0
            else:
                return -1

    def get_goal_diff_for_team(match_data, team_id: int):
        home_team_id = match_data["home_team_api_id"]
        away_team_id = match_data["away_team_api_id"]

        if type(home_team_id) == pandas.Series:
            home_team_id = home_team_id.iloc[0]
        if type(away_team_id) == pandas.Series:
            away_team_id = away_team_id.iloc[0]

        if type(team_id) == pandas.Series:
            team_id = team_id.iloc[0]

        assert team_id == home_team_id or team_id == away_team_id, "team_id must be either home or away team"
        
        home_goals = match_data["home_team_goal"]
        away_goals = match_data["away_team_goal"]

        if team_id == home_team_id:
            return home_goals - away_goals
            
        elif team_id == away_team_id:
            return away_goals - home_goals

    def get_avg_player_rating_for_other_team(sdb, match_data, team_id):
        averages = MatchDataManager.get_avg_player_attribute_for_team(sdb, match_data)
        home_team_id = match_data["home_team_api_id"]
        if type(home_team_id) != np.int64:
            home_team_id = home_team_id.values[0]
        if team_id == home_team_id:
            if "away_team_avg_overall_rating" not in averages:
                return np.nan
            return averages["away_team_avg_overall_rating"]
        await_team_id = match_data["away_team_api_id"]
        if type(await_team_id) != np.int64:
            await_team_id = await_team_id.values[0]
        elif team_id == await_team_id:
            if "home_team_avg_overall_rating" not in averages:
                return np.nan
            return averages["home_team_avg_overall_rating"]
        else:
            return 0
        
    def get_avg_player_attribute_for_team(sdb, match_data, target_feature=None):
        # get team players
        home_team_players = []
        for i in range(1, 12):
            home_team_players.append(match_data[f"home_player_{i}"])
        away_team_players = []
        for i in range(1, 12):
            away_team_players.append(match_data[f"away_player_{i}"])

        home_team_player_attributes = []
        for home_team_player in home_team_players:
            player_id = home_team_player
            date = match_data["date"]
            if type(home_team_player) == pandas.Series:
                player_id = home_team_player.iloc[0]
            if np.isnan(player_id):
                home_team_player_attributes.append({})
            else:
                home_team_player_attributes.append(sdb.get_player(player_id).get_player_attributes(match_data["date"], target_feature))
        away_team_player_attributes = []
        for away_team_player in away_team_players:
            player_id = away_team_player
            if type(home_team_player) == pandas.Series:
                player_id = home_team_player.iloc[0]
            if np.isnan(player_id):
                away_team_player_attributes.append({})
            else:
                away_team_player_attributes.append(sdb.get_player(player_id).get_player_attributes(match_data["date"], target_feature))

        columns = set()
        for player_attributes in home_team_player_attributes:
            # no string columns
            for column, value in player_attributes.items():
                if value is None or type(value) == str or (isinstance(value, (np.floating, float)) and (np.isnan(value) or math.isnan(value))):
                    continue
                columns.add(column)
        for player_attributes in away_team_player_attributes:
            # no string columns
            for column, value in player_attributes.items():
                if value is None or type(value) == str or (isinstance(value, (np.floating, float)) and (np.isnan(value) or math.isnan(value))):
                    continue
                columns.add(column)
        # if any player does not have a column, set it to NaN
        for player_attributes in home_team_player_attributes:
            for column in columns:
                if column not in player_attributes:
                    player_attributes[column] = np.nan
        for player_attributes in away_team_player_attributes:
            for column in columns:
                if column not in player_attributes:
                    player_attributes[column] = np.nan
        # for each column, get the average for each team
        averages = {}
        for column in columns:
            home_team_avg_player_rating = sum([player_attributes[column] for player_attributes in home_team_player_attributes]) / len(home_team_player_attributes)
            away_team_avg_player_rating = sum([player_attributes[column] for player_attributes in away_team_player_attributes]) / len(away_team_player_attributes)
            averages[f"home_team_avg_{column}"] = home_team_avg_player_rating
            averages[f"away_team_avg_{column}"] = away_team_avg_player_rating

        return averages
    
    def get_player_attributes_for_teams(sdb, match_data, target_feature=None):
        # get team players
        home_team_players = []
        for i in range(1, 12):
            home_team_players.append(match_data[f"home_player_{i}"])
        away_team_players = []
        for i in range(1, 12):
            away_team_players.append(match_data[f"away_player_{i}"])

        has_position_data = True
        # get players y positions
        home_team_player_y_positions = []
        for i in range(1, 12):
            if np.isnan(match_data[f"home_player_Y{i}"].iloc[0]):
                has_position_data = False
                break
            home_team_player_y_positions.append(match_data[f"home_player_Y{i}"].iloc[0])
        away_team_player_y_positions = []
        for i in range(1, 12):
            if not has_position_data:
                break
            if np.isnan(match_data[f"away_player_Y{i}"].iloc[0]):
                has_position_data = False
                break
            away_team_player_y_positions.append(match_data[f"away_player_Y{i}"].iloc[0])

        # convert players from possible series to single value
        for i in range(len(home_team_players)):
            if type(home_team_players[i]) == pandas.Series:
                home_team_players[i] = home_team_players[i].iloc[0]
        for i in range(len(away_team_players)):
            if type(away_team_players[i]) == pandas.Series:
                away_team_players[i] = away_team_players[i].iloc[0]
        
        # sort players by y position
        if has_position_data:
            home_team_players = [player for _, player in sorted(zip(home_team_player_y_positions, home_team_players))]
            away_team_players = [player for _, player in sorted(zip(away_team_player_y_positions, away_team_players))]
        else:
            return {}

        home_team_player_attributes = []
        for home_team_player in home_team_players:
            player_id = home_team_player
            if np.isnan(player_id):
                home_team_player_attributes.append({})
            else:
                home_team_player_attributes.append(sdb.get_player(player_id).get_player_attributes(match_data["date"], target_feature))
        away_team_player_attributes = []
        for away_team_player in away_team_players:
            player_id = away_team_player 
            if np.isnan(player_id):
                away_team_player_attributes.append({})
            else:
                away_team_player_attributes.append(sdb.get_player(player_id).get_player_attributes(match_data["date"], target_feature))
        # get column names from player attributes that appear in any player
        columns = set()
        for player_attributes in home_team_player_attributes:
            # no string columns
            for column, value in player_attributes.items():
                if value is None or type(value) == str or (isinstance(value, (np.floating, float)) and (np.isnan(value) or math.isnan(value))):
                    continue
                columns.add(column)
        for player_attributes in away_team_player_attributes:
            # no string columns
            for column, value in player_attributes.items():
                if value is None or type(value) == str or (isinstance(value, (np.floating, float)) and (np.isnan(value) or math.isnan(value))):
                    continue
                columns.add(column)
        # if any player does not have a column, set it to NaN
        for player_attributes in home_team_player_attributes:
            for column in columns:
                if column not in player_attributes:
                    player_attributes[column] = np.nan
        for player_attributes in away_team_player_attributes:
            for column in columns:
                if column not in player_attributes:
                    player_attributes[column] = np.nan
        data = {}
        for column in columns:
            data[f"home_team_{column}"] = [player_attributes[column] for player_attributes in home_team_player_attributes]
            data[f"away_team_{column}"] = [player_attributes[column] for player_attributes in away_team_player_attributes]
        
        return data
    
# function for getting features from match data
def get_features_for_matches(
    matches,
    match_features: List[str],
    team_attribute_features: List[str],
    player_avg_attribute_features: List[str],
    player_individual_attribute_features: List[str],
    label_type="win_loss",
    include_draw=True
):
    '''
    Returns a numpy array of features for the given matches
    '''
    features = []
    output = []
    columns = []
    for i in range(len(matches)):
        if i % 100 == 0:
            print(f"Processed {i} matches")
        temp2 = matches[i].get_features(match_features, team_attribute_features, player_avg_attribute_features, player_individual_attribute_features)
        temp2, columns = temp2
        # dont consider if contains nan
        if temp2 is None or np.isnan(temp2).any():
            continue
        if label_type == "win_loss":
            temp = MatchDataManager.get_game_result_for_team(matches[i].match_data, matches[i].match_data["home_team_api_id"])
            if temp == 1:
                if include_draw:
                    output.append([1, 0, 0])
                else:
                    output.append(1)
            elif temp == 0:
                if include_draw:
                    output.append([0, 1, 0])
                else:
                    continue
            elif temp == -1:
                if include_draw:
                    output.append([0, 0, 1])
                else:
                    output.append(0)
            else:
                raise ValueError("game result must be either 1, 0, or -1")
        elif label_type == "goal_diff":
            output.append(MatchDataManager.get_goal_diff_for_team(matches[i].match_data, matches[i].match_data["home_team_api_id"]))
        elif label_type == "goal_diff_line":
            goal_diff = MatchDataManager.get_goal_diff_for_team(matches[i].match_data, matches[i].match_data["home_team_api_id"])
        else:
            raise ValueError("label_type must be either win_loss or goal_diff")
        features.append(temp2)

    # convert to pandas dataframe
    return np.array(features), np.array(output), np.array(columns)