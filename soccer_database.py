# Class for loading a looking up data from a sqlite3 database

import pandas
import sqlite3
from typing import List

# create soccer database class
class SoccerDatabase:
    def __init__(self, filename: str) -> None:
        # create connection to database
        self.filename = filename
        self.conn = sqlite3.connect(filename)
        self.cur = self.conn.cursor()

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
        # return training and testing matches
        return (train_matches, test_matches)

class TeamManager:
    def __init__(self, soccer_database: SoccerDatabase, team_id: int, date: str) -> None:
        '''
        Creates a TeamManager object from team id and date
        '''
        self.team_id = team_id
        # use date class to convert between data formats
        self.date = date
        # create soccer database object
        self.sdb = soccer_database
        # get team data from database
        self.team_data = None
        # get team attributes from database
        self.team_attributes = None

    def load_team_data(self) -> None:
        '''
        Loads team data from database
        '''
        # get team data from database and select team with team_id
        self.team_data = self.sdb.get_pandas_df("Team")
        self.team_data = self.team_data[self.team_data["team_api_id"] == self.team_id]
        # get team attributes from database and select team with team_id
        self.team_attributes = self.sdb.get_pandas_df("Team_Attributes")
        self.team_attributes = self.team_attributes[self.team_attributes["id"] == self.team_id]
        # there might be multiple entries for the same team, so we need to select the one with the closest date before
        print(self.team_attributes)
        print(self.date)

# class for handling each matches data
class MatchDataManager:
    def __init__(self, soccer_database: SoccerDatabase, match_id: int) -> None:
        '''
        Creates a MatchDataManager object from soccer database
        '''
        self.match_data = {}
        self.sdb = soccer_database
        self.match_id = match_id

    def load_match_data(self) -> None:
        '''
        Loads match data from database
        '''
        # get match data from database
        match_table = self.sdb.get_pandas_df("Match")
        match_data = match_table[match_table["id"] == self.match_id]
        date = match_data["date"].values[0]
        # get team data
        home_team_id = match_data["home_team_api_id"].values[0]
        away_team_id = match_data["away_team_api_id"].values[0]
        home_team_data = self.capture_team_data(home_team_id, date)
        away_team_data = self.capture_team_data(away_team_id, date)
        # get team previous match data
        home_team_prev_matches = self.capture_n_prev_matches(home_team_id, date, 5)
        away_team_prev_matches = self.capture_n_prev_matches(away_team_id, date, 5)
        
        # get home team record from previous matches
        home_team_record = [match_data.get_game_result_for_team(home_team_id) for match_data in home_team_prev_matches]
        home_team_record = sum(home_team_record) / len(home_team_record)
        # get away team record from previous matches
        away_team_record = [match_data.get_game_result_for_team(away_team_id) for match_data in away_team_prev_matches]
        away_team_record = sum(away_team_record) / len(away_team_record)

    

        
    def capture_team_data(self, team_id: int, date: str) -> TeamManager:
        '''
        Captures team data from database
        '''
        # create team manager object
        team_manager = TeamManager(self.sdb, team_id, date)
        # load team data
        team_manager.load_team_data()
        # return team manager object
        return team_manager
    
    def capture_n_prev_matches(self, team_id: int, date: str, n: int) -> List["MatchDataManager"]:
        # query for matches with team_id
        matches = self.sdb.get_pandas_df("Match")
        matches = matches[(matches["home_team_api_id"] == team_id) | (matches["away_team_api_id"] == team_id)]
        # sort matches by date
        matches = matches.sort_values(by="date")
        # get index of current match
        current_match_index = matches[matches["id"] == self.match_id].index[0]
        # get previous matches
        prev_matches = matches.iloc[:current_match_index]
        # get n previous matches
        n_prev_matches = prev_matches.iloc[-n:]
        # return matches
        return [MatchDataManager(self.sdb, match_id) for match_id in n_prev_matches["id"].values]
    
    def get_game_result_for_team(self, team_id: int):
        home_team_id = self.match_data["home_team_api_id"]
        away_team_id = self.match_data["away_team_api_id"]

        assert team_id == home_team_id or team_id == away_team_id, "team_id must be either home or away team"
        
        home_goals = self.match_data["home_team_goal"]
        away_goals = self.match_data["away_team_goal"]

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