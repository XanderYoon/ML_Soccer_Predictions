import helpers
import numpy as np
import pandas as pd

class OddsPredictor():
    """
    This class represents a predictor model using Linear Regression trained on betting odds data. Given betting odds from various providers for a given match, it can predict if the home team is likely to win with ~70% accuracy.
    """

    def __init__(self, data, binary_class_label = helpers.binary_class_label):
        """
        Initialize this prediction model with the dataset to train.
        """
        self.binary_class_label = binary_class_label
        self.odds_features = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']
        self.odds_data = data[self.odds_features + [self.binary_class_label]].dropna()
        self.model_lr = None
        self.top_k_features = None

    def train(self):
        """
        Trains model on all features, picks best features to use
        (the ones that minimize train error), and retrains.
        """
        lr = helpers.logistic_regression(self.odds_data, self.odds_features, suppress_print=True)
        xtrain = lr["train_data"][self.odds_features]
        ytrain = lr["train_data"][self.binary_class_label]
        features_by_importance = helpers.get_sorted_importances(lr, self.odds_features)
        self.top_k_features = helpers.find_top_k_performing_important_features(self.odds_data, features_by_importance)[0]
        lr_top_k = helpers.logistic_regression(self.odds_data[self.top_k_features + [self.binary_class_label]], self.top_k_features)
        self.model_lr = lr_top_k
        print(self.top_k_features)

    def evaluators(self):
        if self.model_lr:
            return self.model_lr["evaluators"]
        return None

    def will_home_win(self, betting_odds):
        """
        Return True if the home team is likely going to win

        Takes in a dictionary of betting odds.
        """
        for key in betting_odds.keys():
            if key not in self.top_k_features:
                print(f"Warning: Odd Provider {key} Not In Top Providers")
                return None
        for key in self.top_k_features:
            if key not in betting_odds.keys():
                print(f"Warning: Odd Provider {key} Not In Top Providers")
                return None

        formatted = {}
        for key, value in betting_odds.items():
            formatted[key] = [value]
        xtrain = pd.DataFrame(formatted)
        return bool(self.model_lr["model"].predict(xtrain) == 1)