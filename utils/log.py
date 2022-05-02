import os

import pandas as pd


class TrainLog():
    """ logs all training stats for later use
    """

    def __init__(self):
        self.metadata = {}
        self.dataframe = pd.DataFrame()


    def add_model_data(self, model):
        """
        """
        pass


    def add_config_data(self, config):
        """
        """
        pass

    
    def add_metrics(self, **args):
        """
        """
        pass

    
    def export(self, dname, dpath):
        """
        """
        pass


class TestLog():
    """ logs all testing stats for later use
    """

    def __init__(self):
        pass