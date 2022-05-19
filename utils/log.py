import os
import json

import pandas as pd


class TrainLog():
    """ logs all training stats for later use
    """

    def __init__(self, title, dirpath):
        self.metadata = {}
        self.dataframe = pd.DataFrame()

        # create dir
        self.dir = os.path.join(dirpath, title)

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        # create csvs
        with open(os.path.join(self.dir, 'train-log.csv'), 'w'):
            pass
        with open(os.path.join(self.dir, 'valid-log.csv'), 'w'):
            pass

        self.train_counter = 0
        self.valid_counter = 0


    def add_model_data(self, model):
        """"""
        self.metadata['ENCODER_CLASS'] = str(type(model.encoder))
        self.metadata['DECODER_CLASS'] = str(type(model.decoder))

        return


    def add_config_data(self, config):
        """"""
        for attr in dir(config):
            if attr[0].isupper():
                val = getattr(config, attr)

                if not isinstance(val, (str, int)):
                    self.metadata[attr] = str(val)
                else:
                    self.metadata[attr] = val

        # write to dir
        with open(os.path.join(self.dir, 'config.json'), 'w') as f:
            json.dump(self.metadata, f)
        
        return


    def add_metrics(self, name, **kwargs):
        """"""
        fname = f'{name}-log.csv'

        if ((self.train_counter == 0) != (self.valid_counter == 0)) or \
            (self.train_counter == self.valid_counter == 0):
            header = ','.join([str(key) for key in kwargs.keys()]) + '\n'

            with open(os.path.join(self.dir, fname), 'a') as f:
                f.write(header)
        
        row = ','.join([str(val) for val in kwargs.values()]) + '\n'

        with open(os.path.join(self.dir, fname), 'a') as f:
            f.write(row)

        if name == 'train':
            self.train_counter += 1
        elif name == 'valid':
            self.valid_counter += 1

        return


class TestLog():
    """ logs all testing stats for later use
    """

    def __init__(self):
        pass