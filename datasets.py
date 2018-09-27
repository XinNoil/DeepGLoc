import math
import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self, train_features, val=False, **kwargs):
        self.train_features = train_features
        if 'train_labels' in kwargs:
            self.train_labels = kwargs['train_labels']
        if 'val_features' in kwargs:
            self.val_features = kwargs['val_features']
        if 'val_labels' in kwargs:
            self.val_labels = kwargs['val_labels']
        self.val = val
    
    def __len__(self):
        return len(self.train_features)
    
    def get_validation(self):
        if not hasattr(self,'val_features'):
            raise Exception('Please specify validation data!')
        if hasattr(self,'val_labels'):
            return (self.val_features, self.val_labels)
        else:
            return self.val_features
            
    
    def get_input_shape(self):
        return (self.train_features.shape[1],)
    
    def get_output_shape(self):
        if self.train_labels is None:
            return None
        else:
            return self.train_labels.shape[1]
    
def load_data(args):
    if args.model=='dnn':
        filename='data/55-4-W_training.csv'
        dataframe = pd.read_csv(filename, header=None)
        dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
        dataset = dataframe.values
        val_num = math.ceil(dataset.shape[0]/3)
        dset = Dataset(dataset[val_num:,4:], 
            train_labels = dataset[val_num:,:2],
            val_features = dataset[:val_num,4:],
            val_labels = dataset[:val_num,:2],
            val = True
        )
    elif args.model=='vae':
        filename='data/55-4-W_training.csv'
        dataframe = pd.read_csv(filename, header=None)
        dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
        dataset = dataframe.values
        val_num = math.ceil(dataset.shape[0]/3)
        dset = Dataset(dataset[val_num:,4:], val_features=dataset[:val_num,4:], val=True)
    elif args.model=='cvae':
        filename='data/55-4-W_training.csv'
        dataframe = pd.read_csv(filename, header=None)
        dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
        dataset = dataframe.values
        val_num = math.ceil(dataset.shape[0]/3)
        dset = Dataset(dataset[val_num:,4:], 
            train_labels = dataset[val_num:,:2],
            val_features = dataset[:val_num,4:],
            val_labels = dataset[:val_num,:2],
            val = True
        )
    else:
        raise Exception('Unknown model:', args.model)
    return dset