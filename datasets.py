import math
import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self,dataname,val=False, **kwargs):
        self.dataname = dataname
        self.val = val
        if 'train_features' in kwargs:
            self.train_features = kwargs['train_features']
            print('train_features: ',self.train_features.shape)
        if 'train_labels' in kwargs:
            self.train_labels = kwargs['train_labels']
            print('train_labels: ',self.train_labels.shape)
        if 'val_features' in kwargs:
            self.val_features = kwargs['val_features']
            print('val_features: ',self.val_features.shape)
        if 'val_labels' in kwargs:
            self.val_labels = kwargs['val_labels']
            print('val_labels: ',self.val_labels.shape)
        if 'test_features' in kwargs:
            self.test_features = kwargs['test_features']
            print('test_features: ',self.test_features.shape)
        if 'test_labels' in kwargs:
            self.test_labels = kwargs['test_labels']
            print('test_labels: ',self.test_labels.shape)
    
    def __len__(self):
        return len(self.train_features)
    
    def get_training(self):
        if not hasattr(self,'train_features'):
            raise Exception('Please specify train data!')
        if hasattr(self,'train_labels'):
            return (self.train_features, self.train_labels)
        else:
            return self.train_features
    
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
    filename='data/'+args.dataname+'_training.csv'
        
    dataframe = pd.read_csv(filename, header=None)
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    dataset = dataframe.values
    train_features = dataset[:,4:]
    train_labels = dataset[:,:2]
    
    filename2='data/'+args.dataname+'_testing.csv'
    dataframe2 = pd.read_csv(filename2, header=None)
    dataframe2 = dataframe2.reindex(np.random.permutation(dataframe2.index))
    dataset2 = dataframe2.values
    val_num = math.ceil(dataset2.shape[0]/3)
    if hasattr(args,'gen'):
        if args.gen:
            filename3='gen_data.csv'
            dataframe3 = pd.read_csv(filename3, header=None)
            dataframe3 = dataframe3.reindex(np.random.permutation(dataframe3.index))
            dataset3 = dataframe3.values
            train_features = np.vstack((train_features,dataset3[:,2:]))
            train_labels = np.vstack((train_labels,dataset3[:,:2]))
            print('add_gen: ',dataset3.shape)
    dset = Dataset(
        dataname=args.dataname,
        val = True,
        train_features = train_features, 
        train_labels = train_labels,
        val_features = dataset2[:val_num,4:],
        val_labels = dataset2[:val_num,:2],
        test_features = dataset2[val_num:,4:], 
        test_labels = dataset2[val_num:,:2],
    )
    return dset