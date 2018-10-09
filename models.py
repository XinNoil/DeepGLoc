import os,sys,time
from abc import ABCMeta, abstractmethod
import numpy as np
import utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Activation
from keras.optimizers import Adam
import layers

class BaseModel(metaclass=ABCMeta):
    '''
    Base model class
    '''

    def __init__(self, **kwargs):
        '''
        Initialization
        '''
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs['name']

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')

        self.input_shape = kwargs['input_shape']

        if 'save' not in kwargs:
            self.save = False

        if 'output' not in kwargs:
            self.output = 'tmp'
        else:
            self.output = kwargs['output']

        self.out_dir = os.path.join('output',self.output)
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        self.trainers = {}
    
    def main_loop(self, datasets, epochs=100, batchsize=100, reporter=[], validation=False):
        '''
        Main learning loop
        '''
        # Start training
        print('\n\n--- START TRAINING ---\n\n')
        num_data = len(datasets)
        self.on_train_begin()
        for e in range(epochs):
            perm = np.random.permutation(num_data)
            start_time = time.time()
            for b in range(0, num_data, batchsize):
                bsize = min(batchsize, num_data - b)
                indx = perm[b:b+bsize]

                # Print current status
                ratio = 100.0 * (b + bsize) / num_data
                print(chr(27) + "[2K", end='')
                print('\rEpoch #%d | %d / %d (%6.2f %%) ' % \
                      (e + 1, b + bsize, num_data, ratio), end='')

                # Get batch and train on it
                x_batch = self.make_batch(datasets, indx)
                losses = self.train_on_batch(x_batch)
                self.report_logs(reporter, losses)

                # Compute ETA
                elapsed_time = time.time() - start_time
                eta = elapsed_time / (b + bsize) * (num_data - (b + bsize))
                print('| ETA: %s ' % utils.time_format(eta), end='')

                sys.stdout.flush()
            train_logs=self.train_on_batch(datasets.get_training())
            val_logs=[]
            if validation:
                val_logs = self.train_on_batch(datasets.get_validation())
                print('\tvalidation ', end='')
                self.report_logs(reporter,val_logs)
                print('\n')
            _logs=(train_logs,val_logs)
            self.on_epoch_end(e,_logs)

    def get_filename(self,prefix,filename,ext):
        return os.path.join(self.out_dir, '%s_%s.%s' % (prefix,filename,ext))

    def save_model(self):
        if not self.save:
            return

        for k, v in self.trainers.items():
            filename = os.path.join(self.out_dir, '%s.hdf5' % (k))
            v.save_weights(filename)

    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            getattr(self, k).load_weights(filename)
    
    def report_logs(self, reporter, losses):
        for k in reporter:
                if k in losses:
                    print('| %s = %8.6f ' % (k, losses[k]), end='')

    @abstractmethod
    def on_train_begin(self):
        '''
        Plase override "on_train_begin" method in the derived model!
        '''
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, _logs):
        '''
        Plase override "on_epoch_end" method in the derived model!
        '''
        pass

    @abstractmethod
    def make_batch(self, datasets, indx):
        '''
        Plase override "make_batch" method in the derived model!
        '''
        pass

    @abstractmethod
    def train_on_batch(self, x_batch):
        '''
        Plase override "train_on_batch" method in the derived model!
        '''
        pass
    
    @abstractmethod
    def predict(self, x_data):
        '''
        Plase override "predict" method in the derived model!
        '''
        pass

class BaseDNN(BaseModel):
    def __init__(self, name='dnn', layer_units=[128,64,16], dropouts=[0,0], output_shape=None, **kwargs):
        super(BaseDNN, self).__init__(name=name, **kwargs)
        self.layer_units = layer_units
        self.dropouts = dropouts
        self.output_shape = output_shape
        self.dnn = self.build_model()
        self.store_to_save('dnn')

    def build_model(self):
        input_x = Input(shape=self.input_shape, name='input')
        x = Dropout(self.dropouts[0])(input_x)

        for n in self.layer_units[:-1]:
            x = Dense(n, kernel_initializer='uniform', activation='relu')(x)
        x = Dropout(self.dropouts[1])(x)
        x = Dense(self.layer_units[-1], kernel_initializer='uniform', activation='tanh')(x)

        output_y = Dense(self.output_shape, kernel_initializer='uniform', name='output')(x)
        dnn = Model(input_x, output_y)
        dnn.compile(loss=[utils.euclidean_error], optimizer='adadelta')
        dnn.summary()
        
        return dnn
    
    def on_train_begin(self):
        self.logs = {'train':[], 'val':[]}

    def on_epoch_end(self, epoch, _logs):
        train_logs, val_logs = _logs
        self.logs['train'].append(train_logs['loss'])
        self.logs['val'].append(val_logs['loss'])

    def make_batch(self, datasets, indx):
        x_true = datasets.train_features[indx]
        y_true = datasets.train_labels[indx]
        return x_true,y_true

    def train_on_batch(self, batch):
        x_true, y_true = batch
        loss = self.dnn.train_on_batch(x_true, y_true)
        return { 'loss': loss }
        
    def predict(self, x_data):
        return self.dnn.predict(x_data)

class VAE(BaseModel):
    def __init__(self, name='vae', layer_units=[128,64], z_dims=16, **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.layer_units = layer_units
        self.z_dims = z_dims

        self.f_enc = self.build_encoder()
        self.f_dec = self.build_decoder()
        self.vae_trainer = self.build_model()
        self.store_to_save('vae_trainer')

    def build_model(self):
        x_true = Input(shape=self.input_shape)
        z_avg, z_log_var = self.f_enc(x_true)
        z = layers.SampleNormal()([z_avg, z_log_var])
        x_pred = self.f_dec(z)
        vae_loss = layers.VAELossLayer()([x_true, x_pred, z_avg, z_log_var])

        vae_trainer = Model(inputs=[x_true], outputs=[vae_loss])
        vae_trainer.compile(loss=[utils.zero_loss], optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        vae_trainer.summary()
        return vae_trainer

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)
        x = Dense(self.layer_units[0], kernel_initializer='uniform', activation='relu')(inputs)
        for n in self.layer_units[1:]:
            x = Dense(n, kernel_initializer='uniform', activation='relu')(x)
        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)

        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)
        encoder = Model(inputs, [z_avg, z_log_var])
        return encoder

    def build_decoder(self):
        layer_units=self.layer_units.copy()
        layer_units.reverse()
        inputs = Input(shape=(self.z_dims,))
        x = Dense(layer_units[0], kernel_initializer='uniform', activation='relu')(inputs)
        for n in layer_units[1:]:
            x = Dense(n, kernel_initializer='uniform', activation='relu')(x)
        x = Dense(self.input_shape[0])(x)
        return Model(inputs, x)
    
    def on_train_begin(self):
        self.logs = {'train':[], 'val':[]}

    def on_epoch_end(self, epoch, _logs):
        train_logs, val_logs = _logs
        self.logs['train'].append(train_logs['loss'])
        self.logs['val'].append(val_logs['loss'])

    def make_batch(self, datasets, indx):
        x_true = datasets.train_features[indx]
        return x_true

    def train_on_batch(self, batch):
        x_true = batch
        loss = self.vae_trainer.train_on_batch(x_true,x_true)
        return { 'loss': loss }
        
    def predict(self, x_data):
        return self.vae_trainer.predict(x_data)

    def encode(x_data):
        return self.f_enc.predict(x_data)
    
    def decode(z_sample):
        return self.f_dec.predict(z_sample)

class CVAE(BaseModel):
    def __init__(self, name='cvae', layer_units=[128,64], y_dims=2, z_dims=16, **kwargs):
        super(CVAE, self).__init__(name=name, **kwargs)
        self.layer_units = layer_units
        self.y_dims = y_dims
        self.z_dims = z_dims

        self.f_enc = self.build_encoder()
        self.f_dec = self.build_decoder()
        self.cvae_trainer = self.build_model()
        self.store_to_save('cvae_trainer')

    def build_model(self):
        x_true = Input(shape=self.input_shape)
        y_true = Input(shape=(self.y_dims,))
        z_avg, z_log_var = self.f_enc([x_true,y_true])
        z = layers.SampleNormal()([z_avg, z_log_var])
        x_pred = self.f_dec([z,y_true])
        vae_loss = layers.VAELossLayer()([x_true, x_pred, z_avg, z_log_var])

        cvae_trainer = Model(inputs=[x_true,y_true], outputs=[vae_loss])
        cvae_trainer.compile(loss=[utils.zero_loss], optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        cvae_trainer.summary()
        return cvae_trainer

    def build_encoder(self):
        x_true = Input(shape=self.input_shape)
        y_true = Input(shape=(self.y_dims,))
        inputs=keras.layers.concatenate([x_true, y_true])
        x = Dense(self.layer_units[0], kernel_initializer='uniform', activation='relu')(inputs)
        for n in self.layer_units[1:]:
            x = Dense(n, kernel_initializer='uniform', activation='relu')(x)
        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)

        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)
        encoder = Model([x_true,y_true], [z_avg, z_log_var])
        return encoder

    def build_decoder(self):
        layer_units=self.layer_units.copy()
        layer_units.reverse()
        z = Input(shape=(self.z_dims,))
        y_true = Input(shape=(self.y_dims,))
        inputs=keras.layers.concatenate([z, y_true])
        x = Dense(layer_units[0], kernel_initializer='uniform', activation='relu')(inputs)
        for n in layer_units[1:]:
            x = Dense(n, kernel_initializer='uniform', activation='relu')(x)
        x = Dense(self.input_shape[0])(x)
        return Model([z,y_true], x)
    
    def on_train_begin(self):
        self.logs = {'train':[], 'val':[]}

    def on_epoch_end(self, epoch, _logs):
        train_logs, val_logs = _logs
        self.logs['train'].append(train_logs['loss'])
        self.logs['val'].append(val_logs['loss'])

    def make_batch(self, datasets, indx):
        x_true = datasets.train_features[indx]
        y_true = datasets.train_labels[indx]
        return [x_true,y_true]

    def train_on_batch(self, batch):
        x_true,y_true = batch
        xy_true = [x_true,y_true]
        loss = self.cvae_trainer.train_on_batch(xy_true,x_true)
        return { 'loss': loss }
        
    def predict(self, xy_data):
        return self.cvae_trainer.predict(xy_data)

    def encode(self, xy_data):
        return self.f_enc.predict(xy_data)
    
    def decode(self, zy_sample):
        return self.f_dec.predict(zy_sample)

    def decodey(self, y, resample_num=1):
        y_sample = y.repeat(resample_num,axis=0)
        z_sample = np.random.standard_normal((y_sample.shape[0],self.z_dims))
        return self.decode([z_sample,y_sample]),y_sample

class CVAER(BaseModel):
    def __init__(self, name='cvaer', layer_units=[128,64], y_dims=2, z_dims=16, **kwargs):
        super(CVAE, self).__init__(name=name, **kwargs)
        self.layer_units = layer_units
        self.y_dims = y_dims
        self.z_dims = z_dims

        self.f_enc = self.build_encoder()
        self.f_dec = self.build_decoder()
        self.f_reg = self.build_regressor()
        self.cvaer_trainer = self.build_model()
        self.store_to_save('cvaer_trainer')
    
    def build_model(self):
        x_true = Input(shape=self.input_shape)
        y_true = Input(shape=(self.y_dims,))
        z_avg, z_log_var = self.f_enc([x_true,y_true])
        z = layers.SampleNormal()([z_avg, z_log_var])
        x_pred = self.f_dec([z,y_true])
        y_pred = self.f_reg(x_true)
        vaer_loss = layers.VAERLossLayer()([x_true, x_pred, z_avg, z_log_var, y_true, y_pred])

        cvae_trainer = Model(inputs=[x_true,y_true], outputs=[vae_loss])
        cvae_trainer.compile(loss=[utils.zero_loss], optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        cvae_trainer.summary()
        return cvae_trainer
    
    def build_encoder(self):
        x_true = Input(shape=self.input_shape)
        y_true = Input(shape=(self.y_dims,))
        inputs=keras.layers.concatenate([x_true, y_true])
        x = Dense(self.layer_units[0], kernel_initializer='uniform', activation='relu')(inputs)
        for n in self.layer_units[1:]:
            x = Dense(n, kernel_initializer='uniform', activation='relu')(x)
        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)

        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)
        encoder = Model([x_true,y_true], [z_avg, z_log_var])
        return encoder

    def build_decoder(self):
        layer_units=self.layer_units.copy()
        layer_units.reverse()
        z = Input(shape=(self.z_dims,))
        y_true = Input(shape=(self.y_dims,))
        inputs=keras.layers.concatenate([z, y_true])
        x = Dense(layer_units[0], kernel_initializer='uniform', activation='relu')(inputs)
        for n in layer_units[1:]:
            x = Dense(n, kernel_initializer='uniform', activation='relu')(x)
        x = Dense(self.input_shape[0])(x)
        return Model([z,y_true], x)
    
    def make_batch(self, datasets, indx):
        x_true = datasets.train_features[indx]
        y_true = datasets.train_labels[indx]
        return [x_true,y_true]

    def train_on_batch(self, batch):
        x_true,y_true = batch
        xy_true = [x_true,y_true]
        loss = self.cvae_trainer.train_on_batch(xy_true,x_true)
        return { 'loss': loss }

    def predict(self, xy_data):
        return self.cvae_trainer.predict(xy_data)

    def encode(self, xy_data):
        return self.f_enc.predict(xy_data)
    
    def decode(self, zy_sample):
        return self.f_dec.predict(zy_sample)

models = {
    'dnn': BaseDNN,
    'vae': VAE,
    'cvae': CVAE,
    'cvaer': CVAER
}