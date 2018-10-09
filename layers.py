from keras.engine.topology import Layer
from keras import backend as K
class SampleNormal(Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_avg, z_log_var):
        batch_size = K.shape(z_avg)[0]
        z_dims = K.shape(z_avg)[1]
        eps = K.random_normal(shape=K.shape(z_avg), mean=0.0, stddev=1.0)
        return z_avg + K.exp(z_log_var / 2.0) * eps

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        return self._sample_normal(z_avg, z_log_var)

class VAELossLayer(Layer):
    __name__ = 'vae_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_avg, z_log_var):
        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
        return rec_loss + kl_loss

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        z_avg = inputs[2]
        z_log_var = inputs[3]
        loss = self.lossfun(x_true, x_pred, z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return x_true

class VAERLossLayer(Layer):
    __name__ = 'vaer_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.alpha = kwargs['alpha']
        super(VAERLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_avg, z_log_var, y_true, y_pred):
        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
        ed_loss = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
        return rec_loss + kl_loss + self.alpha*ed_loss

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        z_avg = inputs[2]
        z_log_var = inputs[3]
        y_true = inputs[4]
        y_pred = inputs[5]
        loss = self.lossfun(x_true, x_pred, z_avg, z_log_var, y_true, y_pred)
        self.add_loss(loss, inputs=inputs)

        return x_true