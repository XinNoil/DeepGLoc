from keras import backend as K

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)
        
# define euclidean loss
def euclidean_error(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)