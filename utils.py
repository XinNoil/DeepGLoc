from __future__ import absolute_import 
from __future__ import print_function
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)

def get_font(fontsize=15):
    font = {
        # 'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize,
    }
    return font
        
# define euclidean loss
def euclidean_error(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

def cdfplot(data,fontsize=15,xlim=None,xlabel='Error distance (m)',ylabel='CDF',is_new=True,legend_str='',linestyle='-',color=None):
    if is_new:
        plt.figure()
    data_num=data.shape[0]
    num_bins = 1000
    counts, bin_edges = np.histogram(data, bins=num_bins)
    counts=counts/data_num
    cdf = np.cumsum(counts)
    if not color:
        h,=plt.plot(bin_edges[1:], cdf,label=legend_str,linestyle=linestyle)
    else:
        h,=plt.plot(bin_edges[1:], cdf,label=legend_str,linestyle=linestyle,color=color)
    plt.xlabel(xlabel,get_font(fontsize))
    plt.ylabel(ylabel,get_font(fontsize))
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    if not xlim:
        plt.xlim(np.min(data),np.max(data))
    plt.ylim((0, 1.1))
    plt.tick_params(labelsize=fontsize)
    return h

def get_prefix(args):
    line='_'
    prefix = line.join((args.dataname,args.model,'e'+str(args.epoch),'b'+str(args.batchsize)))
    if hasattr(args,'gen'):
        if args.gen:
            prefix = line.join((prefix,'gen'))
    return prefix