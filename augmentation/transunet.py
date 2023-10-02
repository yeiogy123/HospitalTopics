import os
import sys
import tensorflow as tf
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras.layers import (Activation,
                          Reshape)

from tensorflow.keras import backend as K

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pre_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pre_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pre_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pre_pos)
    alpha = 0.7
    return (true_pos + 1) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

code_dir = os.getcwd()
sys.path.append(code_dir)
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
lr = 1e-6
decay = 1e-6

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
sess = tf.compat.v1.Session(config=config)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, decay=decay)

    model = transunet_2d((256,256, 1), filter_num=[64, 128, 256, 512, 1024],n_labels=2)
    top_model = multi_gpu_model(model, gpus=8)
    top_model.compile(optimizer=optimizer, loss=focal_tversky, metrics=['accuracy'], sample_weight_mode='temporal')

    sess.graph.finalize()
    tf.compat.v1.keras.backend.set_session(sess)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, mode='auto',
                                           epsilon=0.0001,
                                           cooldown=0,
                                           min_lr=0)

