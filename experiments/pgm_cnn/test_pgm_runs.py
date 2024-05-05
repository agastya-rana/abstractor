import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

import tensorflow as tf
import models
import sklearn.metrics
from sklearn.model_selection import train_test_split

import sys; sys.path.append('../'); sys.path.append('../..')
import autoregressive_abstractor
import utils
from eval_utils import evaluate_seq2seq_model, log_to_wandb
# set up W&B logging
import wandb

wandb.login()
import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
wandb_project_name = args.wandb_project_name


## TODO: Change monitor function here?
def create_callbacks(monitor='val_loss', patience=50):
    callbacks = [wandb.keras.WandbMetricsLogger(log_freq='epoch'),]
    if True:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode='max', restore_best_weights=True))
    return callbacks


## Use standard accuracy metric
metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()
fit_kwargs = {'epochs': 10, 'batch_size': 512}

#region Dataset
seed = 0
data_path = ""
eval_task_data = np.load(data_path)
X, y = eval_task_data["X"], eval_task_data["y"]
test_size = 0.2
val_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=seed)
#endregion

abstractor_cnn_kwargs = dict( 
            #num_layers, num_kernels, kernel_size, stride, input_size, dropout_rate=0.1, name="cnn_encoder", dropout_in_cnn=False, mlp=[], **kwargs
    encoder_kwargs=dict(encoder_type='cnn', num_layers=4, num_kernels=64, kernel_size=3, stride=2, mlp=[64]),
    abstractor_kwargs=dict(
            num_layers=2,
            dff=64, ## feedforward hidden layer dimension
            rel_dim=4, ## dimension of relation tensor (i.e. how many heads)
            symbol_dim=64, ## model dimension
            proj_dim=8, ## dimension of key and query projection
            symmetric_rels=False,
            encoder_kwargs=dict(use_bias=True),
            rel_activation_type='softmax',
            use_self_attn=False,
            use_layer_norm=False,
            dropout_rate=0.2),
)

from models import AbstractorCNNModel
pgm_model = AbstractorCNNModel(**abstractor_cnn_kwargs)
pgm_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
pgm_model((X_train[:32], y_train[:32])) ## this is to call "build"

train_size = 10000
sample_idx = np.random.choice(len(X_train), train_size, replace=False)
X_trains = X_train[sample_idx]
y_trains = y_train[sample_idx]
history = pgm_model.fit(X_trains, y_trains, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
wandb.finish(quiet=True)
del model