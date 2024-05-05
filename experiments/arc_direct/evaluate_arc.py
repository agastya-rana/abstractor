# Procedurally Generated Matrices (PGM) dataset evaluation script
# We use the PGM dataset to evaluate the (relational) reasoning capability of abstractor models
# We benchmark abstractor architectures with CNN encoders against transformer, CNN and WReN baseline models
# For abstractor, we test two schemes:
# 1. Wild Abstractor - 8 context panels + 1 target panel outputs a score for each of 8 target panels; answer chosen by softmax
# 2. Complete Abstractor - 8 context panels + 8 target panels forced to output a single result at once
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

import tensorflow as tf

import sklearn.metrics
from sklearn.model_selection import train_test_split

import sys; sys.path.append('../'); sys.path.append('../..')
import autoregressive_abstractor
import utils
from eval_utils import evaluate_seq2seq_model, log_to_wandb
# region Parser

seed = None

## Descriptions of models:
## 'transformer' is baseline transformer model
## 'wild_abstractor_reln' is the abstractor model using 8 context + 1 target in forward pass
## 'abstractor_reln' is the abstractor model using architecture (b); enc -> abstr -> dec
## 'abstractor_part_reln' is the abstractor model using architecture (d); enc -> abstr; [enc, abstr] -> dec

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
    choices=('transformer', 'wild_abstractor_reln', 'abstractor_reln', 'abstractor_part_reln'),
    help='the model to evaluate on')
parser.add_argument('--n_epochs', default=500, type=int, help='number of epochs to train each model for')
parser.add_argument('--early_stopping', default=True, type=bool, help='whether to use early stopping')
## TODO: change this default based on PGM size
parser.add_argument('--min_train_size', default=500, type=int, help='minimum training set size')
parser.add_argument('--max_train_size', default=5000, type=int, help='maximum training set size')
parser.add_argument('--train_size_step', default=50, type=int, help='training set step size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name', default='abstractor-pgm-cnn', type=str, help='W&B project name')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())
#assert len(tf.config.list_physical_devices('GPU')) > 0

# set up W&B logging
import wandb
wandb.login()
import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
wandb_project_name = args.wandb_project_name


## TODO: Change monitor function here?
def create_callbacks(monitor='val_teacher_forcing_accuracy'):
    callbacks = [wandb.keras.WandbMetricsLogger(log_freq='epoch'),]
    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=50, mode='max', restore_best_weights=True))
    return callbacks

from transformer_modules import TeacherForcingAccuracy
teacher_forcing_acc_metric = TeacherForcingAccuracy(ignore_class=None)
metrics = [teacher_forcing_acc_metric]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()
fit_kwargs = {'epochs': args.n_epochs, 'batch_size': 512}

#region Dataset
eval_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

## TODO: change for PGM
objects, seqs, sorted_seqs, object_seqs, target, labels, start_token = (eval_task_data['objects'], eval_task_data['seqs'], eval_task_data['sorted_seqs'], eval_task_data['object_seqs'], \
    eval_task_data['target'], eval_task_data['labels'], eval_task_data['start_token'])

test_size = 0.2
val_size = 0.1

seqs_train, seqs_test, sorted_seqs_train, sorted_seqs_test, object_seqs_train, object_seqs_test, target_train, target_test, labels_train, labels_test = train_test_split(
    seqs, sorted_seqs, object_seqs, target, labels, test_size=test_size, random_state=seed)
seqs_train, seqs_val, sorted_seqs_train, sorted_seqs_val, object_seqs_train, object_seqs_val, target_train, target_val, labels_train, labels_val = train_test_split(
    seqs_train, sorted_seqs_train, object_seqs_train, target_train, labels_train, test_size=val_size/(1-test_size), random_state=seed)

seqs_length = seqs.shape[1]

source_train, source_val, source_test = object_seqs_train, object_seqs_val, object_seqs_test
#endregion

# region Model kwargs 
transformer_kwargs = dict(
    num_layers=4, num_heads=2, dff=64, 
    input_vocab='vector', target_vocab=seqs_length+1,
    output_dim=seqs_length, embedding_dim=64)

abstractor_reln_kwargs = dict(
        encoder_kwargs=dict(type='cnn', num_layers=2, num_heads=4, dff=64, dropout_rate=0.1),
        abstractor_kwargs=dict(
            num_layers=2,
            dff=64,
            rel_dim=4,
            symbol_dim=64,
            proj_dim=8,
            symmetric_rels=False,
            encoder_kwargs=dict(use_bias=True),
            rel_activation_type='softmax',
            use_self_attn=False,
            use_layer_norm=False,
            dropout_rate=0.2),
        decoder_kwargs=dict(num_layers=1, num_heads=4, dff=64, dropout_rate=0.1),
        input_vocab='vector',
        target_vocab=seqs_length+1,
        embedding_dim=64,
        output_dim=seqs_length,
        abstractor_type='abstractor',
        abstractor_on='encoder',
        decoder_on='abstractor',
        name='abstractor_reln')

# endregion

max_train_size = args.max_train_size
train_size_step = args.train_size_step
min_train_size = args.min_train_size
train_sizes = np.arange(min_train_size, max_train_size+1, step=train_size_step)

num_trials = args.num_trials # num of trials per train set size
start_trial = args.start_trial

print(f'Will evaluate learning curve for `train_sizes` from {min_train_size} to {max_train_size} in increments of {train_size_step}.')
print(f'will run {num_trials} trials for each of the {len(train_sizes)} training set sizes for a total of {num_trials * len(train_sizes)} trials')

def evaluate_learning_curves(create_model, group_name, 
    source_train=source_train, target_train=target_train, labels_train=labels_train,
    source_val=source_val, target_val=target_val, labels_val=labels_val,
    source_test=source_test, target_test=target_test, labels_test=labels_test,
    train_sizes=train_sizes, num_trials=num_trials):
    ## TODO: This needs to change to just source, no target
    for train_size in tqdm(train_sizes, desc='train size'):
        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name})
            model = create_model()
            sample_idx = np.random.choice(len(source_train), train_size, replace=False)
            X_train = source_train[sample_idx], target_train[sample_idx]
            y_train = labels_train[sample_idx]
            X_val = source_val, target_val
            y_val = labels_val

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)

            ## TODO: edit this for our case here.
            def evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=False):
                n = len(source_test)
                output = np.zeros(shape=(n, (seqs_length+1)), dtype=int)
                output[:,0] = start_token
                for i in range(seqs_length):
                    predictions = model((source_test, output[:, :-1]), training=False)
                    predictions = predictions[:, i, :]
                    predicted_id = tf.argmax(predictions, axis=-1)
                    output[:,i+1] = predicted_id

                elementwise_acc = (np.mean(output[:,1:] == labels_test))
                acc_per_position = [np.mean(output[:, i+1] == labels_test[:, i]) for i in range(seqs_length)]
                seq_acc = np.mean(np.all(output[:,1:]==labels_test, axis=1))


                teacher_forcing_acc = teacher_forcing_acc_metric(labels_test, model([source_test, target_test])).numpy()
                teacher_forcing_acc_metric.reset_state()

                if print_:
                    print('element-wise accuracy: %.2f%%' % (100*elementwise_acc))
                    print('full sequence accuracy: %.2f%%' % (100*seq_acc))
                    print('teacher-forcing accuracy:  %.2f%%' % (100*teacher_forcing_acc))


                return_dict = {
                    'elementwise_accuracy': elementwise_acc, 'full_sequence_accuracy': seq_acc,
                    'teacher_forcing_accuracy': teacher_forcing_acc, 'acc_by_position': acc_per_position
                    }

                return return_dict

            eval_dict = evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=False)

            ## TODO: edit this too in the same way
            log_to_wandb(model, eval_dict)
            wandb.finish(quiet=True)

            del model

# endregion


# region Define create_models
# transformer
if args.model == 'transformer':
    def create_model():
        argsort_model = seq2seq_abstracter_models.Transformer(
            **transformer_kwargs)

        argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        argsort_model((source_train[:32], target_train[:32]));

        return argsort_model
    
    group_name = 'Transformer'

# Autoregressive Abstractor
elif args.model == 'abstractor_reln':
    # standard evaluation
    def create_model():
        argsort_model = autoregressive_abstractor.AutoregressiveAbstractor(**autoreg_abstractor_kwargs)

        argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        argsort_model((source_train[:32], target_train[:32]));

        return argsort_model
    
    group_name = 'Abstractor'
# endregion

# region Evaluate Learning Curves
utils.print_section("EVALUATING LEARNING CURVES")
evaluate_learning_curves(create_model, group_name=group_name)
# endregion



