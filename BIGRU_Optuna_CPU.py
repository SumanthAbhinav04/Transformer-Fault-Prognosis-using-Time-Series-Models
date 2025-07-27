#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:49:24 2018

@author: amin
@editors: saeid & qasymjomart
"""

#%% training data
import os
import scipy.io as io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout  # Use GRU instead of CuDNNGRU
from tensorflow.keras.optimizers import RMSprop
import optuna
from optuna import Trial
import matplotlib.pyplot as plt
import math
import pickle
from tensorflow.keras import backend as K

# File and data parameters
file_ind = ['2.5', '3', '3.5', '4', '4.5', '5', 
            '5.5', '6', '7', '8', '8.4', '9',
            '9.5', '10', '11', '12', '13', '15']

Fs = 3000
st = 0.02  # stationary interval in seconds
L = int(st * Fs)  # block length

StartOfSignal = [80000, 45000, 52000, 70000, 70000, 42000, 30000, 34000,
                 50000, 57000, 56000, 75000, 47000, 28000, 50000, 50000,
                 50000, 48000]

#%% Divide data into train, validation, and test sets
data_train_list = []
data_valid_list = []
data_test_list = []

k = -1
for file in file_ind:
    k += 1
    f = io.loadmat('load_current_' + file + 'A.mat')
    a = float(file) * np.ones((len(f['Data1_AI_1']), 1))
    b = np.double(f['Data1_AI_1'])
    a = a[StartOfSignal[k]:]
    b = b[StartOfSignal[k]:]
    N = len(b)
    I = np.floor(N / L) - 1  # total number of observations
    Ntest = int(np.floor(I / 4))  # 1/4 of I for test
    Nvalid = int(np.floor(3 * I / 16))  # validation is 3/16 of I
    Ntrain = int(I - Nvalid - Ntest)
    train_ind_max = Ntrain * L
    valid_ind_max = train_ind_max + Nvalid * L
    test_ind_max = valid_ind_max + Ntest * L

    data_temp_train = np.concatenate((a[0:train_ind_max], b[0:train_ind_max]), axis=1)
    data_temp_valid = np.concatenate((a[train_ind_max:valid_ind_max], b[train_ind_max:valid_ind_max]), axis=1)
    data_temp_test = np.concatenate((a[valid_ind_max:test_ind_max], b[valid_ind_max:test_ind_max]), axis=1)
    data_train_list.append(data_temp_train)
    data_valid_list.append(data_temp_valid)
    data_test_list.append(data_temp_test)

data_train = np.concatenate(data_train_list, axis=0)
data_valid = np.concatenate(data_valid_list, axis=0)
data_test = np.concatenate(data_test_list, axis=0)

#%% Normalize data using mean and std of training data
dmin = data_train.min(axis=0)
dmax = data_train.max(axis=0)
max_min = dmax - dmin
data_train = (data_train - dmin) / max_min
data_valid = (data_valid - dmin) / max_min
data_test = (data_test - dmin) / max_min

#%% Data generator
window = L
step = 1
delay = 0
batch_size = 8

def generator(data, window, delay, min_index, max_index,
              shuffle=False, batch_size=batch_size, step=step):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + window

    while 1:
        if shuffle:
            sample_ind = np.random.randint(min_index, max_index // window, size=batch_size)
            rows = sample_ind * window
        else:
            if i >= max_index:
                i = min_index + window
            rows = np.arange(i, min(i + batch_size * window, max_index), window)
            i = rows[-1] + window
        samples = np.zeros((len(rows), window // step, (data.shape[-1] - 1)))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - window, rows[j], step)
            samples[j] = data[indices, 1:]
            targets[j] = data[rows[j] - 1 + delay][0]
        yield samples, targets

train_gen = generator(data_train, window=window, delay=delay, min_index=0, max_index=None, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(data_valid, window=window, delay=delay, min_index=0, max_index=None, shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(data_test, window=window, delay=delay, min_index=0, max_index=None, step=step, batch_size=batch_size)

val_steps = data_valid.shape[0] // (window * batch_size)
test_steps = data_test.shape[0] // (window * batch_size)

#%% Optuna objective function
def objective(trial):
    # Define hyperparameters to be tuned
    num_layers = trial.suggest_categorical('num_layers', [1, 3, 5, 7])
    num_units = trial.suggest_categorical('num_units', [16, 32, 64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [8, 32])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 1e-4])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  # Dropout rate for overfitting prevention

    # Build model
    model = Sequential()
    for i in range(num_layers):
        return_sequences = (i != num_layers - 1)
        model.add(Bidirectional(GRU(num_units, return_sequences=return_sequences),  # Use GRU instead of CuDNNGRU
                               input_shape=(window // step, 1) if i == 0 else None))
        if return_sequences:  # Add dropout only if there are more layers to come
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='mse', metrics=['mae', 'mape'])

    # Train model
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=150,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    # Return the validation loss
    return min(history.history['val_loss'])

#%% Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Number of trials can be adjusted

# Output the best parameters
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Save the best model
best_model = Sequential()
num_layers = trial.params['num_layers']
num_units = trial.params['num_units']
dropout_rate = trial.params['dropout_rate']

for i in range(num_layers):
    return_sequences = (i != num_layers - 1)
    best_model.add(Bidirectional(GRU(num_units, return_sequences=return_sequences),  # Use GRU instead of CuDNNGRU
                               input_shape=(window // step, 1) if i == 0 else None))
    if return_sequences:
        best_model.add(Dropout(dropout_rate))
best_model.add(Dense(1))

best_model.compile(optimizer=RMSprop(learning_rate=trial.params['learning_rate']), loss='mse', metrics=['mae', 'mape'])
best_model.save('best_model.h5')

# Evaluate the best model
data_test_for_evaluate = data_valid[:, 1:].reshape((len(data_valid) // window, window, 1))
targets_test = data_valid[:, :1].reshape((len(data_valid) // window, window, 1))
predicted_targets = np.zeros((len(data_test_for_evaluate),))
true_targets = np.zeros((len(data_test_for_evaluate),))

for i in range(len(data_test_for_evaluate)):
    true_targets[i] = targets_test[i, window - 1]

for i in range(len(data_test_for_evaluate)):
    sample = np.zeros((1, window // step, (data_valid.shape[-1] - 1)))
    sample[0] = data_test_for_evaluate[i]
    predicted_targets[i] = best_model.predict(sample)

MSE = np.mean((predicted_targets - true_targets) ** 2)
MAE = np.mean(np.abs(predicted_targets - true_targets))
target_mean = np.mean(true_targets)
RRSE = 100 * np.sqrt(MSE * len(true_targets) / np.sum((true_targets - target_mean) ** 2))
RAE = 100 * MAE * len(true_targets) / np.sum(np.abs(true_targets - target_mean))

# Print results in the original format
print('MSE: ', MSE)
print('MAE: ', MAE)
print('RRSE: ', RRSE)
print('RAE: ', RAE)
print('target_mean: ', target_mean)
print('len(true_targets): ', len(true_targets))
print(np.sum((true_targets - target_mean) ** 2))
print(np.sum(np.abs(true_targets - target_mean)) / len(true_targets))

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, np.array(history.history['loss']), 'b--')
plt.plot(epoch_count, np.array(history.history['val_loss']), 'r-')
y = history.history['val_loss']
ymin = min(y)
xpos = y.index(min(y))
xmin = epoch_count[xpos]
y = history.history['val_mae']
yymin = min(y)

print('MSE by formula: ', MSE, ' MSE by model: ', ymin)

string1 = 'MSE = ' + '%.2E' % float(ymin)
string2 = '\n' + 'RAE = ' + str(round(RAE, 2)) + '%' + '\n' + 'RRSE = ' + str(round(RRSE, 2)) + '%'
string = string1 + string2
ax.annotate(string, xy=(xmin, ymin), xycoords='data',
             xytext=(-80, 85), textcoords='offset points',
             bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
             size=12,
             arrowprops=dict(arrowstyle="->"))
plt.title('$\mathit{N}$=' + str(num_layers) + ', $\mathit{M}$=$\mathit{L}$=' + str(num_units))
xint = range(min(epoch_count) - 1, math.ceil(max(epoch_count)), 20)
plt.xticks(xint)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(loc="best")
filename1 = 'best_model_loss'
fig.set_size_inches(5.46, 3.83)
fig.savefig(filename1 + '.pdf', bbox_inches='tight')

# Save scores
score = [ymin, MSE, MAE, RRSE, RAE]
filenameTXT = 'best_model_scores.txt'
np.savetxt(filenameTXT, score)

# Clear session and delete model
K.clear_session()
del best_model