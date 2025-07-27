#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:49:24 2018

@author: amin
@editors: saeid & qasymjomart
"""

#%% Import libraries
import os
import scipy.io as io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna
from optuna import Trial
import matplotlib.pyplot as plt
import math
import pickle
from tensorflow.keras import backend as K

#%% Constants and parameters
file_ind = ['2.5', '3', '3.5', '4', '4.5', '5', 
            '5.5', '6', '7', '8', '8.4', '9',
            '9.5', '10', '11', '12', '13', '15']

Fs = 3000
st = 0.02  # stationary interval in seconds
L = int(st * Fs)  # block length

StartOfSignal = [80000, 45000, 52000, 70000, 70000, 42000, 30000, 34000,
                 50000, 57000, 56000, 75000, 47000, 28000, 50000, 50000,
                 50000, 48000]

#%% Data loading and preprocessing
def load_and_preprocess_data():
    data_train_list = []
    data_valid_list = []
    data_test_list = []

    for k, file in enumerate(file_ind):
        f = io.loadmat(f'load_current_{file}A.mat')
        a = float(file) * np.ones((len(f['Data1_AI_1']), 1))
        b = np.double(f['Data1_AI_1'])
        a = a[StartOfSignal[k]:]
        b = b[StartOfSignal[k]:]
        N = len(b)
        I = np.floor(N / L) - 1
        Ntest = int(np.floor(I / 4))
        Nvalid = int(np.floor(3 * I / 16))
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
    
    # Normalize data
    dmin = data_train.min(axis=0)
    dmax = data_train.max(axis=0)
    max_min = dmax - dmin
    data_train = (data_train - dmin) / max_min
    data_valid = (data_valid - dmin) / max_min
    data_test = (data_test - dmin) / max_min
    
    return data_train, data_valid, data_test

#%% Data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, window, delay, batch_size=8, step=1, shuffle=True):
        self.data = data
        self.window = window
        self.delay = delay
        self.batch_size = batch_size
        self.step = step
        self.shuffle = shuffle
        self.indices = np.arange(len(data) // window)
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        samples = np.zeros((len(batch_indices), self.window // self.step, (self.data.shape[-1] - 1))
        targets = np.zeros((len(batch_indices),))
        
        for i, row in enumerate(batch_indices):
            start_idx = row * self.window
            end_idx = start_idx + self.window
            samples[i] = self.data[start_idx:end_idx:self.step, 1:]
            targets[i] = self.data[end_idx - 1 + self.delay, 0]
            
        return samples, targets
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

#%% Model building
def build_model(trial):
    num_layers = trial.suggest_categorical('num_layers', [1, 3, 5, 7])
    num_units = trial.suggest_categorical('num_units', [16, 32, 64, 128, 256])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 1e-4])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    model = Sequential()
    for i in range(num_layers):
        return_sequences = (i != num_layers - 1)
        model.add(Bidirectional(
            GRU(num_units, return_sequences=return_sequences),
            input_shape=(window // step, 1) if i == 0 else None
        ))
        if return_sequences:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(
        optimizer=RMSprop(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mape']
    )
    return model

#%% Optuna objective function
def objective(trial):
    # Create data generators
    train_gen = DataGenerator(data_train, window, delay, batch_size, step, shuffle=True)
    val_gen = DataGenerator(data_valid, window, delay, batch_size, step, shuffle=False)
    
    # Build and compile model
    model = build_model(trial)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_trial_model.h5', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=500,
        epochs=150,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=0
    )
    
    return min(history.history['val_loss'])

#%% Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data_train, data_valid, data_test = load_and_preprocess_data()
    
    # Training parameters
    window = L
    step = 1
    delay = 0
    batch_size = 8
    val_steps = data_valid.shape[0] // (window * batch_size)
    test_steps = data_test.shape[0] // (window * batch_size)
    
    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Output best parameters
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    
    # Train final model with best parameters
    best_model = build_model(trial)
    best_model.save('best_model.h5')
    
    # Evaluation code remains the same...
    # (Include your evaluation and plotting code here)

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