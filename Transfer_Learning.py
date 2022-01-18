import time

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, Add, Flatten, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from skfuzzy.membership import trimf
from niapy.task import Task, OptimizationType
from niapy.problems import Problem
from niapy.algorithms.basic import OppositionVelocityClampingParticleSwarmOptimization
from sklearn.model_selection import KFold
save_path = 'D:\Artificial_intelligence'


def score_calculation(y, y_pred):
    MAE = np.round(mean_absolute_error(y, y_pred), 5)
    RMSE = np.round(np.sqrt(mean_squared_error(y, y_pred)), 5)
    R2_Score = np.round(r2_score(y, y_pred), 5)
    MSE = np.round(mean_squared_error(y, y_pred), 5)

    print('MSE: ' + str(MSE))
    print('MAE: ' + str(MAE))
    print('RMSE: ' + str(RMSE))
    print('R2 Score: ' + str(R2_Score))

    return MSE, MAE, RMSE, R2_Score


def plot_pred(y, y_pred, model_name):
    residuals = y_pred - y
    res_abs = np.abs(residuals)

    th_1 = 0.025  # Define this value in your case
    th_2 = 0.05  # Define this value in your case
    r1_idx = np.where(res_abs <= th_1)
    r2_idx = np.where((res_abs > th_1) & (res_abs <= th_2))
    r3_idx = np.where(res_abs > th_2)
    # Calculate Error
    MSE, MAE, RMSE, R2_Score = score_calculation(y, y_pred)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    #    plt.scatter(y, y_pred, color='b', alpha=0.15, s=40)

    plt.scatter(y[r1_idx], y_pred[r1_idx], c='royalblue',
                alpha=0.15, s=40, label=r'|R|$\leq$' + str(th_1))
    plt.scatter(y[r2_idx], y_pred[r2_idx], c='yellowgreen',
                alpha=0.15, s=40, label=str(th_1) + r'<|R|$\leq$' + str(th_2))
    plt.scatter(y[r3_idx], y_pred[r3_idx], c='orange',
                alpha=0.15, s=40, label='|R|>' + str(th_2))

    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
    plt.title(model_name + ' Prediction Results')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    info_show = 'MSE: ' + str(MSE) + '\n' \
                                     'MAE: ' + str(MAE) + '\n' \
                                                          'RMSE: ' + str(RMSE) + '\n' \
                                                                                 'R2 Score: ' + str(R2_Score) + '\n'
    plt.text(0.7, 0.09, info_show, ha='left', va='center', transform=ax.transAxes)
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(save_path + '/Pred_' + model_name + '.png')


def plot_history(y, model_name):
    plt.plot(y.history['loss'], label='loss')
    # plt.plot(y.history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.savefig(save_path + '/loss_' + model_name + '.png')


def plot_residuals(y: object, y_pred: object, model_name: object) -> object:
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    th_1 = 0.025  # Define this value in your case
    th_2 = 0.05  # Define this value in your case
    r1_idx = np.where(res_abs <= th_1)
    r2_idx = np.where((res_abs > th_1) & (res_abs <= th_2))
    r3_idx = np.where(res_abs > th_2)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    grid = plt.GridSpec(4, 4, wspace=0.5, hspace=0.5)

    main_ax = plt.subplot(grid[0:3, 1:4])
    #    plt.plot(y_pred, residuals,'ok',markersize=3,alpha=0.2)
    plt.scatter(y_pred[r1_idx], residuals[r1_idx], c='royalblue',
                alpha=0.15, s=40, label=r'|R|$\leq$' + str(th_1))
    plt.scatter(y_pred[r2_idx], residuals[r2_idx], c='yellowgreen',
                alpha=0.15, s=40, label=str(th_1) + r'<|R|$\leq$' + str(th_2))
    plt.scatter(y_pred[r3_idx], residuals[r3_idx], c='orange',
                alpha=0.15, s=40, label='|R|>' + str(th_2))
    plt.plot([y_pred.min(), y_pred.max()], [0, 0], 'r--', lw=1.5)
    plt.legend(loc='upper left')
    plt.grid()
    plt.title('Residuals for ' + model_name + ' Model')

    y_hist = plt.subplot(grid[0:3, 0], xticklabels=[], sharey=main_ax)
    plt.hist(residuals, 60, orientation='horizontal', color='g')
    y_hist.invert_xaxis()
    plt.ylabel('Residuals')
    plt.grid()

    x_hist = plt.subplot(grid[3, 1:4], yticklabels=[], sharex=main_ax)
    plt.hist(y_pred, 60, orientation='vertical', color='g')
    x_hist.invert_yaxis()
    plt.xlabel('Predicted Value')
    plt.grid()

    MSE, MAE, RMSE, R2_Score = score_calculation(y, y_pred)
    info_show = 'MSE: ' + str(MSE) + '\n' \
                                     'MAE: ' + str(MAE) + '\n' \
                                                          'RMSE: ' + str(RMSE) + '\n' \
                                                                                 'R2 Score: ' + str(R2_Score) + '\n'
    plt.text(0.0, 0.05, info_show, ha='left', va='center', transform=ax.transAxes)

    plt.savefig(save_path + '/Res_' + model_name + '.png')


def membership_function(data, ranges):
    zz = None
    data_fuzzy = np.copy(data)
    for j in range(len(data_fuzzy[0])):
        data_array = []
        for i in data_fuzzy[:, j]:
            if i <= ranges[0]:
                data_array.append([1, 0, 0])
            elif ranges[0] < i < ranges[1]:
                data_array.append(
                    [((0 - 1) / (ranges[1] - ranges[0])) * (i - ranges[0]) + 1, ((1 - 0) / (ranges[1] - ranges[0])) *
                     (i - ranges[0]), 0])
            elif i == ranges[1]:
                data_array.append([0, 1, 0])
            elif ranges[1] < i < ranges[2]:
                data_array.append(
                    [0, ((0 - 1) / (ranges[2] - ranges[1])) * (i - ranges[1]) + 1, ((1 - 0) / (ranges[2] - ranges[1])) *
                     (i - ranges[1])])
            else:
                data_array.append([0, 0, 1])
        if zz is None:
            zz = data_array
        else:
            zz = np.hstack((zz, data_array))
    zz1 = np.copy(zz)
    return zz1


def trensfer_learning_model(pre_x, pre_y, train_x_t, train_y_t):
    base_model = Sequential()
    inputs = Input(shape=(24, 1))

    x = Conv1D(filters=96, kernel_size=3, activation='relu')(inputs)
    x = MaxPool1D(1)(x)
    x = Conv1D(filters=96, kernel_size=3, activation='relu')(x)
    x = MaxPool1D(1)(x)
    x = Conv1D(filters=96, kernel_size=3, activation='relu')(x)
    x = MaxPool1D(1)(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)


    y = Dense(72, activation='relu')(inputs)
    y = Dropout(0.2)(y)
    y = Dense(72, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Flatten()(y)
    z = Add()([x, y])
    z = Flatten()(z)
    z = Dense(8, activation='relu')(z)
    outputs = Dense(1, activation='relu')(z)

    base_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    base_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    base_model.fit(pre_x, pre_y, batch_size=32, epochs=100, verbose=0)

    for layer in base_model.layers:
        layer.trainable = False

    base_model1 = base_model.layers[-2].output
    x = base_model1
    x = Dense(8)(x)
    x = Dense(4)(x)
    x = Dense(2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=base_model.inputs, outputs=outputs)

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(train_x_t, train_y_t, epochs=100, batch_size=32, verbose=0)

    for layer in model.layers:
        layer.trainable = True
    initial_learning_rate = 1e-5
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='mse', metrics=["mse"])

    return model


df = pd.read_csv('bmw.csv')
labelencoder = LabelEncoder()
df['model'] = labelencoder.fit_transform(df['model'])
df['transmission'] = labelencoder.fit_transform(df['transmission'])
df['fuelType'] = labelencoder.fit_transform(df['fuelType'])

data1 = df.values
data1 = data1.astype('float')
data_x1, data_y, data_x2 = np.hsplit(data1, [2, 3])
data_x = np.hstack([data_x1, data_x2])
scalex = MinMaxScaler(feature_range=(0, 1))
data_x = scalex.fit_transform(data_x)
scaley = MinMaxScaler(feature_range=(0, 1))
data_y = scaley.fit_transform(data_y)
data_y = data_y.reshape(-1, )

c = 0
def run(sol):
    tf.keras.backend.clear_session()
    global train_x, train_y, c
    c += 1
    data_x_fuzzy = membership_function(train_x, [sol[0], sol[0] + sol[1], sol[0] + sol[1] + sol[2]])
    pre_x_r, train_x_r, test_x_r = data_x_fuzzy[0: int(len(data_x_fuzzy) * 0.5)], \
                                   data_x_fuzzy[int(len(data_x_fuzzy) * 0.5): int(len(data_x_fuzzy) * 0.8)], \
                                   data_x_fuzzy[int(len(data_x_fuzzy) * 0.8)::]
    pre_y_r, train_y_r, test_y_r = train_y[0: int(len(train_y) * 0.5)], \
                                   train_y[int(len(train_y) * 0.5): int(len(train_y) * 0.8)], \
                                   train_y[int(len(train_y) * 0.8)::]
    model = trensfer_learning_model(pre_x_r, pre_y_r, train_x_r, train_y_r)

    his = model.fit(train_x_r, train_y_r, batch_size=32, epochs=500, validation_data=[test_x_r, test_y_r], verbose=0)
    predict = model.predict(test_x_r)
    score = r2_score(test_y_r, predict)
    print(c)
    return score, model, his


class My_Problem(Problem):
    def __init__(self, dimension, lower=0, upper=1):
        Problem.__init__(self, dimension, lower, upper)
        self.best_score = 0
        self.best_history = []
        self.best_model = []

    def _evaluate(self, sol):
        val, model_n, his = run(sol)
        if self.best_score < val:
            self.best_score = val
            self.best_model = model_n
            self.best_history = his
        return val


my_problem = My_Problem(dimension=3)
iteration = 20
particle = 5
kf = KFold(n_splits=10, shuffle=True)

for train_idx, test_idx in kf.split(data_x):
    global train_x, train_y
    train_x, test_x = data_x[train_idx], data_x[test_idx]
    train_y, test_y = data_y[train_idx], data_y[test_idx]

    task = Task(problem=my_problem, max_iters=iteration, optimization_type=OptimizationType.MAXIMIZATION,
                enable_logging=True)
    algo = OppositionVelocityClampingParticleSwarmOptimization(population_size=particle)
    best = algo.run(task)

    train_data_fuzzy = membership_function(train_x, best[0])
    test_data_fuzzy = membership_function(test_x, best[0])
    pre_x, train_x = train_data_fuzzy[0: int(len(train_data_fuzzy) * 0.8)], \
                     train_data_fuzzy[int(len(train_data_fuzzy) * 0.8)::]
    pre_y, train_y = train_y[0: int(len(train_y) * 0.8)], train_y[int(len(train_y) * 0.8)::]

    base_model = Sequential()
    inputs = Input(shape=(24, 1))

    x = Conv1D(filters=96, kernel_size=3, activation='relu')(inputs)
    x = MaxPool1D(1)(x)
    x = Conv1D(filters=96, kernel_size=3, activation='relu')(x)
    x = MaxPool1D(1)(x)
    x = Conv1D(filters=96, kernel_size=3, activation='relu')(x)
    x = MaxPool1D(1)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    y = Dense(72, activation='relu')(inputs)
    y = Dropout(0.2)(y)
    y = Dense(72, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Flatten()(y)
    z = Add()([x, y])
    z = Flatten()(z)
    z = Dense(8, activation='relu')(z)
    outputs = Dense(1, activation='relu')(z)

    base_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    base_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    base_model.fit(pre_x, pre_y, batch_size=32, epochs=100, verbose=1)

    for layer in base_model.layers:
        layer.trainable = False

    base_model1 = base_model.layers[-2].output
    x = base_model1
    x = Dense(8)(x)
    x = Dense(4)(x)
    x = Dense(2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=base_model.inputs, outputs=outputs)

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=1)

    for layer in model.layers:
        layer.trainable = True
    initial_learning_rate = 1e-5
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='mse', metrics=["mse"])
    his = model.fit(train_x, train_y, batch_size=32, epochs=500, validation_data=[test_x, test_y], verbose=1)
    predict = model.predict(test_x)
    print(predict.shape)
    print(test_y.shape)
    plot_history(his, 'Transfer_learning')
    plot_pred(test_y, predict, 'Transfer_learning')
    plot_residuals(test_y, predict, 'Transfer_learning')
