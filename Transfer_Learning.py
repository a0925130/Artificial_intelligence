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

global data_x, data_y

pre_x = []
pre_y = []
train_x = []
train_y = []
test_x = []
test_y = []
df = pd.read_csv('bmw.csv')
labelencoder = LabelEncoder()
df['model'] = labelencoder.fit_transform(df['model'])
df['transmission'] = labelencoder.fit_transform(df['transmission'])
df['fuelType'] = labelencoder.fit_transform(df['fuelType'])



data = df.values
data = data.astype('float')
data_x1, data_y, data_x2 = np.hsplit(data, [2, 3])
data_x = np.hstack([data_x1, data_x2])
scalex = MinMaxScaler(feature_range=(0, 1))
data_x = scalex.fit_transform(data_x)
scaley = MinMaxScaler(feature_range=(0, 1))
data_y = scaley.fit_transform(data_y)
data_y = data_y.reshape(-1, )


def run(sol):
    global data_x, data_y
    for i in range(8):
        data_x[i, :] = trimf(data_x[i, :], [sol[0], sol[0] + sol[1], sol[0] + sol[1] + sol[2]])

    pre_x = data_x[0: 5000]
    pre_y = data_y[0: 5000]

    train_x = data_x[5000: 9000]
    train_y = data_y[5000: 9000]

    # test_train_x = data_x[0: 9000]
    # test_train_y = data_y[0: 9000]

    test_x = data_x[9000:]
    test_y = data_y[9000:]

    base_model = Sequential()

    inputs = Input(shape=(8, 1))

    x = Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
    x = MaxPool1D(1)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPool1D(1)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPool1D(1)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    y = Dense(32, activation='relu')(inputs)
    y = Dropout(0.2)(y)
    y = Dense(32, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Flatten()(y)
    z = Add()([x, y])
    z = Flatten()(z)
    z = Dense(8, activation='relu')(z)
    outputs = Dense(1, activation='relu')(z)

    base_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    base_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    base_model.fit(pre_x, pre_y, batch_size=32, epochs=10, verbose=1)

    for layer in base_model.layers:
        layer.trainable = False

    base_model1 = base_model.layers[-2].output
    x = base_model1
    x = Dense(8)(x)
    x = Dense(4)(x)
    x = Dense(2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=base_model.inputs, outputs=outputs)
    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)

    # for layer in model.layers[:3]:
    #     layer.trainable = False
    # for layer in model.layers[3:]:
    #     layer.trainable = True*
    for layer in model.layers:
        layer.trainable = True

    model.summary()
    initial_learning_rate = 1e-5
    # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='mse', metrics=["mse"])
    history = model.fit(train_x, train_y, batch_size=32, epochs=100, validation_data=[test_x, test_y], verbose=1)
    predict = model.predict(test_x)
    score = r2_score(test_y, predict)
    return score, history, predict, test_y


class My_Problem(Problem):
    def __init__(self, dimension, lower=0, upper=100):
        Problem.__init__(self, dimension, lower, upper)
        self.best_score = 0
        self.best_history = []
        self.best_predict = []
        self.test = []
    def _evaluate(self, sol):
        val, history, predict, test = run(sol)
        self.test = np.copy(test)
        if self.best_score < val:
            self.best_score = val
            self.best_history = np.copy(history)
            self.best_predict = np.copy(predict)
        return val


my_problem = My_Problem(dimension=3)
iteration = 20
particle = 5

task = Task(problem=my_problem, max_iters=iteration, optimization_type=OptimizationType.MAXIMIZATION,
            enable_logging=True)
algo = OppositionVelocityClampingParticleSwarmOptimization(population_size=particle)
best = algo.run(task)

# plot_history(my_problem.best_history, 'Transfer_learning')
plot_pred(my_problem.test, my_problem.best_predict, 'Transfer_learning')
plot_residuals(my_problem.test, my_problem.best_predict, 'Transfer_learning')