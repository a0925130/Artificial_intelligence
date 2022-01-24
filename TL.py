import time
import numpy as np

np.set_printoptions(threshold=np.inf)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from niapy.task import Task, OptimizationType
from niapy.problems import Problem
from sklearn.neighbors import KNeighborsRegressor
from niapy.algorithms.basic import GreyWolfOptimizer
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold

save_path = r"C:\Users\SingYan\PycharmProjects\Artificial_intelligence"


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


def plot_residuals(y, y_pred, model_name):
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



def run(sol):
    global train_x, train_y
    data_x_fuzzy = membership_function(train_x, [sol[0], sol[0] + sol[1], sol[0] + sol[1] + sol[2]])
    pre_x_r, train_x_r, test_x_r = data_x_fuzzy[0: int(len(data_x_fuzzy) * 0.5)], \
                                   data_x_fuzzy[int(len(data_x_fuzzy) * 0.5): int(len(data_x_fuzzy) * 0.8)], \
                                   data_x_fuzzy[int(len(data_x_fuzzy) * 0.8)::]
    pre_y_r, train_y_r, test_y_r = train_y[0: int(len(train_y) * 0.5)], \
                                   train_y[int(len(train_y) * 0.5): int(len(train_y) * 0.8)], \
                                   train_y[int(len(train_y) * 0.8)::]
    input_shape = (2,)
    RF = RandomForestRegressor()
    RF.fit(pre_x_r, pre_y_r)
    RF_predict = np.array(RF.predict(train_x_r))
    KNN = KNeighborsRegressor()
    KNN.fit(pre_x_r, pre_y_r)
    KNN_predict = np.array(KNN.predict(train_x_r))
    nn_train = np.stack((RF_predict, KNN_predict))
    nn_train = nn_train.T
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape))
    for i in range(sol[3].astype('int64')):
        model.add(Dense(16))
        model.add(Dropout(0.1))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(nn_train, train_y_r, epochs=20, batch_size=32, verbose=1)
    RF_predict_r = np.array(RF.predict(test_x_r))
    KNN_predict_r = np.array(KNN.predict(test_x_r))
    nn_test = np.stack((RF_predict_r, KNN_predict_r))
    nn_test = nn_test.T
    predict = model.predict(nn_test)
    score = r2_score(test_y_r, predict)
    tf.keras.backend.clear_session()
    return score


class My_Problem(Problem):
    def __init__(self, dimension, lower=[0, 0, 0, 1], upper=[1, 1, 1, 100]):
        Problem.__init__(self, dimension, lower, upper)
        self.best_score = 0
        self.fitness_array = [0]
        self.count = 0
        self.count_array = [0]
    def _evaluate(self, sol):
        val = run(sol)
        if self.best_score < val:
            self.best_score = val
        self.count += 1
        self.fitness_array.append(self.best_score)
        self.count_array.append(self.count)
        return val



iteration = 20
particle = 5
kf = KFold(n_splits=10, shuffle=True)
count_kfold = 0
fitness_arr = []
count_arr = []
for train_idx, test_idx in kf.split(data_x):
    global train_x, train_y
    train_x, test_x = data_x[train_idx], data_x[test_idx]
    train_y, test_y = data_y[train_idx], data_y[test_idx]
    my_problem = My_Problem(dimension=4)
    task = Task(problem=my_problem, max_iters=iteration, optimization_type=OptimizationType.MAXIMIZATION,
                enable_logging=True)
    algo = GreyWolfOptimizer(population_size=particle)
    best = algo.run(task)
    if count_kfold == 0:
        fitness_arr = np.copy(my_problem.fitness_array)
        count_arr = np.copy(my_problem.count_array)
    else:
        for i in range(len(fitness_arr)):
            fitness_arr[i] += my_problem.fitness_array[i]
    train_data_fuzzy = membership_function(train_x, [best[0][0], best[0][0]+best[0][1], best[0][0]+best[0][1]+best[0][2]])
    test_data_fuzzy = membership_function(test_x, [best[0][0], best[0][0]+best[0][1], best[0][0]+best[0][1]+best[0][2]])
    pre_x, train_x = train_data_fuzzy[0: int(len(train_data_fuzzy) * 0.8)], \
                     train_data_fuzzy[int(len(train_data_fuzzy) * 0.8)::]
    pre_y, train_y = train_y[0: int(len(train_y) * 0.8)], train_y[int(len(train_y) * 0.8)::]

    input_shape = (2,)
    RF = RandomForestRegressor()
    RF.fit(pre_x, pre_y)
    RF_predict = np.array(RF.predict(train_x))
    KNN = KNeighborsRegressor()
    KNN.fit(pre_x, pre_y)
    KNN_predict = np.array(KNN.predict(train_x))
    nn_train = np.stack((RF_predict, KNN_predict))
    nn_train = nn_train.T
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape))
    for i in range(best[0][3].astype('int64')):
        model.add(Dense(16))
        model.add(Dropout(0.1))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    his = model.fit(nn_train, train_y, epochs=20, batch_size=32, verbose=1)
    RF_predict_r = np.array(RF.predict(test_data_fuzzy))
    KNN_predict_r = np.array(KNN.predict(test_data_fuzzy))
    nn_test = np.stack((RF_predict_r, KNN_predict_r))
    nn_test = nn_test.T
    predict = model.predict(nn_test)
    predict = predict.reshape(-1, )
    np.savez('TL_data' + str(count_kfold), history=np.array(his.history), predict=predict, test_data=test_y)
    plot_history(his, 'Transfer_learning' + str(count_kfold))
    plot_pred(test_y, predict, 'Transfer_learning' + str(count_kfold))
    plot_residuals(test_y, predict, 'Transfer_learning' + str(count_kfold))
    count_kfold += 1
    if count_kfold >= 3:
        break

fitness_arr = fitness_arr/3
plt.plot(count_arr, fitness_arr, linewidth=1)
np.save('TL_fitness_array', fitness_arr)
plt.xlabel('Evaluation')
plt.ylabel('Fitness Value')
plt.title('Learning Curve')
plt.grid()
plt.legend()
plt.show()
plt.savefig(save_path + '/Learning_Curve.png')