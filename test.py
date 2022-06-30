import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

plt.ion()
lossarr = []
accuarr = []
sorce = []
mse_sorce1 = []
mse_sorce2 = []
mse_sorce3 = []
mse_sorce4 = []
mse_sorce5 = []
mse_sorce6 = []
mae_sorce1 = []
mae_sorce2 = []
mae_sorce3 = []
mae_sorce4 = []
mae_sorce5 = []
mae_sorce6 = []

r2_sorce1 = []
r2_sorce2 = []
r2_sorce3 = []
r2_sorce4 = []
r2_sorce5 = []
r2_sorce6 = []

best_svr = []
mse_sorce = []
pred_r2_pre = 0
data_x = []
data_y = []

save_path = 'D:\Artificial_intelligence'


def score_calculation(y, y_pred):
    MAE = np.round(mean_absolute_error(y, y_pred), 4)
    RMSE = np.round(np.sqrt(mean_squared_error(y, y_pred)), 4)
    R2_Score = np.round(r2_score(y, y_pred), 4)
    MSE = np.round(mean_squared_error(y, y_pred), 4)

    print('MSE: ' + str(MSE))
    print('MAE: ' + str(MAE))
    print('RMSE: ' + str(RMSE))
    print('R2 Score: ' + str(R2_Score))

    return MSE, MAE, RMSE, R2_Score


def plot_pred(y, y_pred, model_name):
    residuals = y_pred - y
    res_abs = np.abs(residuals)

    th_1 = 0.5  # Define this value in your case
    th_2 = 1  # Define this value in your case
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
    unit_show = "(1e4)"
    plt.text(0.78, 0.058, info_show, ha='left', va='center', transform=ax.transAxes)
    plt.text(-0.07, 1.02, unit_show, ha='left', va='center', transform=ax.transAxes)
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(save_path + '/Pred_' + model_name + '.png')
    plt.show()


def connect_array(pre_array, add_array):
    if pre_array == []:
        pre_array = add_array
    else:
        pre_array = np.concatenate((pre_array, add_array))
    return pre_array


def plot_residuals(y, y_pred, model_name):
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    th_1 = 0.5  # Define this value in your case
    th_2 = 1  # Define this value in your case
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
    unit_show = "(1e4)"
    plt.text(0.0, 0.05, info_show, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.2, 1.02, unit_show, ha='left', va='center', transform=ax.transAxes)
    plt.savefig(save_path + '/Res_' + model_name + '.png')
    plt.show()


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
print(data_y.shape)
data_y = scaley.fit_transform(data_y)
data_y = data_y.reshape(-1, )

kf = KFold(n_splits=10, shuffle=True)
(x_train, x_test), (y_train, y_test) = np.split(data_x, [int(len(data_x) * 0.8)], axis=0), \
                                       np.split(data_y, [int(len(data_y) * 0.8)], axis=0)
# y_array = []
# predict_array = []
#
# ########## Knn ##########
# for train_index, test_index in kf.split(data_x):
#     kng = KNeighborsRegressor(n_neighbors=5)
#     kng.fit(data_x[train_index], data_y[train_index])
#     prediction = kng.predict(data_x[test_index])
#     predict_array = connect_array(predict_array, prediction)
#     y_array = connect_array(y_array, data_y[test_index])
# predict_array = predict_array.reshape(-1, 1)
# predict_array = scaley.inverse_transform(predict_array)
# y_array = y_array.reshape(-1, 1)
# y_array = scaley.inverse_transform(y_array)
# predict_array = predict_array / 10000
# y_array = y_array / 10000
# plot_pred(y_array, predict_array, 'KNN')
# plot_residuals(y_array, predict_array, 'KNN')
# np.savetxt("KNN_predict.txt", predict_array)
# np.savetxt("KNN_test.txt", y_array)
#
# y_array = []
# predict_array = []
# ########## SVR ##########
# for train_index, test_index in kf.split(data_x):
#     poly_svr = SVR(kernel='linear')
#     poly_svr.fit(data_x[train_index], data_y[train_index])
#     poly_predict = poly_svr.predict(data_x[test_index])
#     predict_array = connect_array(predict_array, poly_predict)
#     y_array = connect_array(y_array, data_y[test_index])
# predict_array = predict_array.reshape(-1, 1)
# predict_array = scaley.inverse_transform(predict_array)
# y_array = y_array.reshape(-1, 1)
# y_array = scaley.inverse_transform(y_array)
# predict_array = predict_array / 10000
# y_array = y_array / 10000
# plot_pred(y_array, predict_array, 'SVR')
# plot_residuals(y_array, predict_array, 'SVR')
# np.savetxt("SVR_predict.txt", predict_array)
# np.savetxt("SVR_test.txt", y_array)
#
# y_array = []
# predict_array = []
# ########## Decision Tree ##########
# for train_index, test_index in kf.split(data_x):
#     clf = tree.DecisionTreeRegressor()
#     clf = clf.fit(data_x[train_index], data_y[train_index])
#     predict = clf.predict(data_x[test_index])
#     predict_array = connect_array(predict_array, predict)
#     y_array = connect_array(y_array, data_y[test_index])
# predict_array = predict_array.reshape(-1, 1)
# predict_array = scaley.inverse_transform(predict_array)
# y_array = y_array.reshape(-1, 1)
# y_array = scaley.inverse_transform(y_array)
# predict_array = predict_array / 10000
# y_array = y_array / 10000
# plot_pred(y_array, predict_array, 'Decision Tree')
# plot_residuals(y_array, predict_array, 'Decision Tree')
# np.savetxt("DT_predict.txt", predict_array)
# np.savetxt("DT_test.txt", y_array)
#
y_array = []
predict_array = []
########## Random Forest Regression ##########
for train_index, test_index in kf.split(data_x):
    regression = RandomForestRegressor(n_estimators=10, random_state=0)
    regression.fit(data_x[train_index], data_y[train_index])
    predict = regression.predict(data_x[test_index])
    predict_array = connect_array(predict_array, predict)
    y_array = connect_array(y_array, data_y[test_index])
predict_array = predict_array.reshape(-1, 1)
predict_array = scaley.inverse_transform(predict_array)
y_array = y_array.reshape(-1, 1)
y_array = scaley.inverse_transform(y_array)
predict_array = predict_array / 10000
y_array = y_array / 10000
plot_pred(y_array, predict_array, 'Random Forest Regression')
plot_residuals(y_array, predict_array, 'Random Forest Regression')
np.savetxt("RF_predict.txt", predict_array)
np.savetxt("RF_test.txt", y_array)

