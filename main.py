import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Add, Conv1D
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

plt.ion()
lossarr = []
accuarr = []
sorce = []
cvsorce1 = []
cvsorce2 = []
cvsorce3 = []
cvsorce4 = []
cvsorce5 = []
data_x = []
data_y = []
df = pd.read_csv('bmw.csv')
data = df.values
data = data.astype(float)
for i in range(len(data)):
    data_x.append((data[i, 0], data[i, 1], data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8]))
    data_y.append(data[i, 2])
data_x = np.asarray(data_x)
data_y = np.asarray(data_y)
scale = MinMaxScaler(feature_range=(0, 1))
data_x = scale.fit_transform(data_x)
kf = KFold(n_splits=10, shuffle=True)
# (x_train, x_test), (y_train, y_test) = np.split(data_x, [int(len(data_x) * 0.8)], axis=0), \
#                                        np.split(data_y, [int(len(data_y) * 0.8)], axis=0)

########## Knn ##########
for train_index, test_index in kf.split(data_x):
    kng = KNeighborsRegressor(n_neighbors=5)
    kng.fit(data_x[train_index], data_y[train_index])
    prediction = kng.predict(data_x[test_index])
    mape = mean_absolute_percentage_error(data_y[test_index], prediction)
    cvsorce1.append(mape)

########## SVR ##########
for train_index, test_index in kf.split(data_x):
    poly_svr = SVR(kernel='poly')
    poly_svr.fit(data_x[train_index], data_y[train_index])
    poly_predict = poly_svr.predict(data_x[test_index])
    mape = mean_absolute_percentage_error(data_y[test_index], poly_predict)
    cvsorce2.append(mape)

########## Api_Function ##########
input_shape = (None, 8)
batch_size = 1
kernel_size = 3
pool_size = 2
dropout = 0.1
filters = 64
kernel_size1 = 3
pool_size1 = 1
filters1 = 32

for train_index, test_index in kf.split(data_x):
    model = Sequential()

    inputs = Input(shape=(8,))
    x = Dense(16, activation='relu')(inputs)
    x = Dropout(dropout)(x)

    y = Dense(128, activation='relu')(inputs)
    y = Dropout(dropout)(y)
    y = Dense(16, activation='relu')(y)
    y = Dropout(dropout)(y)

    z = Add()([x, y])
    z = Flatten()(z)
    z = Dropout(dropout)(z)
    outputs = Dense(1, activation='relu')(z)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mape'])
    history = model.fit(data_x[train_index], data_y[train_index], batch_size=1, epochs=100, verbose=1)
    sorce = model.evaluate(data_x[test_index], data_y[test_index], batch_size=1, verbose=0)
    cvsorce3.append(sorce)

########## Decision Tree ##########
for train_index, test_index in kf.split(data_x):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(data_x[train_index], data_y[train_index])
    pre = clf.predict(data_x[test_index])
    mape = mean_absolute_percentage_error(data_y[test_index], pre)
    cvsorce4.append(mape)

########## Random Forest Regression ##########
for train_index, test_index in kf.split(data_x):
    regression = RandomForestRegressor(n_estimators=10, random_state=0)
    regression.fit(data_x[train_index], data_y[train_index])
    pred = regression.predict(data_x[test_index])
    mape = mean_absolute_percentage_error(data_y[test_index], pred)
    cvsorce5.append(mape)

print("Knn = ", np.mean(cvsorce1))
print('SVR = ', np.mean(cvsorce2))
print('Api_Function = ', np.mean(cvsorce3))
print('Decision Tree = ', np.mean(cvsorce4))
print(' Random Forest Regression = ', np.mean(cvsorce5))
