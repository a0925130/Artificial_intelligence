import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

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


kf = KFold(n_splits=10, shuffle=True)
(x_train, x_test), (y_train, y_test) = np.split(data_x, [int(len(data_x) * 0.8)], axis=0), \
                                       np.split(data_y, [int(len(data_y) * 0.8)], axis=0)

########## Knn ##########
for train_index, test_index in kf.split(data_x):
    kng = KNeighborsRegressor(n_neighbors=5)
    kng.fit(data_x[train_index], data_y[train_index])
    prediction = kng.predict(data_x[test_index])
    pred_score = mean_squared_error(data_y[test_index], prediction)
    cvsorce1.append(pred_score)

########## SVR ##########
for train_index, test_index in kf.split(data_x):
    poly_svr = SVR(kernel='linear')
    poly_svr.fit(data_x[train_index], data_y[train_index])
    poly_predict = poly_svr.predict(data_x[test_index])
    pred_score = mean_squared_error(data_y[test_index], poly_predict)
    cvsorce2.append(pred_score)

# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(8,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.compile(loss='mse', optimizer='adam')
# history = model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=1, validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, batch_size=1, verbose=0)

for train_index, test_index in kf.split(data_x):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(8,)))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    history = model.fit(data_x[train_index], data_y[train_index], batch_size=2, epochs=100, verbose=1, validation_data=(x_test, y_test))
    sorce = model.evaluate(data_x[test_index], data_y[test_index], batch_size=1, verbose=0)
    cvsorce3.append(sorce)
########## Api_Function ##########
# input_shape = (None, 8)
# batch_size = 1
# kernel_size = 3
# pool_size = 2
# dropout = 0.1
# filters = 64
# kernel_size1 = 3
# pool_size1 = 1
# filters1 = 32
#
# for train_index, test_index in kf.split(data_x):
#     model = Sequential()
#
#     inputs = Input(shape=(8,))
#     x = Dense(16, activation='relu')(inputs)
#     x = Dropout(dropout)(x)
#
#     y = Dense(128, activation='relu')(inputs)
#     y = Dropout(dropout)(y)
#     y = Dense(16, activation='relu')(y)
#     y = Dropout(dropout)(y)
#
#     z = Add()([x, y])
#     z = Flatten()(z)
#     z = Dropout(dropout)(z)
#     outputs = Dense(1, activation='relu')(z)
#
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mape'])
#     history = model.fit(data_x[train_index], data_y[train_index], batch_size=1, epochs=1, verbose=1)
#     sorce = model.evaluate(data_x[test_index], data_y[test_index], batch_size=1, verbose=0)
#     cvsorce3.append(sorce)
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

########## Decision Tree ##########
for train_index, test_index in kf.split(data_x):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(data_x[train_index], data_y[train_index])
    pre = clf.predict(data_x[test_index])
    pred_score = mean_squared_error(data_y[test_index], pre)
    cvsorce4.append(pred_score)

########## Random Forest Regression ##########
for train_index, test_index in kf.split(data_x):
    regression = RandomForestRegressor(n_estimators=10, random_state=0)
    regression.fit(data_x[train_index], data_y[train_index])
    pred = regression.predict(data_x[test_index])
    pred_score = mean_squared_error(data_y[test_index], pred)
    cvsorce5.append(pred_score)

# print('score = ', score)
print("Knn = ", np.mean(cvsorce1))
print('SVR = ', np.mean(cvsorce2))
print('MLP = ', np.mean(cvsorce3))
print('Decision Tree = ', np.mean(cvsorce4))
print('Random Forest Regression = ', np.mean(cvsorce5))
