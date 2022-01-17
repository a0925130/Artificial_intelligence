import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Add, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

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

model = Sequential()

inputs = Input(shape=(8,1))
x = Conv1D(filters=16, kernel_size=1, activation='relu', input_shape=(8, 1))(inputs)
x = Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=(8, 1))(x)
x = Dropout(0.1)(x)


y = Dense(16, activation='relu')(inputs)
y = Dense(8, activation='relu')(y)
y = Dropout(0.1)(y)
print(y.shape)
z = Add()([x, y])
z = Flatten()(z)
z = Dropout(0.1)(z)
outputs = Dense(1, activation='relu')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)
sorce = model.evaluate(x_test, y_test, batch_size=1, verbose=0)