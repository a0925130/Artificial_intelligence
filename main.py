import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Add, Conv1D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

plt.ion()
lossarr = []
accuarr = []
sorce = []
cvsorce = []
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

(x_train, x_test), (y_train, y_test) = np.split(data_x, [int(len(data_x) * 0.8)], axis=0), \
                                       np.split(data_y, [int(len(data_y) * 0.8)], axis=0)
input_shape = (None, 8)
batch_size = 1
kernel_size = 3
pool_size = 2
dropout = 0.1
filters = 64
kernel_size1 = 3
pool_size1 = 1
filters1 = 32

model = Sequential()
# model.add(Dense(5, input_shape=(8,)))
# model.add(Dense(1))
# model.summary()
# model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mape'])
# history =model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=1)
# sorce = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
# print(sorce)

# # for train, test in kfold.split(data_x, data_y):
# # x_res, y_res = SMOTE(random_state=42).fit_resample(x_train, y_train)
#
# # x_train = np.concatenate(x_train, x_res)
# # print("now = ", x_train.shape)

model = Sequential()
inputs = Input(shape=(8,))

x = Dense(64, activation='relu')(inputs)
x = Dropout(dropout)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(dropout)(x)
y = Dense(128, activation='relu')(inputs)
y = Dropout(dropout)(y)
y = Dense(64, activation='relu')(y)
y = Dropout(dropout)(y)
y = Dense(32, activation='relu')(y)
y = Dropout(dropout)(y)

z = Add()([x, y])

z = Flatten()(z)
z = Dropout(dropout)(z)
outputs = Dense(1, activation='relu')(z)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mape'])
history =model.fit(x_train, y_train, batch_size=3, epochs=100, verbose=1)
sorce = model.evaluate(x_test, y_test, batch_size=1, verbose=0)

# for i in range(40):
#     # model.add(Dense(5, input_shape=(13,)))
#     # model.add(Dropout(0.1))
#     # model.add(Dense(5))
#     # model.add(Dropout(0.1))
#     # model.add(Dense(5))
#     # model.add(Dropout(0.1))
#     # model.add(Dense(5, activation='softmax'))
#     # model.summary()
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#
#     history =model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)
#     sorce = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
#     lossarr.append(history.history['loss'][0])
#     accuarr.append(history.history['accuracy'][0])
#     plt.subplot(211)
#     plt.cla()
#     plt.plot(lossarr, label='loss value')
#     plt.title('learning curse')
#     plt.ylabel("loss")
#     plt.xlabel("epochs")
#     plt.legend()
#
#     plt.subplot(212)
#     plt.cla()
#     plt.plot(accuarr, label='accuracy value')
#     plt.title('learning curse')
#     plt.ylabel("accuracy")
#     plt.xlabel("epochs")
#     plt.legend()
#
#     plt.show()
#     plt.pause(0.01)
#     # cvsorce.append(sorce[1]*100)
#     # print(np.mean(cvsorce))
print(sorce)
