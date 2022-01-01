import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.keras.layers import Dense, Dropout, Input, Add, Flatten, Conv1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

data_x = []
data_y = []
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

pre_x = data_x[0: 5000]
pre_y = data_y[0: 5000]

train_x = data_x[5000: 9000]
train_y = data_y[5000: 9000]

test_x = data_x[9000:]
test_y = data_y[9000:]

# base_model = Sequential()
# base_model.add(Dense(128, input_shape=(8,)))
# base_model.add(Dense(64))
# base_model.add(Dense(32))
# base_model.add(Dense(16))
# base_model.add(Dense(1))
#
# base_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# base_model.fit(pre_x, pre_y, epochs=20, batch_size=5)
base_model = Sequential()

inputs = Input(shape=(8, 1))

x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
x = Conv1D(filters=64, kernel_size=1, activation='relu')(x)
x = Conv1D(filters=64, kernel_size=1, activation='relu')(x)
x = Dropout(0.2)(x)

y = Dense(64, activation='relu')(inputs)
y = Dropout(0.2)(y)
y = Dense(64, activation='relu')(y)
y = Dropout(0.2)(y)
#
z = Add()([x, y])
z = Flatten()(z)
z = Dense(8, activation='relu')(z)
outputs = Dense(1, activation='relu')(z)

base_model = tf.keras.Model(inputs=inputs, outputs=outputs)
base_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = base_model.fit(pre_x, pre_y, batch_size=5, epochs=20, verbose=1)
socre = base_model.predict(test_x)
r21 = r2_score(socre, test_y)

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
model.fit(train_x, train_y, epochs=20, batch_size=5)

for layer in model.layers[:3]:
    layer.trainable = False
for layer in model.layers[3:]:
    layer.trainable = True

model.summary()
initial_learning_rate = 1e-3
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='mse', metrics=["mse"])
model.fit(train_x, train_y, batch_size=5, epochs=1000)
socre = model.predict(test_x)
r2 = r2_score(socre, test_y)
print(r21)
print(r2)
