import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

pre_x, train_x, test_x = data_x[0: int(len(data_x) * 0.5)], data_x[int(len(data_x) * 0.5): int(len(data_x) * 0.8)], \
                         data_x[int(len(data_x) * 0.8)::]
