import keras
from keras.layers import Input, Dense, Dropout, Activation, LSTM
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
import os
import numpy as np

timesteps = 25
number_of_samples = 2500
nb_samples = number_of_samples
frame_row = 52
frame_col = 52

epochs = 30
batch_size = 30

# data = np.random.random((2500, timesteps, frame_row, frame_col, channels))
# label = np.random.random((2500, 1))
#
# X_train = data[0:2000, :]
# y_train = label[0:2000]
#
# X_test = data[2000:, :]
# y_test = label[2000:, :]

train_folder_cw = 'req_npy/train/cw'
train_folder_ccw = 'req_npy/train/ccw'
train_folder_oth = 'req_npy/train/oth'
valid_folder_cw = 'req_npy/valid/cw'
valid_folder_ccw = 'req_npy/valid/ccw'
valid_folder_oth = 'req_npy/valid/oth'
test_folder_cw = 'req_npy/test/cw'
test_folder_ccw = 'req_npy/test/ccw'
test_folder_oth = 'req_npy/test/oth'

X_train = []
y_train = []
for file in os.listdir(train_folder_cw):
    entry = np.load(os.path.join(train_folder_cw, file))
    X_train.append(entry)
    y_train.append(0)
for file in os.listdir(train_folder_ccw):
    entry = np.load(os.path.join(train_folder_ccw, file))
    X_train.append(entry)
    y_train.append(1)
for file in os.listdir(train_folder_oth):
    entry = np.load(os.path.join(train_folder_oth, file))
    X_train.append(entry)
    y_train.append(2)

X_test = []
y_test = []
for file in os.listdir(valid_folder_cw):
    entry = np.load(os.path.join(valid_folder_cw, file))
    X_test.append(entry)
    y_test.append(0)
for file in os.listdir(valid_folder_ccw):
    entry = np.load(os.path.join(valid_folder_ccw, file))
    X_test.append(entry)
    y_test.append(1)
for file in os.listdir(valid_folder_oth):
    entry = np.load(os.path.join(valid_folder_oth, file))
    X_test.append(entry)
    y_test.append(2)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

X_train = np.expand_dims(X_train, axis=4)
X_test = np.expand_dims(X_test, axis=4)

# %%

model = Sequential()

model.add(TimeDistributed(Conv2D(timesteps, (3, 3), padding='same'), input_shape=(25, 52, 52, 1)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Conv2D(32, (5, 5))))
model.add(BatchNormalization())
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.2)))

model.add(TimeDistributed(Conv2D(64, (5, 5))))
model.add(BatchNormalization())
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.2)))

model.add(TimeDistributed(Conv2D(64, (3, 3))))
model.add(BatchNormalization())
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.2)))

model.add(TimeDistributed(Conv2D(32, (3, 3))))
model.add(BatchNormalization())
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.2)))

model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512)))

model.add(TimeDistributed(Dense(35, name="first_dense")))

model.add(LSTM(20, return_sequences=True, name="lstm_layer"))

# %%
model.add(TimeDistributed(Dense(1), name="time_distr_dense_one"))
model.add(GlobalAveragePooling1D(name="global_avg"))

model.add(Dense(3, activation='softmax'))

# %%
optimizer = keras.optimizers.RMSprop(lr=1e-4)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print('Training Started with training set shape {}, test set shape {}'.
      format((X_train.shape, y_train.shape), (X_test.shape, y_test.shape)))
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))
