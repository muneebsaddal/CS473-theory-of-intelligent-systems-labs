import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# input arrays
training_data = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]], "float32")
target_data = np.array([[0],[1],[1],[1],[1],[1],[1],[1]], "float32")

# building or gate model
model = Sequential()
model.add(Dense(2, input_dim=3, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# Compiling model
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, nb_epoch=500, verbose=2)
