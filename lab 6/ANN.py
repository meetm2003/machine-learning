from matplotlib import pyplot as plt
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(2017)
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils import to_categorical
from IPython.display import SVG

nb_classes = 10 # class size
# flatten 28*28 images to a 784 vector for each image
input_unit_size = 28*28

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], input_unit_size)
X_test  = X_test.reshape(X_test.shape[0], input_unit_size)
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')

# Scale the values by dividing 255 i.e., means foreground (black)
X_train /= 255
X_test  /= 255

# One-hot encoding
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# create model
model = Sequential()
model.add(Dense(input_unit_size, input_dim=input_unit_size, activation='relu'))
model.add(Dense(nb_classes, kernel_initializer='normal', activation='softmax'))
 
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# model training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=500, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))

# Output:
# Epoch 1/5
# 120/120 - 3s - 24ms/step - accuracy: 0.8972 - loss: 0.3744 - val_accuracy: 0.9459 - val_loss: 0.1892
# Epoch 2/5
# 120/120 - 2s - 17ms/step - accuracy: 0.9561 - loss: 0.1573 - val_accuracy: 0.9587 - val_loss: 0.1357
# Epoch 3/5
# 120/120 - 2s - 16ms/step - accuracy: 0.9704 - loss: 0.1079 - val_accuracy: 0.9680 - val_loss: 0.1057
# Epoch 4/5
# 120/120 - 2s - 16ms/step - accuracy: 0.9776 - loss: 0.0796 - val_accuracy: 0.9747 - val_loss: 0.0872
# Epoch 5/5
# 120/120 - 2s - 16ms/step - accuracy: 0.9826 - loss: 0.0629 - val_accuracy: 0.9760 - val_loss: 0.0778
# Error: 2.40%