import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam

# Load CIFAR-10 dataset with custom cache directory
(X_train, y_train), (X_test, y_test) = cifar10.load_data(cache_dir="D:\docs\Study-Materials-BVM-22CP308\sem 6\python ML\machine-learning\lab 6")

# Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Output:

# Epoch 1/5
# 391/391 [==============================] - 61s 153ms/step - loss: 1.5462 - accuracy: 0.4407 - val_loss: 1.2045 - val_accuracy: 0.5743
# Epoch 2/5
# 391/391 [==============================] - 60s 154ms/step - loss: 1.1823 - accuracy: 0.5836 - val_loss: 1.0608 - val_accuracy: 0.6320
# Epoch 3/5
# 391/391 [==============================] - 58s 149ms/step - loss: 1.0454 - accuracy: 0.6339 - val_loss: 0.9721 - val_accuracy: 0.6596
# Epoch 4/5
# 391/391 [==============================] - 64s 163ms/step - loss: 0.9489 - accuracy: 0.6672 - val_loss: 0.8957 - val_accuracy: 0.6905
# Epoch 5/5
# 391/391 [==============================] - 60s 152ms/step - loss: 0.8802 - accuracy: 0.6937 - val_loss: 0.8615 - val_accuracy: 0.7009
# Test loss: 0.8614934682846069
# Test accuracy: 0.7009000182151794
