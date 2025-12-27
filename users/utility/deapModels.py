from django.conf import settings
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. Deap models will run in simulation mode.")

def buildDeapModel():
    if not TF_AVAILABLE:
        print("Returning dummy history.")
        return {
            'acc': [0.7, 0.75, 0.8, 0.82],
            'loss': [0.5, 0.45, 0.4, 0.38],
            'val_acc': [0.65, 0.7, 0.72, 0.75],
            'val_loss': [0.55, 0.5, 0.48, 0.45]
        }

    deap = os.path.join(settings.MEDIA_ROOT, 'deapData.csv')
    x = pd.read_csv(deap, low_memory=False)
    print(x.values.shape)
    print(x.values.shape)
    y = x.values[:, 0]
    pixels = x.values[:, 1]
    print(type(pixels))
    print(len(pixels))
    print(len(pixels[0]))
    print(pixels[10][10])
    p = pixels[10].split(' ')
    print(len(p))
    X = np.zeros((pixels.shape[0], 48 * 48))

    for ix in range(X.shape[0]):
        p = pixels[ix].split(' ')
        for iy in range(X.shape[1]):
            X[ix, iy] = int(p[iy])

    temp = X
    # for ix in range(4):
    #     plt.figure(ix)
    #     plt.imshow(temp[ix].reshape((48, 48)), interpolation='none', cmap='gray')
    #     plt.show()
    print(X)
    print(y)
    X = X / 255
    print(X)
    X_train = X[0:30000, :]
    Y_train = y[0:30000]
    print(X_train.shape, Y_train.shape)

    X_test = X[30000:32300, :]
    Y_test = y[30000:32300]
    print(X_test.shape, Y_test.shape)
    X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))
    X_test = X_test.reshape((X_test.shape[0], 48, 48, 1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    print(Y_train.shape)
    print(Y_test.shape)

    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

    datagen.fit(X_train)
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.22))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['acc'])
    batch_size = 64
    # epochs = 40
    epochs = 2

    # steps_per_epoch = len(X) // batch_size
    # validation_steps = len((X_test, Y_test)) // batch_size
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=epochs,
                        shuffle=True,
                        verbose=2)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    return history.history