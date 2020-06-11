import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2

import seaborn as sns


def create_and_train():
    batch_size = 128
    num_classes = 10
    epochs = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = Sequential()
    model.add(Conv2D(18, kernel_size=5, padding='same', activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(36, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(72, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Flatten())
    model.add(Dense(288, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy", metrics=["accuracy"])
    score = model.evaluate(x_test, y_test, verbose=0)
    fit_info = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=15,
                         verbose=1,
                         validation_data=(x_test, y_test))
    print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))
    preds = model.predict_classes(x_test)
    cv2.imshow("blabla", x_test[2])
    print(preds[2])
    cv2.waitKey(0)
    return model


if __name__ == "__main__":
    trained_model = create_and_train()

    trained_model.save("models/mnist_model")
