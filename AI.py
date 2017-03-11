from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import keras.models


def load_model():
    try:
        model = keras.models.load_model('6Nimmt.h5')
    except OSError or ValueError:
        print("No previously trained model found! Creating new one...")
        model = Sequential()
        model.add(Dense(512, init='lecun_uniform', input_shape=(456,)))
        model.add(Activation('relu'))

        model.add(Dense(512, init='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(104, init='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

    return model


def save_model(model):
    model.save('6Nimmt.h5')
