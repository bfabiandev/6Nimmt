from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import keras.models


class AI:
    @staticmethod
    def load_model():
        try:
            model = keras.models.load_model('6Nimmt.h5')
        except OSError or ValueError:
            print("No previously trained model found! Creating new one...")
            model = Sequential()
            model.add(Dense(1024, init='lecun_uniform', input_shape=(456,), activation='relu'))
#            model.add(Dropout(0.2))

            model.add(Dense(1024, init='lecun_uniform', activation='relu'))
#            model.add(Dropout(0.2))

            model.add(Dense(104, init='lecun_uniform', activation='relu'))

            rms = RMSprop()
            model.compile(loss='mse', optimizer=rms)

        return model

    @staticmethod
    def save_model(model):
        model.save('6Nimmt.h5')
