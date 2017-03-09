from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import keras.models

def loadModel():
    try:
        model = keras.models.load_model('6Nimmt.h5')
    except OSError or ValueError:
        model = Sequential()
        model.add(Dense(512, init='lecun_uniform', input_shape=(10,)))
        model.add(Activation('relu'))
    
        model.add(Dense(512, init='lecun_uniform'))
        model.add(Activation('relu'))
    
        model.add(Dense(104, init='lecun_uniform'))
        model.add(Activation('linear'))
    
        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)
        
    return model


def saveModel(model):
    model.save('6Nimmt.h5')