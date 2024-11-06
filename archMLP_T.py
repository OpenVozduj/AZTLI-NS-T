from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import RootMeanSquaredError

def MLP(latent_dim=7):
    model_mlp = Sequential()
    model_mlp.add(Input(shape=(14,)))
    model_mlp.add(Dense(64, activation='tanh'))
    model_mlp.add(Dense(latent_dim, activation='sigmoid'))
    
    model_mlp.compile(optimizer='adam', loss='mean_squared_error', 
                      metrics=[RootMeanSquaredError()])
    return model_mlp
