import archVAE_T as av
import archMLP_T as am
import numpy as np
import readerGraphics_T as rg
from sklearn.preprocessing import MinMaxScaler
from cv2 import split

#%%
def norm_inputs():
    XT = np.load('airfoilsCSTplus.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normXT = scaler.fit_transform(XT)
    return normXT, scaler 

def predictGraphs(Xtest):
    vae = av.ensembleVAE()
    vae.load_weights('vaet_7_1500.weights.h5')
    
    mlp = am.MLP()
    mlp.load_weights('mlp_7A1_1500.weights.h5')
    
    _, scaler = norm_inputs()
    X_Test = scaler.fit_transform(Xtest)
    
    zPred = mlp.predict(X_Test)
    Ygraphs = vae.decoder.predict(zPred)
    return Ygraphs

def predictK(cy, Xtest):
    graphs = predictGraphs(Xtest)
    alpha = np.array([])
    K = np.array([])
    for i in range(len(graphs)):
        graphCy, graphK = split(graphs[i])
        alpha = np.append(alpha, rg.searchAlphawithCy(cy, graphCy))
        K = np.append(K, rg.searchEwithCy(cy, graphK))
    return K, alpha