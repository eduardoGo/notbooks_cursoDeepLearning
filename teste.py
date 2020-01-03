import numpy
import pandas
import tensorflow
from keras.models import Model
from keras.layers import Dense,Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import random
from random import randrange
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")

def baseline_model_MLP():
    
    inputs = Input(shape=(4,))
    x = Dense(8, activation = 'relu')(inputs)
    outputs = Dense(3, activation='softmax')(x)
    
    #criando o modelo
    model = Model(input=inputs,output=outputs)
    
    # compilar modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model




df = sns.load_dataset("iris")

dataset = df.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


dummy_y = np_utils.to_categorical(encoded_Y)


estimator = KerasClassifier(build_fn=baseline_model_MLP, epochs=200, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))