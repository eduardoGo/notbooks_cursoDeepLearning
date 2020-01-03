import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import random
from random import randrange




seed = 7
numpy.random.seed(seed)


dataframe = sns.load_dataset("iris")

#dataframe = pandas.read_csv("/home/pedro/Documents/git/ML/aulas_notbook/perception/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

print(dataframe.head(4))