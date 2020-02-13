import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


results=[]
for i in range(100):    
    dim =5002
    # load train csv file ([:] means from beginning to end)
    dataframe = pandas.read_csv("train.csv", header=None)
    print(dataframe.head())
    dataset = dataframe.values
    trainX = dataset[:,0:dim]
    trainY = dataset[:,dim]
    
    print("Printing train data")
    print(trainX.shape)
    print(trainY.shape)
    
    
    
    # load validation csv file
    dataframe = pandas.read_csv("test.csv", header=None)
    dataset = dataframe.values
    testX = dataset[:,0:dim]
    testY = dataset[:,dim]
    
    print("Printing validation data")
    print(testX.shape)
    print(testY.shape)
    
    
    # create model
    model = Sequential()
    
    #TODO hidden layer
    model.add(Dense(3000, input_dim=dim, activation='sigmoid'))
    
    model.add(Dense(1500, activation='sigmoid'))
    #TODO output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    #prints model summary
    model.summary()
    
    # Fit the model
    model.fit(trainX, trainY, epochs=100, batch_size=4)
    
    # evaluate the model
    scores = model.evaluate(testX, testY)
    
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("Test number: %d" % (i+1))
    results.append(scores[1]*100)

result = 0    
maximum = 0
minimum = 100
for i in range(len(results)):
    result = result + results[i]
    if maximum<results[i]:
        maximum = results[i]
    if minimum>results[i]:
        minimum = results[i]
result = result/len(results)    

print("\n\nFINAL RESULT\n")
print("acc: %.2f%%" % result)
print("min: %.2f%%" % minimum)
print("max: %.2f%%" % maximum)