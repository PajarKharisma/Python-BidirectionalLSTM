import sys
import os
sys.path.append("../")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# local module and class
import modules.ReadCsv as rc
import modules.PreProcess as pp
import modules.ConfusionMatrix as cm
import numpy as np
from subproc.Vocabulary import *

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
import keras
sys.stderr = stderr

inputPath = '../../input/data_input.csv'
vocabPath = '../../vocabulary/analysis.json'

def crossValidation(TOP_WORDS, dataInput):
    seed = 7
    np.random.seed(seed)

    # split data and label
    X, Y = list(zip(*dataInput))
    Y = np.array(Y)

    # pad Sentences with Keras
    maxDataLenght = 30 
    X = sequence.pad_sequences(X, maxlen=maxDataLenght)

    # set k value for cross validation
    split = 10
    kfold = StratifiedKFold(n_splits=split, shuffle=True, random_state=seed)
    cvscores = []
    for i in range(0,5):
        cvscores.append([])

    # cross validation process
    for train, test in kfold.split(X, Y):
        # build model
        embeddingVectorLength = 32
        model = Sequential()
        model.add(Embedding(TOP_WORDS, embeddingVectorLength, input_length=maxDataLenght))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy',
            cm.f1_m,
            cm.precision_m,
            cm.recall_m
        ])
        model.fit(X[train], Y[train], epochs=10, batch_size=64, verbose=0)
        
        # evaluate the model
        # 1 = accuracy
        # 2 = f1_score
        # 3 = precission
        # 4 = recall
        scores = model.evaluate(X[test], Y[test], verbose=0)
        for i in range(1,5):
            print("%s : %.2f%%" % (model.metrics_names[i], scores[i]*100))
            cvscores[i].append(scores[i]*100)
        print("========================================================")

    print("Mean : ")
    for i in range(1,5):
        metricName = ''
        if i == 1:
            metricName = 'acc'
        elif i == 2:
            metricName = 'f1_m'
        elif i == 3:
            metricName = 'precision_m'
        elif i == 4:
            metricName = 'recall_m'
        print("%s : %.2f%% (+/- %.2f%%)" % (metricName, np.mean(cvscores[i]), np.std(cvscores[i])))

    # show result to matplotlib
    plt.style.use('classic')
    fig, axs = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

    iteration = [*range(1,11)]
    fig.suptitle('Result Data')
    (ax1, ax2), (ax3, ax4) = axs

    ax1.plot(iteration, cvscores[1])
    ax1.set_ylabel('Accuracy (%)')

    ax2.plot(iteration, cvscores[2], 'tab:green')
    ax2.set_ylabel('F1 Measurement (%)')

    ax3.plot(iteration, cvscores[3], 'tab:orange')
    ax3.set_ylabel('Precission (%)')

    ax4.plot(iteration, cvscores[4], 'tab:red')
    ax4.set_ylabel('Recall (%)')

    plt.show()

def splitValidation(TOP_WORDS, dataInput):
    # split dataset for train and test
    train, test = train_test_split(dataInput, test_size=0.1)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)


    # pad Sentences with Keras
    maxDataLenght = 30 
    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    # build model for split
    embeddingVectorLength = 32
    model = Sequential()
    model.add(Embedding(TOP_WORDS, embeddingVectorLength, input_length=maxDataLenght))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # split validation
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, batch_size=64)

def main():
    # load dataset and store it to list
    posData, negData = rc.csv2array(inputPath)

    # preprocess
    posData = pp.getResult(posData)
    negData = pp.getResult(negData)

    # give label to every data. 1 for positive and 0 for negative
    dataLabeled = list(zip(posData, np.ones(len(posData))))
    dataLabeled.extend(list(zip(negData, np.zeros(len(negData)))))

    # bag of words process. Give value to everu word
    TOP_WORDS = 500
    vocab = Vocabulary(TOP_WORDS, vocabPath)
    dataText = [line[0] for line in dataLabeled]
    vocab.PrepareVocabulary(dataText)

    # convert word to vector
    datas, labels = zip(*dataLabeled)
    dataInt = vocab.TransformSentencesToId(datas)
    dataLabeledInt = list(zip(dataInt, labels))

    crossValidation(TOP_WORDS, dataLabeledInt)
    # splitValidation(TOP_WORDS, dataLabeledInt)

def testData():
    dataset = np.loadtxt('../../input/coba.csv', delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]

if __name__ == "__main__":
    main()