import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("../")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# local module and class
import modules.ReadCsv as rc
import modules.PreProcess as pp
import numpy as np
import subproc.CreateEmbedding as ce
import subproc.Loging as log
from subproc.Vocabulary import *

import keras
import time
import matplotlib.pyplot as plt
import keras_metrics as km
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Bidirectional, Flatten, Dropout
from keras import regularizers
# from gensim.models import Word2Vec, KeyedVectors
from numpy import dot
from numpy.linalg import norm
sys.stderr = stderr
# np.seterr(divide='ignore', invalid='ignore')

inputPath = '../../input/data_input-Copy.csv'

# parameter yang diuji
NUM_OF_EPOCHS = 20
NUM_OF_ATTRIBUTES = 500
NUM_OF_NEURONS = 150

def crossValidation1(dataInput, maxDataLength):
    seed = 7
    np.random.seed(seed)

    # split data and label
    X, Y = list(zip(*dataInput))
    Y = np.array(Y)
    
    # set k value for cross validation
    split = 10
    kfold = StratifiedKFold(n_splits=split, shuffle=True, random_state=seed)
    cvscores = []
    for i in range(0,5):
        cvscores.append([])

    # cross validation process
    for train, test in kfold.split(X, Y):
        # build model
        model = Sequential()
        model.add(Embedding(input_dim = NUM_OF_ATTRIBUTES, output_dim = 32, input_length = maxDataLength))
        model.add(Bidirectional(LSTM(NUM_OF_NEURONS, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy',
            km.binary_f1_score(),
            km.binary_precision(),
            km.binary_recall()
        ])
        # print(model.summary())
        model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), epochs=NUM_OF_EPOCHS, batch_size=256, verbose=0)
        
        # evaluate the model
        # 1 = accuracy
        # 2 = f1_score
        # 3 = precission
        # 4 = recall
        # model = load_model('model.weights.best.hdf5', custom_objects={'f1_m':cm.f1_m, 'precision_m':cm.precision_m, 'recall_m':cm.recall_m})
        scores = model.evaluate(X[train], Y[train], verbose=0)
        for i in range(1,5):
            logFile += "%s : %.2f%%\n" % (model.metrics_names[i], scores[i]*100)
            print("%s : %.2f%%" % (model.metrics_names[i], scores[i]*100))
            cvscores[i].append(scores[i]*100)
        print("========================================================")

    print("Mean : ")
    for i in range(1,5):
        metricName = ''
        if i == 1:
            metricName = 'acc'
        elif i == 2:
            metricName = 'f1_score'
        elif i == 3:
            metricName = 'precision'
        elif i == 4:
            metricName = 'recall'
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

def Lstm(dataInput, maxDataLength):
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLength) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLength)

    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(NUM_OF_ATTRIBUTES, embedding_vector_length, input_length=maxDataLength))
    model.add(Bidirectional(LSTM(NUM_OF_NEURONS, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        km.binary_f1_score(),
        km.binary_precision(),
        km.binary_recall()
    ])
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=NUM_OF_EPOCHS, batch_size=512, verbose=0)
    scores = model.evaluate(xTest, yTest, verbose=0)
    for i in range(1,5):
        print("%s : %.2f%%" % (model.metrics_names[i], scores[i]*100))
    print("========================================================")

def main():
    # load dataset and store it to list
    posData, negData = rc.csv2array(inputPath)
    print('Finish read data...')
    # preprocess
    posData = pp.getResult(posData)
    negData = pp.getResult(negData)

    # give label to every data. 1 for positive and 0 for negative
    dataLabeled = list(zip(posData, np.ones(len(posData))))
    dataLabeled.extend(list(zip(negData, np.zeros(len(negData)))))

    datas, labels = zip(*dataLabeled)

    maxDataLength = pp.getMaxPad(datas)
    # tokenizer = Tokenizer(num_words = NUM_OF_ATTRIBUTES)
    # tokenizer.fit_on_texts(datas)
    # dataInt = tokenizer.texts_to_sequences(datas)
    dataInt = [one_hot(data, NUM_OF_ATTRIBUTES) for data in datas]
    dataLabeledInt = list(zip(dataInt, labels))

    start_time = time.time()
    print('Start cross validation...')

    Lstm(dataLabeledInt, maxDataLength)

    finish_time = time.time()
    print('Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))

if __name__ == "__main__":
    main()