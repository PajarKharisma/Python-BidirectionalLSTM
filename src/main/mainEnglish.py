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
import subproc.CreateEmbedding as ce
from subproc.Vocabulary import *

import keras
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

inputPath = '../../input/input.csv'
vocabPath = '../../vocabulary/corpus.json'
modelPath = '../../vocabulary/w2v/CBOW/idwiki_word2vec_200.bin'

def crossValidation1(TOP_WORDS, dataInput, embeddingMatrix):
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
    #     # build model
        embeddingVectorLength = 200
        model = Sequential()
        model.add(Embedding(input_dim=TOP_WORDS, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
        # model.add(Flatten())
        model.add(Bidirectional(LSTM(100, return_sequences=False), merge_mode="sum"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy',
            cm.f1_m,
            cm.precision_m,
            cm.recall_m
        ])
        # print(model.summary())
        model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), epochs=10, batch_size=256, verbose=0)
        
        # evaluate the model
        # 1 = accuracy
        # 2 = f1_score
        # 3 = precission
        # 4 = recall
        # model = load_model('model.weights.best.hdf5', custom_objects={'f1_m':cm.f1_m, 'precision_m':cm.precision_m, 'recall_m':cm.recall_m})
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