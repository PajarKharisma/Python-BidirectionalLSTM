import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("../")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# local module and class
import modules.ReadCsv as rc
import modules.PreProcess as pp
import modules.ConfusionMatrix as cm
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

inputPath = '../../input/data_input.csv'
vocabPath = '../../vocabulary/corpus.json'
modelPath = '../../vocabulary/w2v/Skipgram/idwiki_word2vec_200.bin'
logPath = '../../log/'
logName = 'Percobaan-3.txt'

# parameter yang diuji
NUM_OF_EPOCHS = 20
NUM_OF_ATTRIBUTES = 2500
NUM_OF_NEURONS = 150

# Bidirectional LSTM 1 layer
def crossValidation1(dataInput, embeddingMatrix, maxDataLenght):
    global logFile
    logFile += log.summaryLog(method='Bi-LSTM 1 Layer', numEpochs=NUM_OF_EPOCHS, numAttributes=NUM_OF_ATTRIBUTES, numNeurons=NUM_OF_NEURONS, sg=1) + '\n'
    logFile += '=' * 30 + '\n\n'
    seed = 7
    np.random.seed(seed)

    # split data and label
    X, Y = list(zip(*dataInput))
    Y = np.array(Y)

    # pad Sentences with Keras
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
        embeddingVectorLength = 200
        model = Sequential()
        model.add(Embedding(input_dim=NUM_OF_ATTRIBUTES, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
        # model.add(Flatten())
        model.add(Bidirectional(LSTM(NUM_OF_NEURONS, return_sequences=False), merge_mode="sum"))
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
        logFile += '=' * 30 + '\n'
        print("========================================================")

    logFile += 'Mean : \n'
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
        logFile += "%s : %.2f%% (+/- %.2f%%)\n" % (metricName, np.mean(cvscores[i]), np.std(cvscores[i]))
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

# Bidirectional LSTM 3 layer
def crossValidation2(NUM_OF_ATTRIBUTES, dataInput, embeddingMatrix):
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
        model.add(Embedding(input_dim=NUM_OF_ATTRIBUTES, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
        # model.add(Flatten())
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Bidirectional(LSTM(100, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy',
            cm.f1_m,
            cm.precision_m,
            cm.recall_m
        ])
        # print(model.summary())
        model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), epochs=20, batch_size=256, verbose=0)
        
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

# LSTM 1 layer
def crossValidation3(NUM_OF_ATTRIBUTES, dataInput, embeddingMatrix):
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
        model.add(Embedding(input_dim=NUM_OF_ATTRIBUTES, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
        # model.add(Flatten())
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy',
            cm.f1_m,
            cm.precision_m,
            cm.recall_m
        ])
        # print(model.summary())
        model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), epochs=20, batch_size=256, verbose=0)
        
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

def splitValidation(NUM_OF_ATTRIBUTES, dataInput):
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
    model.add(Embedding(NUM_OF_ATTRIBUTES, embeddingVectorLength, input_length=maxDataLenght))
    model.add(Bidirectional(LSTM(100, return_sequences=False, dropout=0.25, recurrent_dropout=0.001)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy',
            cm.f1_m,
            cm.precision_m,
            cm.recall_m
        ])
    print(model.summary())

    # split validation
    # model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, batch_size=64)
    # model.save('lstm_model.h5')
    hst = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, batch_size=64)

    plt.style.use('classic')
    # fig, axs = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

    # fig.suptitle('Result Data')
    # (ax1, ax2), (ax3, ax4) = axs

    # ax1.plot(hst.history['acc'], label='train')
    # ax1.plot(hst.history['val_acc'], label='validation')
    # ax1.set_xlabel('Accuracy')
    # ax1.legend()

    # ax2.plot(hst.history['f1_m'], label='train')
    # ax2.plot(hst.history['val_f1_m'], label='validation')
    # ax2.set_xlabel('F Measurement')
    # ax2.legend()

    # ax3.plot(hst.history['precision_m'], label='train')
    # ax3.plot(hst.history['val_precision_m'], label='validation')
    # ax3.set_xlabel('Precision')
    # ax3.legend()

    # ax4.plot(hst.history['recall_m'], label='train')
    # ax4.plot(hst.history['val_recall_m'], label='validation')
    # ax4.set_xlabel('Recall')
    # ax4.legend()

    print(np.mean(hst.history['acc']))
    plt.title('Accuracy')
    plt.plot(hst.history['acc'])
    # plt.plot(hst.history['val_loss'], label='validation')
    plt.legend()
    plt.show()
    plt.show()

def nbMethod(NUM_OF_ATTRIBUTES, dataInput):
    # split dataset for train and test
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    maxDataLenght = 30
    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    Naive = naive_bayes.MultinomialNB()
    Naive.fit(xTrain, yTrain)
    predictions_NB = Naive.predict(xTrain)
    print("Naive Bayes Accuracy Score -> ", float("%.2f" % round(accuracy_score(predictions_NB, yTrain)*100, 2)))
    print("Naive Bayes Precission Score -> ", float("%.2f" % round(precision_score(predictions_NB, yTrain)*100, 2)))
    print("Naive Bayes Recall Score -> ", float("%.2f" % round(recall_score(predictions_NB, yTrain)*100, 2)))
    print("Naive Bayes F1 Measurement Score -> ", float("%.2f" % round(f1_score(predictions_NB, yTrain)*100, 2)))

def svmMethod(NUM_OF_ATTRIBUTES, dataInput):
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    maxDataLenght = 30 
    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(xTrain, yTrain)
    predictions_SVM = SVM.predict(xTest)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", float("%.2f" % round(accuracy_score(predictions_SVM, yTest)*100, 2)))
    print("SVM Precission Score -> ", float("%.2f" % round(precision_score(predictions_SVM, yTest)*100, 2)))
    print("SVM Recall Score -> ", float("%.2f" % round(recall_score(predictions_SVM, yTest)*100, 2)))
    print("SVM F1 Measurement Score -> ", float("%.2f" % round(f1_score(predictions_SVM, yTest)*100, 2)))

def createModel(NUM_OF_ATTRIBUTES, dataInput):
    # split dataset for train and test
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)


    # pad Sentences with Keras
    maxDataLenght = 500
    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    # build model for split
    embeddingVectorLength = 512
    model = Sequential()
    model.add(Embedding(NUM_OF_ATTRIBUTES, embeddingVectorLength, input_length=maxDataLenght))
    model.add(Bidirectional(LSTM(100, return_sequences=False, dropout=0.25, recurrent_dropout=0.1)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        cm.f1_m,
        cm.precision_m,
        cm.recall_m
        ])
    print(model.summary())

    # split validation
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, batch_size=32, callbacks=[checkpointer])
    # model.save('lstm_model.h5')

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

    global logFile
    logFile = '================ SUMMARY ================\n'
    logFile += 'Jumlah data NON_HS : ' + str(len(posData)) + '\n'
    logFile += 'Jumlah data HS : ' + str(len(negData)) + '\n'
    logFile += 'Jumlah semua data : ' + str(len(negData) + len(posData)) + '\n'

    datas, labels = zip(*dataLabeled)

    maxDataLenght = pp.getMaxPad(datas)

    vocab = Vocabulary(NUM_OF_ATTRIBUTES, datas, vocabPath)
    vocab.prepareVocabulary()
    corpus = vocab.getVocab(vocabPath)
    dataInt = vocab.transformSentencesToId(datas, vocabPath)

    dataLabeledInt = list(zip(dataInt, labels))

    embeddingMatrix = ce.createEmbeddingMatrix(modelPath, NUM_OF_ATTRIBUTES, corpus)


    start_time = time.time()
    print('Start cross validation...')

    # 1 - Bidirectional LSTM 1 Layer
    # 2 - Bidirectional LSTM 3 Layer
    # 3 - LSTM 1 Layer
    # check(NUM_OF_ATTRIBUTES, dataLabeledInt, embeddingMatrix)
    crossValidation1(dataLabeledInt, embeddingMatrix, maxDataLenght)
    # splitValidation(NUM_OF_ATTRIBUTES, dataLabeledInt)
    # createModel(NUM_OF_ATTRIBUTES, dataLabeledInt)
    # nbMethod(NUM_OF_ATTRIBUTES, dataLabeledInt)
    # svmMethod(NUM_OF_ATTRIBUTES, dataLabeledInt)

    finish_time = time.time()

    logFile += 'Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time))
    log.writeLog(logPath, logName, logFile)

def testData():
    dataset = np.loadtxt('../../input/coba.csv', delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # logFile = open(logPath+'log.txt', "w", encoding='utf-8')
    # for pos in posData:
    #     logFile.write(pos + '\n')
    # logFile.close()
    # print('Finish preprocess')
    # return 0

if __name__ == "__main__":
    main()