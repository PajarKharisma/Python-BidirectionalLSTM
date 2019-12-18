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

import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing import sequence
sys.stderr = stderr

inputPath = '../../input/data_input-Copy.csv'
modelPath = '../../vocabulary/w2v/CBOW/idwiki_word2vec_200.bin'

def nbMethod(dataInput, maxDataLenght):
    # split dataset for train and test
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))
    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    Naive = naive_bayes.MultinomialNB()
    Naive.fit(xTrain, yTrain)
    predictions_NB = Naive.predict(xTrain)
    print("Naive Bayes Accuracy Score -> ", float("%.2f" % round(accuracy_score(predictions_NB, yTrain)*100, 2)))
    print("Naive Bayes Precission Score -> ", float("%.2f" % round(precision_score(predictions_NB, yTrain)*100, 2)))
    print("Naive Bayes Recall Score -> ", float("%.2f" % round(recall_score(predictions_NB, yTrain)*100, 2)))
    print("Naive Bayes F1 Measurement Score -> ", float("%.2f" % round(f1_score(predictions_NB, yTrain)*100, 2)))

def svmMethod(dataInput, maxDataLenght):
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(xTrain, yTrain)
    predictions_SVM = SVM.predict(xTest)
    
    print("SVM Accuracy Score -> ", float("%.2f" % round(accuracy_score(predictions_SVM, yTest)*100, 2)))
    print("SVM Precission Score -> ", float("%.2f" % round(precision_score(predictions_SVM, yTest)*100, 2)))
    print("SVM Recall Score -> ", float("%.2f" % round(recall_score(predictions_SVM, yTest)*100, 2)))
    print("SVM F1 Measurement Score -> ", float("%.2f" % round(f1_score(predictions_SVM, yTest)*100, 2)))

def rfdtMethod(dataInput, maxDataLenght):
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(xTrain, yTrain)
    prediction_rfdt = clf.predict(xTest)

    print("RFDT Accuracy Score -> ", float("%.2f" % round(accuracy_score(prediction_rfdt, yTest)*100, 2)))
    print("RFDT Precission Score -> ", float("%.2f" % round(precision_score(prediction_rfdt, yTest)*100, 2)))
    print("RFDT Recall Score -> ", float("%.2f" % round(recall_score(prediction_rfdt, yTest)*100, 2)))
    print("RFDT F1 Measurement Score -> ", float("%.2f" % round(f1_score(prediction_rfdt, yTest)*100, 2)))

def blrMethod(dataInput, maxDataLenght):
    train, test = train_test_split(dataInput, test_size=0.2)
    xTrain, yTrain = list(zip(*train))
    xTest, yTest = list(zip(*test))

    # convert label to vector
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    xTrain = sequence.pad_sequences(xTrain, maxlen=maxDataLenght) 
    xTest = sequence.pad_sequences(xTest, maxlen=maxDataLenght)

    blr = LogisticRegression()
    blr.fit(xTrain, yTrain)
    prediction_blr = blr.predict(xTest)

    print("BLR Accuracy Score -> ", float("%.2f" % round(accuracy_score(prediction_blr, yTest)*100, 2)))
    print("BLR Precission Score -> ", float("%.2f" % round(precision_score(prediction_blr, yTest)*100, 2)))
    print("BLR Recall Score -> ", float("%.2f" % round(recall_score(prediction_blr, yTest)*100, 2)))
    print("BLR F1 Measurement Score -> ", float("%.2f" % round(f1_score(prediction_blr, yTest)*100, 2)))

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
    maxDataLenght = pp.getMaxPad(datas)

    dataInt = ce.getMeanEmbeddingValue(modelPath, datas)
    dataLabeledInt = list(zip(dataInt, labels))

    start_time = time.time()
    # nbMethod(dataLabeledInt, maxDataLenght)
    # svmMethod(dataLabeledInt, maxDataLenght)
    rfdtMethod(dataLabeledInt, maxDataLenght)
    # blrMethod(dataLabeledInt, maxDataLenght)

    finish_time = time.time()
    print('Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))

if __name__ == "__main__":
    main()