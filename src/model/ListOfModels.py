import keras
import keras_metrics as km
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Bidirectional, Flatten, Dropout, GRU
from keras import regularizers

def gruModel(embeddingMatrix, maxDataLenght, embeddingVectorLength, numAttributes, numNeurons):
    model = Sequential()
    model.add(Embedding(input_dim=numAttributes, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
    model.add(GRU(numNeurons, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        km.binary_f1_score(),
        km.binary_precision(),
        km.binary_recall()
    ])
    
    return model
    
def biLstmModel1(embeddingMatrix, maxDataLenght, embeddingVectorLength, numAttributes, numNeurons):
    model = Sequential()
    model.add(Embedding(input_dim=numAttributes, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
    model.add(Bidirectional(LSTM(numNeurons, return_sequences=False), merge_mode="sum"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        km.binary_f1_score(),
        km.binary_precision(),
        km.binary_recall()
    ])

    return model

def biLstmModel3(embeddingMatrix, maxDataLenght, embeddingVectorLength, numAttributes, numNeurons):
    model = Sequential()
    model.add(Embedding(input_dim=numAttributes, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
    model.add(Bidirectional(LSTM(numNeurons, return_sequences=True), merge_mode="sum"))
    model.add(Bidirectional(LSTM(numNeurons, return_sequences=True), merge_mode="sum"))
    model.add(Bidirectional(LSTM(numNeurons, return_sequences=False), merge_mode="sum"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        km.binary_f1_score(),
        km.binary_precision(),
        km.binary_recall()
    ])

    return model

def lstmModel1(embeddingMatrix, maxDataLenght, embeddingVectorLength, numAttributes, numNeurons):
    model = Sequential()
    model.add(Embedding(input_dim=numAttributes, output_dim=embeddingVectorLength, weights=[embeddingMatrix], input_length=maxDataLenght, trainable=False))
    model.add(LSTM(numNeurons, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        km.binary_f1_score(),
        km.binary_precision(),
        km.binary_recall()
    ])

    return model